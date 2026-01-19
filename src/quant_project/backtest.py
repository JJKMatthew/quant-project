"""回测引擎

负责执行选股策略和交易策略，计算回测结果。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .strategies.base import BaseSelectionStrategy, BaseTradingStrategy


@dataclass
class BacktestConfig:
    """回测配置"""

    start_date: str  # 格式: YYYY-MM-DD
    end_date: str  # 格式: YYYY-MM-DD

    # 策略
    selection_strategy: BaseSelectionStrategy  # 选股策略
    trading_strategy: BaseTradingStrategy  # 交易策略

    # 资金管理
    initial_capital: float = 100000.0  # 初始资金
    max_stocks: Optional[int] = 10  # 最多持仓股票数
    position_size: str = "equal_weight"  # 仓位分配方式: equal_weight(等权) 或 equal_amount(等额)

    # 选股频率
    rebalance_frequency: str = "monthly"  # 选股频率: weekly, monthly, quarterly

    # 费用
    commission_rate: float = 0.0003  # 佣金费率（万三）
    min_commission: float = 5.0  # 最低佣金

    # 滑点
    slippage: float = 0.0  # 滑点百分比

    # 基准
    benchmark: Optional[str] = "000300"  # 基准代码，如沪深300


@dataclass
class Trade:
    """交易记录"""

    date: pd.Timestamp
    symbol: str
    action: str  # buy, sell
    price: float
    shares: int
    amount: float
    commission: float


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is not None and self.entry_price > 0:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        return None


@dataclass
class BacktestResult:
    """回测结果"""

    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_curve: Optional[pd.DataFrame] = None

    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "config": str(self.config),
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "num_trades": len(self.trades),
            "num_positions": len(self.positions),
        }


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """运行回测

        Args:
            data: 包含所有股票历史数据的 DataFrame
                  必须包含列: date, symbol, open, high, low, close, volume, amount

        Returns:
            回测结果
        """
        # 初始化结果
        result = BacktestResult(config=self.config)

        # 确保数据按日期排序
        data = data.copy().sort_values("date")

        # 获取交易日期范围
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)

        # 获取所有交易日期
        all_dates = sorted(data["date"].unique())
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        if not trading_dates:
            return result

        # 初始化账户状态
        cash = self.config.initial_capital
        holdings: Dict[str, int] = {}  # symbol -> shares
        entry_prices: Dict[str, float] = {}  # symbol -> entry price
        entry_dates: Dict[str, pd.Timestamp] = {}  # symbol -> entry date

        # 生成调仓日期
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        # 记录账户价值曲线
        equity_curve = []

        # 记录交易
        trades: List[Trade] = []
        positions: List[Position] = []

        for current_date in trading_dates:
            # 获取当天的价格数据
            price_data = data[data["date"] == current_date].copy()

            if price_data.empty:
                # 没有数据，使用前一天的价格计算净值
                if equity_curve:
                    prev_value = equity_curve[-1]["equity"]
                    equity_curve.append({
                        "date": current_date,
                        "cash": cash,
                        "holdings_value": prev_value - cash,
                        "equity": prev_value,
                    })
                continue

            # 检查是否需要调仓
            is_rebalance = current_date in rebalance_dates

            # 先处理交易策略的退出信号
            if holdings and (is_rebalance or current_date == end_date):
                for symbol in list(holdings.keys()):
                    if holdings[symbol] > 0:
                        stock_data = data[data["symbol"] == symbol].sort_values("date")
                        stock_data = stock_data[stock_data["date"] <= current_date]

                        if len(stock_data) >= 2:
                            # 生成交易信号
                            entry_date = entry_dates[symbol]
                            signals = self.config.trading_strategy.generate_signals(
                                stock_data, entry_date, current_date
                            )

                            # 检查最后一天的信号
                            last_signal = signals[signals["date"] == current_date]
                            if not last_signal.empty:
                                signal = last_signal.iloc[0]["signal"]
                                if signal in (0, -1):  # SELL 或 EXIT
                                    # 卖出
                                    sell_price = self._get_price(price_data, symbol, "open")
                                    if sell_price > 0:
                                        sell_amount = holdings[symbol] * sell_price
                                        commission = self._calculate_commission(sell_amount)
                                        cash += sell_amount - commission

                                        # 记录交易
                                        trades.append(Trade(
                                            date=current_date,
                                            symbol=symbol,
                                            action="sell",
                                            price=sell_price,
                                            shares=holdings[symbol],
                                            amount=sell_amount,
                                            commission=commission,
                                        ))

                                        # 记录持仓
                                        positions.append(Position(
                                            symbol=symbol,
                                            entry_date=entry_date,
                                            entry_price=entry_prices[symbol],
                                            shares=holdings[symbol],
                                            exit_date=current_date,
                                            exit_price=sell_price,
                                        ))

                                        holdings[symbol] = 0

            # 调仓：卖出当前持仓
            if is_rebalance and holdings:
                for symbol in list(holdings.keys()):
                    if holdings[symbol] > 0:
                        sell_price = self._get_price(price_data, symbol, "open")
                        if sell_price > 0:
                            sell_amount = holdings[symbol] * sell_price
                            commission = self._calculate_commission(sell_amount)
                            cash += sell_amount - commission

                            trades.append(Trade(
                                date=current_date,
                                symbol=symbol,
                                action="sell",
                                price=sell_price,
                                shares=holdings[symbol],
                                amount=sell_amount,
                                commission=commission,
                            ))

                            positions.append(Position(
                                symbol=symbol,
                                entry_date=entry_dates[symbol],
                                entry_price=entry_prices[symbol],
                                shares=holdings[symbol],
                                exit_date=current_date,
                                exit_price=sell_price,
                            ))

                            holdings[symbol] = 0
                            del entry_prices[symbol]
                            del entry_dates[symbol]

            # 调仓：选股并买入
            if is_rebalance:
                selected = self.config.selection_strategy.select(
                    data, current_date, self.config.max_stocks
                )

                if selected and cash > 0:
                    # 计算每只股票的买入金额
                    num_stocks = min(len(selected), self.config.max_stocks or len(selected))
                    if self.config.position_size == "equal_weight":
                        # 等权重分配
                        per_stock_amount = cash / num_stocks
                    else:
                        # 等额分配（实际与等权重相同）
                        per_stock_amount = cash / num_stocks

                    # 买入
                    for symbol in selected[:num_stocks]:
                        buy_price = self._get_price(price_data, symbol, "open")
                        if buy_price > 0:
                            # 计算买入股数（考虑滑点）
                            actual_price = buy_price * (1 + self.config.slippage)
                            shares = int(per_stock_amount / actual_price)

                            if shares > 0:
                                buy_amount = shares * actual_price
                                commission = self._calculate_commission(buy_amount)
                                cash -= buy_amount + commission

                                trades.append(Trade(
                                    date=current_date,
                                    symbol=symbol,
                                    action="buy",
                                    price=actual_price,
                                    shares=shares,
                                    amount=buy_amount,
                                    commission=commission,
                                ))

                                holdings[symbol] = shares
                                entry_prices[symbol] = actual_price
                                entry_dates[symbol] = current_date

            # 计算当前持仓价值
            holdings_value = 0.0
            for symbol, shares in holdings.items():
                price = self._get_price(price_data, symbol, "close")
                if price > 0:
                    holdings_value += shares * price

            # 记录账户价值
            equity = cash + holdings_value
            equity_curve.append({
                "date": current_date,
                "cash": cash,
                "holdings_value": holdings_value,
                "equity": equity,
            })

        # 处理最后未平仓的持仓
        final_date = trading_dates[-1]
        final_price_data = data[data["date"] == final_date]

        for symbol, shares in list(holdings.items()):
            if shares > 0:
                sell_price = self._get_price(final_price_data, symbol, "close")
                if sell_price > 0:
                    positions.append(Position(
                        symbol=symbol,
                        entry_date=entry_dates[symbol],
                        entry_price=entry_prices[symbol],
                        shares=shares,
                        exit_date=final_date,
                        exit_price=sell_price,
                    ))

        # 构建结果
        result.trades = trades
        result.positions = positions
        result.equity_curve = pd.DataFrame(equity_curve)

        # 计算绩效指标
        self._calculate_metrics(result)

        return result

    def _get_price(
        self,
        price_data: pd.DataFrame,
        symbol: str,
        price_col: str = "close"
    ) -> float:
        """获取股票价格"""
        row = price_data[price_data["symbol"] == symbol]
        if not row.empty:
            return float(row.iloc[0][price_col])
        return 0.0

    def _calculate_commission(self, amount: float) -> float:
        """计算佣金"""
        commission = amount * self.config.commission_rate
        return max(commission, self.config.min_commission)

    def _get_rebalance_dates(self, trading_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """获取调仓日期"""
        if not trading_dates:
            return []

        rebalance_dates = []

        if self.config.rebalance_frequency == "weekly":
            # 每周第一个交易日调仓
            for date in trading_dates:
                if date.weekday() == 0:  # 周一
                    rebalance_dates.append(date)
        elif self.config.rebalance_frequency == "monthly":
            # 每月第一个交易日调仓
            current_month = None
            for date in trading_dates:
                if date.month != current_month:
                    rebalance_dates.append(date)
                    current_month = date.month
        elif self.config.rebalance_frequency == "quarterly":
            # 每季度第一个交易日调仓
            current_quarter = None
            for date in trading_dates:
                quarter = (date.month - 1) // 3
                if quarter != current_quarter:
                    rebalance_dates.append(date)
                    current_quarter = quarter
        else:
            # 默认只在开始时选股一次
            if trading_dates:
                rebalance_dates.append(trading_dates[0])

        return rebalance_dates

    def _calculate_metrics(self, result: BacktestResult) -> None:
        """计算绩效指标"""
        if result.equity_curve.empty:
            return

        curve = result.equity_curve

        # 总收益率
        initial_value = curve.iloc[0]["equity"]
        final_value = curve.iloc[-1]["equity"]
        result.total_return = (final_value - initial_value) / initial_value

        # 年化收益率
        days = (curve.iloc[-1]["date"] - curve.iloc[0]["date"]).days
        if days > 0:
            years = days / 365.0
            if years > 0:
                result.annual_return = (final_value / initial_value) ** (1 / years) - 1

        # 最大回撤
        equity_values = curve["equity"].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        result.max_drawdown = drawdown.min()

        # 夏普比率（简化版）
        daily_returns = np.diff(equity_values) / equity_values[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1:
            # 使用年化无风险利率为 0
            sharpe = np.mean(daily_returns) / np.std(daily_returns)
            # 年化：乘以 sqrt(252)
            result.sharpe_ratio = sharpe * np.sqrt(252)

        # 胜率
        win_trades = [p for p in result.positions if p.return_pct is not None and p.return_pct > 0]
        total_trades = len([p for p in result.positions if p.return_pct is not None])
        if total_trades > 0:
            result.win_rate = len(win_trades) / total_trades
