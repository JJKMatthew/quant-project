"""回测引擎

负责执行选股策略和交易策略，计算回测结果。

核心流程：
1. 数据准备：获取所有股票的历史数据
2. 回测循环：逐日模拟交易
3. 选股：在调仓日调用选股策略
4. 交易：根据交易策略生成买卖信号
5. 绩效计算：计算各种回测指标

注意事项：
- 数据必须包含列: date, symbol, open, high, low, close, volume, amount
- 回测使用开盘价交易，收盘价计算持仓价值
- 支持多种调仓频率: once, weekly, monthly, quarterly
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .strategies.base import BaseSelectionStrategy, BaseTradingStrategy


@dataclass
class BacktestConfig:
    """回测配置

    配置项说明：
    - start_date/end_date: 回测日期范围
    - selection_strategy: 决定"买什么"的策略
    - trading_strategy: 决定"何时买卖"的策略
    - initial_capital: 初始资金量
    - max_stocks: 最大同时持仓股票数量
    - position_size: 仓位分配方式（目前仅支持 equal_weight）
    - rebalance_frequency: 调仓频率
    - commission_rate: 交易佣金率，默认万三（0.0003）
    - slippage: 模拟滑点，买入价+滑点，卖出价-滑点
    """

    start_date: str  # 格式: YYYY-MM-DD
    end_date: str  # 格式: YYYY-MM-DD

    # 策略
    selection_strategy: BaseSelectionStrategy  # 选股策略：决定买什么
    trading_strategy: BaseTradingStrategy  # 交易策略：决定何时买卖

    # 资金管理
    initial_capital: float = 100000.0  # 初始资金
    max_stocks: Optional[int] = 10  # 最多持仓股票数（分散风险）
    position_size: str = "equal_weight"  # 仓位分配: equal_weight(等权重分配)

    # 选股频率
    rebalance_frequency: str = "monthly"  # 调仓频率: once/weekly/monthly/quarterly

    # 费用
    commission_rate: float = 0.0003  # 佣金费率（万三，即0.03%）
    min_commission: float = 5.0  # 最低佣金（部分券商每笔交易最低5元）

    # 滑点
    slippage: float = 0.0  # 滑点百分比（模拟实际交易时的价格偏差）

    # 基准
    benchmark: Optional[str] = "000300"  # 基准代码，如沪深300（暂未实现对比）


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
    """回测引擎

    核心方法 run() 的执行流程：
    1. 初始化账户状态（现金、持仓、记录）
    2. 确定调仓日期
    3. 逐日循环：
       a. 检查交易策略的卖出信号
       b. 调仓日：清空当前持仓
       c. 调仓日：选股并买入新股票
       d. 计算当日净值
    4. 计算绩效指标
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """运行回测

        Args:
            data: 包含所有股票历史数据的 DataFrame
                  必须包含列: date, symbol, open, high, low, close, volume, amount

        Returns:
            回测结果 BacktestResult，包含交易记录、持仓记录和绩效指标

        执行逻辑：
        - 开盘价用于模拟实际成交（更保守）
        - 收盘价用于计算持仓市值
        - 调仓日先清空再选股买入
        - 每日记录净值用于计算绩效指标
        """
        # ========== 第一步：初始化 ==========
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

        # ========== 第二步：初始化账户状态 ==========
        cash = self.config.initial_capital  # 可用现金
        holdings: Dict[str, int] = {}  # symbol -> shares，持仓数量
        entry_prices: Dict[str, float] = {}  # symbol -> entry price，买入价格
        entry_dates: Dict[str, pd.Timestamp] = {}  # symbol -> entry date，买入日期

        # 生成调仓日期
        rebalance_dates = self._get_rebalance_dates(trading_dates)

        # 记录账户价值曲线
        equity_curve = []

        # 记录交易和持仓
        trades: List[Trade] = []
        positions: List[Position] = []

        # ========== 第三步：逐日回测循环 ==========
        for current_date in trading_dates:
            # 获取当天的价格数据
            price_data = data[data["date"] == current_date].copy()

            if price_data.empty:
                # 当天没有数据（如节假日），继承前一天的净值
                if equity_curve:
                    prev_value = equity_curve[-1]["equity"]
                    equity_curve.append({
                        "date": current_date,
                        "cash": cash,
                        "holdings_value": prev_value - cash,
                        "equity": prev_value,
                    })
                continue

            # 判断是否是调仓日
            is_rebalance = current_date in rebalance_dates

            # ---------- 步骤1：处理交易策略的卖出信号 ----------
            # 注意：先处理信号卖出，再处理调仓日清仓
            if holdings and (is_rebalance or current_date == end_date):
                for symbol in list(holdings.keys()):
                    if holdings[symbol] > 0:
                        stock_data = data[data["symbol"] == symbol].sort_values("date")
                        stock_data = stock_data[stock_data["date"] <= current_date]

                        if len(stock_data) >= 2:
                            # 调用交易策略生成买卖信号
                            entry_date = entry_dates[symbol]
                            signals = self.config.trading_strategy.generate_signals(
                                stock_data, entry_date, current_date
                            )

                            # 检查当日的信号，如果为 SELL(0) 或 EXIT(-1) 则卖出
                            last_signal = signals[signals["date"] == current_date]
                            if not last_signal.empty:
                                signal = last_signal.iloc[0]["signal"]
                                if signal in (0, -1):  # SELL 或 EXIT
                                    # 执行卖出（用开盘价，更保守）
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

                                        # 记录完成持仓（用于计算胜率等指标）
                                        positions.append(Position(
                                            symbol=symbol,
                                            entry_date=entry_date,
                                            entry_price=entry_prices[symbol],
                                            shares=holdings[symbol],
                                            exit_date=current_date,
                                            exit_price=sell_price,
                                        ))

                                        # 清空持仓
                                        holdings[symbol] = 0

            # ---------- 步骤2：调仓日清空当前持仓 ----------
            # 调仓日强制卖出所有持仓，重新选股
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

            # ---------- 步骤3：调仓日选股并买入 ----------
            if is_rebalance:
                # 调用选股策略获取符合条件的股票
                selected = self.config.selection_strategy.select(
                    data, current_date, self.config.max_stocks
                )

                if selected and cash > 0:
                    # 等权重分配：每只股票投入相同资金
                    num_stocks = min(len(selected), self.config.max_stocks or len(selected))
                    per_stock_amount = cash / num_stocks

                    # 逐个买入
                    for symbol in selected[:num_stocks]:
                        buy_price = self._get_price(price_data, symbol, "open")
                        if buy_price > 0:
                            # 计算买入股数（考虑滑点，买入价更高）
                            actual_price = buy_price * (1 + self.config.slippage)
                            shares = int(per_stock_amount / actual_price)

                            if shares > 0:
                                buy_amount = shares * actual_price
                                commission = self._calculate_commission(buy_amount)
                                cash -= buy_amount + commission

                                # 记录买入交易
                                trades.append(Trade(
                                    date=current_date,
                                    symbol=symbol,
                                    action="buy",
                                    price=actual_price,
                                    shares=shares,
                                    amount=buy_amount,
                                    commission=commission,
                                ))

                                # 更新持仓状态
                                holdings[symbol] = shares
                                entry_prices[symbol] = actual_price
                                entry_dates[symbol] = current_date

            # ---------- 步骤4：计算当日净值 ----------
            holdings_value = 0.0
            for symbol, shares in holdings.items():
                price = self._get_price(price_data, symbol, "close")
                if price > 0:
                    holdings_value += shares * price

            equity = cash + holdings_value
            equity_curve.append({
                "date": current_date,
                "cash": cash,
                "holdings_value": holdings_value,
                "equity": equity,
            })

        # ========== 第四步：处理最后未平仓的持仓 ==========
        # 回测结束时，所有未平仓持仓按收盘价结算
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

        # ========== 第五步：构建结果并计算指标 ==========
        result.trades = trades
        result.positions = positions
        result.equity_curve = pd.DataFrame(equity_curve)

        # 计算绩效指标（总收益率、年化收益、最大回撤、夏普比率、胜率）
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
        """计算绩效指标

        计算的指标：
        - total_return: 总收益率 = (期末净值 - 期初净值) / 期初净值
        - annual_return: 年化收益率 = (期末净值/期初净值)^(1/年数) - 1
        - max_drawdown: 最大回撤 = min((净值 - 历史最高净值) / 历史最高净值)
        - sharpe_ratio: 夏普比率 = (日收益率均值 / 日收益率标准差) * sqrt(252)
        - win_rate: 胜率 = 盈利持仓数 / 总持仓数
        """
        if result.equity_curve.empty:
            return

        curve = result.equity_curve

        # 总收益率：从期初到期末的整体收益
        initial_value = curve.iloc[0]["equity"]
        final_value = curve.iloc[-1]["equity"]
        result.total_return = (final_value - initial_value) / initial_value

        # 年化收益率：将总收益率年化，便于不同时间周期的策略对比
        days = (curve.iloc[-1]["date"] - curve.iloc[0]["date"]).days
        if days > 0:
            years = days / 365.0
            if years > 0:
                # 使用复利公式年化：年化收益 = (1 + 总收益)^(1/年数) - 1
                result.annual_return = (final_value / initial_value) ** (1 / years) - 1

        # 最大回撤：净值从峰值下跌的最大幅度（通常为负数）
        # peak 记录历史最高点，drawdown 计算当前点相对峰值的回撤
        equity_values = curve["equity"].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        result.max_drawdown = drawdown.min()  # 取最小值（最大回撤）

        # 夏普比率：衡量单位风险的收益，数值越高越好
        # 简化版假设无风险利率为0
        daily_returns = np.diff(equity_values) / equity_values[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        if len(daily_returns) > 1:
            sharpe = np.mean(daily_returns) / np.std(daily_returns)
            # 年化：年交易日约252天
            result.sharpe_ratio = sharpe * np.sqrt(252)

        # 胜率：盈利交易占比，衡量策略的准确度
        win_trades = [p for p in result.positions if p.return_pct is not None and p.return_pct > 0]
        total_trades = len([p for p in result.positions if p.return_pct is not None])
        if total_trades > 0:
            result.win_rate = len(win_trades) / total_trades
