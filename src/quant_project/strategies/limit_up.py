"""涨停板打板策略实现

高风险打板玩法，专注于捕捉临近涨停的股票。
目标：大概率吃到冲涨停，短线快进快出。
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from .base import BaseSelectionStrategy, BaseTradingStrategy, Signal


def is_main_board(symbol: str) -> bool:
    """判断是否为主板股票（排除科创板和创业板）

    主板股票代码规则：
    - 上海主板：600xxx、601xxx、603xxx、605xxx
    - 深圳主板：000xxx、001xxx

    排除：
    - 科创板：688xxx
    - 创业板：300xxx
    - 北交所：43xxxx、83xxxx、87xxxx

    Args:
        symbol: 股票代码，如 "sh600000" 或 "sz000001"

    Returns:
        True 为主板股票，False 为其他板块
    """
    code = symbol[2:]  # 去掉 "sh" 或 "sz" 前缀

    # 科创板：688开头
    if code.startswith("688"):
        return False
    # 创业板：300开头
    if code.startswith("300"):
        return False
    # 北交所：43、83、87开头
    if code.startswith("43") or code.startswith("83") or code.startswith("87"):
        return False

    # 主板股票
    if code.startswith("60"):  # 上海主板
        return True
    if code.startswith("00") or code.startswith("001"):  # 深圳主板
        return True

    return False


class LimitUpSelectionStrategy(BaseSelectionStrategy):
    """涨停板打板选股策略

    专注于捕捉临近涨停的股票，适合高风险打板玩法。

    选股思路：
    1. 当天涨幅筛选：通常选择涨幅在7%-9.5%之间的股票（接近10%涨停板）
    2. 量能放大：成交量显著放大，量比大于1.5-2.0
    3. 突破确认：价格突破近期高点
    4. 流动性保证：成交额满足最低要求
    5. 价格适中：排除低价垃圾股和超高价股
    6. 主板限制：只选主板股票，排除ST、科创板、创业板

    参数说明：
    - min_pct_chg: 最低涨幅（%），默认7%，接近涨停但还未封板
    - max_pct_chg: 最高涨幅（%），默认9.8%，避免追已封板的
    - min_volume_ratio: 最小量比，默认1.5，要求成交量放大
    - min_amount: 最小成交额（万元），默认5000万，保证流动性
    - min_price: 最低价格，默认3元
    - max_price: 最高价格，默认50元
    - min_turnover: 最小换手率（%），默认2%
    - max_turnover: 最大换手率（%），默认25%，避免过度炒作
    - breakout_days: 突破回溯天数，默认10天
    - use_macd: 是否使用MACD确认，默认True
    """

    def __init__(
        self,
        # 涨幅筛选（核心）
        min_pct_chg: float = 7.0,       # 最低涨幅7%（接近涨停）
        max_pct_chg: float = 9.8,       # 最高涨幅9.8%（还未完全封板）

        # 量能筛选
        min_volume_ratio: float = 1.5,  # 最小量比1.5倍
        volume_ratio_days: int = 5,     # 量比回溯天数

        # 基础过滤
        min_price: float = 3.0,         # 最低价格
        max_price: float = 50.0,        # 最高价格
        min_amount: float = 5000,       # 最小成交额（万元）
        min_turnover: float = 2.0,      # 最小换手率（%）
        max_turnover: float = 25.0,     # 最大换手率（%）

        # 突破筛选
        use_breakout: bool = True,      # 是否使用突破过滤
        breakout_days: int = 10,        # 突破回溯天数

        # 技术指标
        use_macd: bool = True,          # 是否使用MACD
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.min_pct_chg = min_pct_chg
        self.max_pct_chg = max_pct_chg
        self.min_volume_ratio = min_volume_ratio
        self.volume_ratio_days = volume_ratio_days
        self.min_price = min_price
        self.max_price = max_price
        self.min_amount = min_amount
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover
        self.use_breakout = use_breakout
        self.breakout_days = breakout_days
        self.use_macd = use_macd
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def _calculate_volume_ratio(
        self, data: pd.DataFrame, current_idx: int, period: int
    ) -> float:
        """计算量比"""
        if current_idx < period:
            return 1.0

        current_volume = data.iloc[current_idx]["volume"]
        avg_volume = data.iloc[current_idx - period : current_idx]["volume"].mean()

        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def _is_breakout(
        self, data: pd.DataFrame, current_idx: int, period: int
    ) -> bool:
        """判断是否突破"""
        if current_idx < period:
            return False

        current_close = data.iloc[current_idx]["close"]
        recent_high = data.iloc[current_idx - period : current_idx]["high"].max()

        return current_close > recent_high

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        candidates = []

        for symbol in data["symbol"].unique():
            # 过滤ST股票
            if "ST" in symbol or "st" in symbol:
                continue

            # 只选择主板股票（排除科创板、创业板）
            if not is_main_board(symbol):
                continue

            stock_data = data[data["symbol"] == symbol].sort_values("date")
            stock_data = stock_data.reset_index(drop=True)

            # 获取截至指定日期的数据
            stock_data = stock_data[stock_data["date"] <= date]

            if len(stock_data) < max(self.volume_ratio_days, self.breakout_days):
                continue

            # 获取当前交易日
            target_idx = stock_data[stock_data["date"] == date].index
            if len(target_idx) == 0:
                target_idx = stock_data.index[-1]
            else:
                target_idx = target_idx[0]

            current_row = stock_data.iloc[target_idx]

            # 1. 价格过滤
            if current_row["close"] < self.min_price or current_row["close"] > self.max_price:
                continue

            # 2. 涨幅筛选（核心）- 找接近涨停的股票
            if "pct_chg" in current_row.index:
                pct_chg = current_row["pct_chg"]
                if pct_chg < self.min_pct_chg or pct_chg > self.max_pct_chg:
                    continue
            else:
                # 如果没有pct_chg列，计算涨幅
                if target_idx >= 1:
                    prev_close = stock_data.iloc[target_idx - 1]["close"]
                    pct_chg = (current_row["close"] - prev_close) / prev_close * 100
                    if pct_chg < self.min_pct_chg or pct_chg > self.max_pct_chg:
                        continue
                else:
                    continue

            # 3. 成交额过滤（万元转元）
            if current_row["amount"] < self.min_amount * 10000:
                continue

            # 4. 换手率过滤
            if "turnover" in current_row.index:
                if current_row["turnover"] < self.min_turnover or current_row["turnover"] > self.max_turnover:
                    continue

            # 5. 量比筛选（量能放大）
            volume_ratio = self._calculate_volume_ratio(stock_data, target_idx, self.volume_ratio_days)
            if volume_ratio < self.min_volume_ratio:
                continue

            # 6. 突破筛选
            if self.use_breakout:
                if not self._is_breakout(stock_data, target_idx, self.breakout_days):
                    continue

            # 7. MACD筛选
            if self.use_macd and len(stock_data) >= self.macd_slow:
                stock_data_copy = stock_data.copy()
                macd_line, signal_line, _ = self._calculate_macd(
                    stock_data_copy["close"], self.macd_fast, self.macd_slow, self.macd_signal
                )
                stock_data_copy["macd"] = macd_line
                stock_data_copy["signal"] = signal_line

                if target_idx >= 1:
                    macd_curr = stock_data_copy.loc[target_idx, "macd"]
                    signal_curr = stock_data_copy.loc[target_idx, "signal"]
                    macd_prev = stock_data_copy.loc[target_idx - 1, "macd"]
                    signal_prev = stock_data_copy.loc[target_idx - 1, "signal"]

                    # 要求MACD金叉或柱状图为正
                    is_golden_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
                    is_histogram_positive = (macd_curr > signal_curr)

                    if not (is_golden_cross or is_histogram_positive):
                        continue

            # 通过所有筛选条件
            candidates.append((symbol, pct_chg, current_row["amount"], volume_ratio))

        # 按涨幅和成交额综合排序（优先涨幅）
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        selected = [sym for sym, _, _, _ in candidates]

        if max_stocks is not None and len(selected) > max_stocks:
            selected = selected[:max_stocks]

        return selected

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pct_chg={self.min_pct_chg}-{self.max_pct_chg}%, "
            f"volume_ratio>={self.min_volume_ratio}, "
            f"amount>={self.min_amount}万)"
        )


class LimitUpTradingStrategy(BaseTradingStrategy):
    """涨停板打板交易策略

    专为打板设计的交易策略，目标是吃到涨停后快速获利了结。

    交易规则：
    1. 当天买入：在接近涨停时买入
    2. 短期持有：通常持有1-3天
    3. 严格止损：亏损达到5%时止损
    4. 灵活止盈：
       - 如果第二天高开高走，达到3%立即止盈
       - 如果盈利达到8%，启动移动止盈（最高价的92%）
    5. 强制退出：第3个交易日收盘时必须退出（避免追高后被套）

    参数说明：
    - hold_days: 最长持有天数，默认3天
    - stop_loss_pct: 止损百分比，默认5%
    - quick_profit_pct: 快速止盈百分比，默认3%（适合第二天高开）
    - trail_profit_pct: 启动移动止盈的盈利百分比，默认8%
    - trail_stop_pct: 移动止盈回撤百分比，默认8%（即最高价的92%）
    """

    def __init__(
        self,
        hold_days: int = 3,              # 最长持有3天
        stop_loss_pct: float = 0.05,     # 止损5%
        quick_profit_pct: float = 0.03,  # 快速止盈3%（第二天高开适用）
        trail_profit_pct: float = 0.08,  # 启动移动止盈8%
        trail_stop_pct: float = 0.08,    # 移动止盈回撤8%
    ):
        self.hold_days = hold_days
        self.stop_loss_pct = stop_loss_pct
        self.quick_profit_pct = quick_profit_pct
        self.trail_profit_pct = trail_profit_pct
        self.trail_stop_pct = trail_stop_pct

    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # 获取入场到出场之间的数据
        mask = (data["date"] >= entry_date) & (data["date"] <= exit_date)
        trading_data = data[mask].copy().sort_values("date")

        if len(trading_data) == 0:
            return pd.DataFrame(columns=["date", "signal"])

        # 获取入场价格（入场日的收盘价）
        entry_price = trading_data.iloc[0]["close"]

        signals = []
        exited = False
        hold_count = 0
        highest_price = entry_price
        trail_stop_price = None

        for _, row in trading_data.iterrows():
            current_price = row["close"]

            if exited:
                signals.append({"date": row["date"], "signal": Signal.SELL})
                continue

            hold_count += 1

            # 更新最高价
            highest_price = max(highest_price, current_price)

            # 计算收益率
            return_rate = (current_price - entry_price) / entry_price

            # 1. 检查止损
            if return_rate <= -self.stop_loss_pct:
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
                continue

            # 2. 快速止盈（适合第二天高开的情况）
            if hold_count == 2 and return_rate >= self.quick_profit_pct:
                # 第二天如果达到3%快速止盈
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
                continue

            # 3. 移动止盈
            if return_rate >= self.trail_profit_pct:
                # 启动移动止盈
                trail_stop_price = highest_price * (1 - self.trail_stop_pct)

            if trail_stop_price is not None and current_price <= trail_stop_price:
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
                continue

            # 4. 强制退出（达到持有天数）
            if hold_count >= self.hold_days:
                # 最后一天强制退出
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
            else:
                signals.append({"date": row["date"], "signal": Signal.HOLD})

        return pd.DataFrame(signals)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hold={self.hold_days}d, "
            f"stop_loss={self.stop_loss_pct*100:.1f}%, "
            f"quick_profit={self.quick_profit_pct*100:.1f}%, "
            f"trail_profit={self.trail_profit_pct*100:.1f}%)"
        )
