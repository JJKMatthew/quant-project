"""中短期高收益策略实现

专为5日持有期设计的目标收益率>=10%的选股和交易策略。
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


class HighYieldShortTermSelectionStrategy(BaseSelectionStrategy):
    """中短期高收益选股策略

    专为5日持有期设计的选股策略，目标是筛选出短期强势股。

    选股思路：
    1. 短期强势：过去3-5天涨幅筛选
    2. 量价配合：成交量放大、换手率适中
    3. 突破信号：突破近期高点或关键阻力位
    4. 技术指标：RSI不超买、MACD金叉
    5. 价格过滤：排除ST股、超低价股、超高价股
    6. 成交额筛选：保证流动性

    参数说明：
    - lookback_days: 回溯天数，用于计算短期涨幅
    - min_return: 最低涨幅要求（百分比），如0.05表示5%
    - max_return: 最高涨幅限制（百分比），避免追高
    - min_price, max_price: 价格区间过滤
    - min_amount: 最小成交额要求（万元）
    - min_turnover: 最小换手率（%）
    - max_turnover: 最大换手率（%），过滤过度炒作
    - use_rsi: 是否使用RSI过滤
    - rsi_min, rsi_max: RSI区间，避免超买超卖
    - use_macd: 是否使用MACD金叉过滤
    - use_breakout: 是否使用突破过滤
    - breakout_days: 突破回溯天数
    - volume_ratio_days: 量比回溯天数
    - min_volume_ratio: 最小量比
    """

    def __init__(
        self,
        # 基础过滤
        min_price: float = 3.0,      # 最低价格（排除低价垃圾股）
        max_price: float = 50.0,     # 最高价格（避免追高）
        min_amount: float = 3000,    # 最小成交额（万元）
        min_turnover: float = 2.0,   # 最小换手率（%）
        max_turnover: float = 15.0,  # 最大换手率（%）

        # 涨幅筛选
        lookback_days: int = 5,      # 回溯天数
        min_return: float = 3.0,     # 最低涨幅（%），过去5天至少涨3%
        max_return: float = 20.0,    # 最高涨幅（%），避免追太高

        # 量价配合
        volume_ratio_days: int = 5,  # 量比回溯天数
        min_volume_ratio: float = 1.3,  # 最小量比，要求成交量放大

        # 突破筛选
        use_breakout: bool = True,   # 是否使用突破过滤
        breakout_days: int = 20,     # 突破回溯天数

        # 技术指标
        use_rsi: bool = True,        # 是否使用RSI
        rsi_period: int = 14,        # RSI周期
        rsi_min: float = 30,         # RSI下限
        rsi_max: float = 70,         # RSI上限（避免超买）

        use_macd: bool = True,       # 是否使用MACD
        macd_fast: int = 12,         # MACD快线
        macd_slow: int = 26,        # MACD慢线
        macd_signal: int = 9,        # MACD信号线

        # 均线过滤
        use_ma: bool = True,         # 是否使用均线
        ma_short: int = 5,           # 短期均线
        ma_long: int = 20,           # 长期均线
        ma_trend: str = "above",     # 趋势：above（短期在长期之上）/golden_cross（金叉）
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_amount = min_amount
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover
        self.lookback_days = lookback_days
        self.min_return = min_return
        self.max_return = max_return
        self.volume_ratio_days = volume_ratio_days
        self.min_volume_ratio = min_volume_ratio
        self.use_breakout = use_breakout
        self.breakout_days = breakout_days
        self.use_rsi = use_rsi
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.use_macd = use_macd
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.use_ma = use_ma
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.ma_trend = ma_trend

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

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

    def _is_breakout(
        self, data: pd.DataFrame, current_idx: int, period: int
    ) -> bool:
        """判断是否突破"""
        if current_idx < period:
            return False

        current_close = data.iloc[current_idx]["close"]
        recent_high = data.iloc[current_idx - period : current_idx]["high"].max()

        return current_close > recent_high

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

            if len(stock_data) < max(self.lookback_days, self.ma_long, self.rsi_period):
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

            # 2. 成交额过滤（万元转元）
            if self.min_amount is not None and current_row["amount"] < self.min_amount * 10000:
                continue

            # 3. 换手率过滤
            if "turnover" in current_row.index:
                if current_row["turnover"] < self.min_turnover or current_row["turnover"] > self.max_turnover:
                    continue

            # 4. 短期涨幅筛选
            if target_idx >= self.lookback_days:
                start_idx = target_idx - self.lookback_days
                start_price = stock_data.iloc[start_idx]["close"]
                return_rate = (current_row["close"] - start_price) / start_price * 100

                if return_rate < self.min_return or return_rate > self.max_return:
                    continue
            else:
                continue

            # 5. 量比筛选
            if self.min_volume_ratio is not None:
                volume_ratio = self._calculate_volume_ratio(stock_data, target_idx, self.volume_ratio_days)
                if volume_ratio < self.min_volume_ratio:
                    continue

            # 6. 突破筛选
            if self.use_breakout:
                if not self._is_breakout(stock_data, target_idx, self.breakout_days):
                    continue

            # 7. RSI筛选
            if self.use_rsi:
                stock_data_copy = stock_data.copy()
                stock_data_copy["rsi"] = self._calculate_rsi(stock_data_copy["close"], self.rsi_period)
                current_rsi = stock_data_copy.loc[target_idx, "rsi"]

                if current_rsi < self.rsi_min or current_rsi > self.rsi_max:
                    continue

            # 8. MACD筛选
            if self.use_macd:
                stock_data_copy = stock_data.copy()
                macd_line, signal_line, _ = self._calculate_macd(
                    stock_data_copy["close"], self.macd_fast, self.macd_slow, self.macd_signal
                )
                stock_data_copy["macd"] = macd_line
                stock_data_copy["signal"] = signal_line

                # 要求MACD柱状图为正（或MACD线上穿信号线）
                if target_idx >= 1:
                    macd_curr = stock_data_copy.loc[target_idx, "macd"]
                    signal_curr = stock_data_copy.loc[target_idx, "signal"]
                    macd_prev = stock_data_copy.loc[target_idx - 1, "macd"]
                    signal_prev = stock_data_copy.loc[target_idx - 1, "signal"]

                    # 金叉：MACD从下方向上穿过信号线
                    is_golden_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
                    # 或MACD柱状图为正且增加
                    is_histogram_positive = (macd_curr > signal_curr) and (macd_curr - signal_curr > 0)

                    if not (is_golden_cross or is_histogram_positive):
                        continue
                else:
                    continue

            # 9. 均线筛选
            if self.use_ma:
                stock_data_copy = stock_data.copy()
                stock_data_copy["ma_short"] = stock_data_copy["close"].rolling(self.ma_short).mean()
                stock_data_copy["ma_long"] = stock_data_copy["close"].rolling(self.ma_long).mean()

                ma_short_curr = stock_data_copy.loc[target_idx, "ma_short"]
                ma_long_curr = stock_data_copy.loc[target_idx, "ma_long"]

                if self.ma_trend == "above":
                    # 短期均线在长期均线之上
                    if pd.isna(ma_short_curr) or pd.isna(ma_long_curr) or ma_short_curr < ma_long_curr:
                        continue
                elif self.ma_trend == "golden_cross":
                    # 金叉：短期均线从下方向上穿过长期均线
                    if target_idx < 1:
                        continue
                    ma_short_prev = stock_data_copy.loc[target_idx - 1, "ma_short"]
                    ma_long_prev = stock_data_copy.loc[target_idx - 1, "ma_long"]

                    if pd.isna(ma_short_prev) or pd.isna(ma_long_prev):
                        continue

                    is_golden_cross = (ma_short_prev < ma_long_prev) and (ma_short_curr > ma_long_curr)
                    if not is_golden_cross:
                        continue

            # 通过所有筛选条件
            candidates.append(symbol)

        # 按成交额和涨幅综合排序
        candidate_scores = []
        for symbol in candidates:
            stock_data = data[data["symbol"] == symbol].sort_values("date")
            stock_data = stock_data[stock_data["date"] <= date]

            if len(stock_data) == 0:
                continue

            target_idx = stock_data[stock_data["date"] == date].index
            if len(target_idx) == 0:
                target_idx = stock_data.index[-1]
            else:
                target_idx = target_idx[0]

            current_row = stock_data.iloc[target_idx]

            # 计算综合得分
            score = current_row["amount"]  # 成交额权重

            # 涨幅在合理区间内给额外加分
            if target_idx >= self.lookback_days:
                start_idx = target_idx - self.lookback_days
                start_price = stock_data.iloc[start_idx]["close"]
                return_rate = (current_row["close"] - start_price) / start_price * 100

                # 涨幅在5%-12%之间给加分
                if 5 <= return_rate <= 12:
                    score *= 1.2

            candidate_scores.append((symbol, score))

        # 按得分排序
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [sym for sym, _ in candidate_scores]

        if max_stocks is not None and len(selected) > max_stocks:
            selected = selected[:max_stocks]

        return selected

    def __repr__(self) -> str:
        params = f"return={self.min_return}%-{self.max_return}%, price={self.min_price}-{self.max_price}"
        if self.use_rsi:
            params += f", RSI={self.rsi_min}-{self.rsi_max}"
        if self.use_macd:
            params += ", MACD"
        if self.use_breakout:
            params += f", breakout({self.breakout_days}d)"
        return f"{self.__class__.__name__}({params})"


class FiveDayHoldTradingStrategy(BaseTradingStrategy):
    """5日固定持有期交易策略

    专为5日持有期设计的交易策略，目标是获得10%以上的收益。

    交易规则：
    1. 固定持有5个交易日
    2. 止损：亏损达到5%时止损
    3. 止盈：
       - 盈利达到8%时启动移动止盈
       - 移动止盈线为最高价的90%
    4. 强制退出：第5个交易日收盘时必须退出

    参数说明：
    - hold_days: 持有天数，默认5
    - stop_loss_pct: 止损百分比，默认5%
    - trail_profit_pct: 启动移动止盈的盈利百分比，默认8%
    - trail_stop_pct: 移动止盈回撤百分比，默认10%（即最高价的90%）
    """

    def __init__(
        self,
        hold_days: int = 5,            # 持有天数
        stop_loss_pct: float = 0.05,   # 止损百分比（5%）
        trail_profit_pct: float = 0.08,  # 启动移动止盈的盈利百分比（8%）
        trail_stop_pct: float = 0.10,  # 移动止盈回撤百分比（10%）
    ):
        self.hold_days = hold_days
        self.stop_loss_pct = stop_loss_pct
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

            # 2. 检查移动止盈
            if return_rate >= self.trail_profit_pct:
                # 启动移动止盈，止损线设为最高价回撤trail_stop_pct
                trail_stop_price = highest_price * (1 - self.trail_stop_pct)

            if trail_stop_price is not None and current_price <= trail_stop_price:
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
                continue

            # 3. 检查强制退出（达到持有天数）
            if hold_count >= self.hold_days:
                # 最后一天强制退出
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
            else:
                signals.append({"date": row["date"], "signal": Signal.HOLD})

        return pd.DataFrame(signals)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(hold={self.hold_days}d, "
            f"stop_loss={self.stop_loss_pct*100:.1f}%, "
            f"trail_profit={self.trail_profit_pct*100:.1f}%, "
            f"trail_stop={self.trail_stop_pct*100:.1f}%)"
        )


class MomentumBreakthroughSelectionStrategy(BaseSelectionStrategy):
    """动量突破选股策略（激进版）

    专注于捕捉强势股的突破机会，适合短线交易。

    核心逻辑：
    1. 筛选强势股：过去N天涨幅排名靠前
    2. 突破确认：价格突破N天高点
    3. 量能确认：成交量显著放大
    4. 趋势确认：短期均线多头排列

    参数说明：
    - momentum_days: 动量计算天数
    - top_n: 选择前N只强势股
    - breakout_days: 突破回溯天数
    - min_volume_ratio: 最小量比
    - use_macd: 是否使用MACD确认
    """

    def __init__(
        self,
        momentum_days: int = 10,      # 动量计算天数
        top_n: Optional[int] = 10,    # 选择前N只
        breakout_days: int = 20,      # 突破回溯天数
        min_volume_ratio: float = 1.5,  # 最小量比
        use_macd: bool = True,        # 是否使用MACD
        min_price: float = 3.0,
        max_price: float = 100.0,
        min_amount: float = 5000,     # 最小成交额（万元）
    ):
        self.momentum_days = momentum_days
        self.top_n = top_n
        self.breakout_days = breakout_days
        self.min_volume_ratio = min_volume_ratio
        self.use_macd = use_macd
        self.min_price = min_price
        self.max_price = max_price
        self.min_amount = min_amount

    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """计算MACD"""
        ema_fast = prices.ewm(span=12, adjust=False).mean()
        ema_slow = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        momentum = []

        for symbol in data["symbol"].unique():
            # 过滤ST股票
            if "ST" in symbol or "st" in symbol:
                continue

            # 只选择主板股票（排除科创板、创业板）
            if not is_main_board(symbol):
                continue

            stock_data = data[data["symbol"] == symbol].sort_values("date")
            stock_data = stock_data[stock_data["date"] <= date]

            if len(stock_data) < self.momentum_days + 1:
                continue

            target_idx = stock_data[stock_data["date"] == date].index
            if len(target_idx) == 0:
                target_idx = stock_data.index[-1]
            else:
                target_idx = target_idx[0]

            current_row = stock_data.iloc[target_idx]

            # 1. 价格过滤
            if current_row["close"] < self.min_price or current_row["close"] > self.max_price:
                continue

            # 2. 成交额过滤
            if current_row["amount"] < self.min_amount * 10000:
                continue

            # 3. 计算动量（过去N天涨幅）
            start_idx = max(0, target_idx - self.momentum_days)
            start_price = stock_data.iloc[start_idx]["close"]
            momentum_rate = (current_row["close"] - start_price) / start_price * 100

            # 只关注涨幅为正的股票
            if momentum_rate <= 0:
                continue

            # 4. 突破确认
            if target_idx >= self.breakout_days:
                recent_high = stock_data.iloc[target_idx - self.breakout_days : target_idx]["high"].max()
                is_breakout = current_row["close"] > recent_high
                if not is_breakout:
                    continue
            else:
                continue

            # 5. 量能确认
            if target_idx >= self.momentum_days:
                current_volume = stock_data.iloc[target_idx]["volume"]
                avg_volume = stock_data.iloc[target_idx - self.momentum_days : target_idx]["volume"].mean()

                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio < self.min_volume_ratio:
                        continue

            # 6. MACD确认
            if self.use_macd:
                stock_data_copy = stock_data.reset_index(drop=True)
                macd_line, signal_line, histogram = self._calculate_macd(stock_data_copy["close"])

                local_idx = stock_data_copy[stock_data_copy["date"] == date].index
                if len(local_idx) == 0:
                    local_idx = stock_data_copy.index[-1]
                else:
                    local_idx = local_idx[0]

                if local_idx >= 1:
                    macd_curr = macd_line.iloc[local_idx]
                    signal_curr = signal_line.iloc[local_idx]
                    macd_prev = macd_line.iloc[local_idx - 1]
                    signal_prev = signal_line.iloc[local_idx - 1]

                    # 要求金叉或MACD柱状图为正
                    is_golden_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)
                    histogram_pos = (macd_curr - signal_curr) > 0

                    if not (is_golden_cross or histogram_pos):
                        continue

            momentum.append((symbol, momentum_rate, current_row["amount"]))

        # 按动量（涨幅）排序，涨幅相同时按成交额排序
        momentum.sort(key=lambda x: (x[1], x[2]), reverse=True)

        selected = [sym for sym, _, _ in momentum]
        if self.top_n is not None:
            selected = selected[:self.top_n]
        if max_stocks is not None:
            selected = selected[:max_stocks]

        return selected

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(momentum={self.momentum_days}d, "
            f"breakout={self.breakout_days}d, top_n={self.top_n})"
        )
