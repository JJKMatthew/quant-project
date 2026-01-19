"""交易策略实现

提供多种常用的交易策略。
"""

from typing import Optional
import pandas as pd
import numpy as np

from .base import BaseTradingStrategy, Signal


class HoldTradingStrategy(BaseTradingStrategy):
    """持有策略

    买入后一直持有，直到出场日期。
    """

    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # 获取入场到出场之间的数据
        mask = (data["date"] >= entry_date) & (data["date"] <= exit_date)
        trading_data = data[mask].copy().sort_values("date")

        # 所有交易日都持有
        trading_data["signal"] = Signal.HOLD

        return trading_data[["date", "signal"]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FixedStopLossTradingStrategy(BaseTradingStrategy):
    """固定止损策略

    当亏损达到指定百分比时止损。
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.05,  # 止损百分比，如 0.05 表示 5%
        take_profit_pct: Optional[float] = None,  # 止盈百分比，None 表示不止盈
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

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

        for _, row in trading_data.iterrows():
            current_price = row["close"]

            if exited:
                signals.append({"date": row["date"], "signal": Signal.SELL})
            else:
                # 计算收益率
                return_rate = (current_price - entry_price) / entry_price

                # 检查止损
                if return_rate <= -self.stop_loss_pct:
                    signals.append({"date": row["date"], "signal": Signal.EXIT})
                    exited = True
                # 检查止盈
                elif self.take_profit_pct is not None and return_rate >= self.take_profit_pct:
                    signals.append({"date": row["date"], "signal": Signal.EXIT})
                    exited = True
                else:
                    signals.append({"date": row["date"], "signal": Signal.HOLD})

        return pd.DataFrame(signals)

    def __repr__(self) -> str:
        params = f"stop_loss={self.stop_loss_pct*100:.1f}%"
        if self.take_profit_pct is not None:
            params += f", take_profit={self.take_profit_pct*100:.1f}%"
        return f"{self.__class__.__name__}({params})"


class MovingAverageCrossTradingStrategy(BaseTradingStrategy):
    """均线交叉交易策略

    短期均线上穿长期均线时买入/持有，下穿时卖出。
    """

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        stop_loss_pct: Optional[float] = None,
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.stop_loss_pct = stop_loss_pct

    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # 获取入场到出场之间的数据
        mask = (data["date"] >= entry_date) & (data["date"] <= exit_date)
        trading_data = data[mask].copy().sort_values("date")

        if len(trading_data) < self.long_period:
            # 数据不足，全程持有
            trading_data = trading_data.copy()
            trading_data["signal"] = Signal.HOLD
            return trading_data[["date", "signal"]]

        # 计算均线
        trading_data = trading_data.copy()
        trading_data["ma_short"] = trading_data["close"].rolling(self.short_period).mean()
        trading_data["ma_long"] = trading_data["close"].rolling(self.long_period).mean()

        # 记录入场价格
        entry_price = trading_data.iloc[0]["close"]

        signals = []
        exited = False
        last_signal = Signal.HOLD

        for idx, row in trading_data.iterrows():
            if exited:
                signals.append({"date": row["date"], "signal": Signal.SELL})
                continue

            # 等到有足够的均线数据
            if pd.isna(row["ma_short"]) or pd.isna(row["ma_long"]):
                signals.append({"date": row["date"], "signal": Signal.HOLD})
                continue

            # 获取前一交易日的数据
            current_idx = trading_data.index.get_loc(idx)
            if current_idx == 0:
                signals.append({"date": row["date"], "signal": Signal.HOLD})
                continue

            prev_idx = trading_data.index[current_idx - 1]
            prev_row = trading_data.loc[prev_idx]

            ma_short_curr = row["ma_short"]
            ma_long_curr = row["ma_long"]
            ma_short_prev = prev_row["ma_short"]
            ma_long_prev = prev_row["ma_long"]

            # 检查止损
            if self.stop_loss_pct is not None:
                return_rate = (row["close"] - entry_price) / entry_price
                if return_rate <= -self.stop_loss_pct:
                    signals.append({"date": row["date"], "signal": Signal.EXIT})
                    exited = True
                    continue

            # 金叉：短期均线从下向上穿过长期均线 → 买入/持有
            if (ma_short_prev < ma_long_prev) and (ma_short_curr > ma_long_curr):
                signals.append({"date": row["date"], "signal": Signal.HOLD})
                last_signal = Signal.HOLD
            # 死叉：短期均线从上向下穿过长期均线 → 卖出
            elif (ma_short_prev > ma_long_prev) and (ma_short_curr < ma_long_curr):
                signals.append({"date": row["date"], "signal": Signal.SELL})
                last_signal = Signal.SELL
            else:
                # 保持上一状态
                signals.append({"date": row["date"], "signal": last_signal})

        return pd.DataFrame(signals)

    def __repr__(self) -> str:
        params = f"short={self.short_period}, long={self.long_period}"
        if self.stop_loss_pct is not None:
            params += f", stop_loss={self.stop_loss_pct*100:.1f}%"
        return f"{self.__class__.__name__}({params})"


class ATRStopLossTradingStrategy(BaseTradingStrategy):
    """ATR（平均真实波幅）止损策略

    使用 ATR 指标进行动态止损，适应市场波动。
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,  # ATR 倍数
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """计算 ATR（平均真实波幅）"""
        df = df.copy()

        # 计算真实波幅 TR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        return atr

    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # 获取入场到出场之间的数据
        mask = (data["date"] >= entry_date) & (data["date"] <= exit_date)
        trading_data = data[mask].copy().sort_values("date")

        if len(trading_data) < self.atr_period:
            # 数据不足，全程持有
            trading_data = trading_data.copy()
            trading_data["signal"] = Signal.HOLD
            return trading_data[["date", "signal"]]

        # 计算 ATR
        trading_data = trading_data.copy()
        trading_data["atr"] = self._calculate_atr(trading_data)

        # 记录入场价格和止损线
        entry_price = trading_data.iloc[0]["close"]
        stop_loss_price = None

        signals = []
        exited = False

        for _, row in trading_data.iterrows():
            if exited:
                signals.append({"date": row["date"], "signal": Signal.SELL})
                continue

            # 等待有 ATR 数据
            if pd.isna(row["atr"]):
                signals.append({"date": row["date"], "signal": Signal.HOLD})
                continue

            # 计算止损线：入场价减去 ATR 的倍数
            if stop_loss_price is None:
                stop_loss_price = entry_price - row["atr"] * self.atr_multiplier
            else:
                # 可以选择使用追踪止损
                potential_stop = row["high"] - row["atr"] * self.atr_multiplier
                stop_loss_price = max(stop_loss_price, potential_stop)

            # 检查是否触及止损
            if row["low"] <= stop_loss_price:
                signals.append({"date": row["date"], "signal": Signal.EXIT})
                exited = True
            else:
                signals.append({"date": row["date"], "signal": Signal.HOLD})

        return pd.DataFrame(signals)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(period={self.atr_period}, multiplier={self.atr_multiplier})"
