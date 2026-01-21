"""选股策略实现

提供多种常用的选股策略，可直接使用或作为自定义策略的参考。

策略列表：
1. PriceRangeSelectionStrategy - 价格区间选股（基础策略）
2. MovingAverageSelectionStrategy - 均线金叉/死叉选股
3. MomentumSelectionStrategy - 动量选股（涨幅排名）

使用建议：
- 初学者建议从 PriceRangeSelectionStrategy 开始
- 趋势投资者可使用 MovingAverageSelectionStrategy
- 追逐强势股可使用 MomentumSelectionStrategy
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from .base import BaseSelectionStrategy


class PriceRangeSelectionStrategy(BaseSelectionStrategy):
    """价格范围选股策略

    简单的条件选股策略，基于价格、成交额、涨跌幅等条件筛选股票。

    适用场景：
    - 适合作为自定义策略的基础筛选层
    - 适合对市场有基本了解的投资者

    筛选逻辑：
    1. 价格区间过滤（排除低价垃圾股和高价股）
    2. 成交额过滤（保证流动性）
    3. 涨跌幅过滤（筛选强势或弱势股）
    4. 换手率过滤（排除过度炒作）
    5. 按成交额降序排序（选择流动性好的股票）
    """

    def __init__(
        self,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        min_pct_chg: Optional[float] = None,
        max_pct_chg: Optional[float] = None,
        min_turnover: Optional[float] = None,
        max_turnover: Optional[float] = None,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.min_pct_chg = min_pct_chg
        self.max_pct_chg = max_pct_chg
        self.min_turnover = min_turnover
        self.max_turnover = max_turnover

    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        # 获取指定日期的所有股票数据
        mask = data["date"] == date
        if not mask.any():
            return []

        candidates = data[mask].copy()

        # 应用筛选条件
        if self.min_price is not None:
            candidates = candidates[candidates["close"] >= self.min_price]
        if self.max_price is not None:
            candidates = candidates[candidates["close"] <= self.max_price]
        if self.min_amount is not None:
            candidates = candidates[candidates["amount"] >= self.min_amount]
        if self.max_amount is not None:
            candidates = candidates[candidates["amount"] <= self.max_amount]
        if self.min_pct_chg is not None and "pct_chg" in candidates.columns:
            candidates = candidates[candidates["pct_chg"] >= self.min_pct_chg]
        if self.max_pct_chg is not None and "pct_chg" in candidates.columns:
            candidates = candidates[candidates["pct_chg"] <= self.max_pct_chg]
        if self.min_turnover is not None and "turnover" in candidates.columns:
            candidates = candidates[candidates["turnover"] >= self.min_turnover]
        if self.max_turnover is not None and "turnover" in candidates.columns:
            candidates = candidates[candidates["turnover"] <= self.max_turnover]

        # 按成交额降序排序
        candidates = candidates.sort_values("amount", ascending=False)

        # 限制股票数量
        selected = candidates["symbol"].tolist()
        if max_stocks is not None and len(selected) > max_stocks:
            selected = selected[:max_stocks]

        return selected

    def __repr__(self) -> str:
        params = []
        if self.min_price is not None:
            params.append(f"min_price={self.min_price}")
        if self.max_price is not None:
            params.append(f"max_price={self.max_price}")
        if self.min_amount is not None:
            params.append(f"min_amount={self.min_amount}")
        if self.max_pct_chg is not None:
            params.append(f"min_pct_chg={self.min_pct_chg}")
        return f"{self.__class__.__name__}({', '.join(params)})"


class MovingAverageSelectionStrategy(BaseSelectionStrategy):
    """均线选股策略

    选择收盘价上穿/下穿均线的股票。

    适用场景：
    - 趋势市场中的顺势交易
    - 捕捉趋势转折点

    核心概念：
    - 金叉（Golden Cross）：短期均线从下方向上穿过长期均线，看多信号
    - 死叉（Death Cross）：短期均线从上方向下穿过长期均线，看空信号

    常用参数：
    - 5日/20日：短线金叉，适合短线操作
    - 10日/30日：中线金叉，适合波段操作
    - 20日/60日：长线金叉，适合中长线

    注意事项：
    - 需要足够的历史数据（至少 long_period + 1 天）
    - 假突破后可能很快失败，需配合止损
    """

    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 20,
        mode: str = "golden_cross",  # golden_cross: 金叉(上穿), death_cross: 死叉(下穿)
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.mode = mode

    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        candidates = []

        for symbol in data["symbol"].unique():
            stock_data = data[data["symbol"] == symbol].sort_values("date")

            # 获取截至指定日期的数据
            stock_data = stock_data[stock_data["date"] <= date]

            if len(stock_data) < self.long_period + 1:
                continue

            # 计算短期和长期均线
            stock_data = stock_data.copy()
            stock_data["ma_short"] = stock_data["close"].rolling(self.short_period).mean()
            stock_data["ma_long"] = stock_data["close"].rolling(self.long_period).mean()

            # 检查是否在指定日期发生金叉或死叉
            target_row = stock_data[stock_data["date"] == date]
            if len(target_row) == 0:
                # 找最近一个交易日
                if len(stock_data) == 0:
                    continue
                target_idx = stock_data.index[-1]
            else:
                target_idx = target_row.index[0]

            if target_idx < self.long_period:
                continue

            # 获取前一交易日
            prev_idx = stock_data.index[stock_data.index.get_loc(target_idx) - 1]

            ma_short_curr = stock_data.loc[target_idx, "ma_short"]
            ma_long_curr = stock_data.loc[target_idx, "ma_long"]
            ma_short_prev = stock_data.loc[prev_idx, "ma_short"]
            ma_long_prev = stock_data.loc[prev_idx, "ma_long"]

            # 金叉：短期均线从下方向上穿过长期均线
            if self.mode == "golden_cross":
                if (ma_short_prev < ma_long_prev) and (ma_short_curr > ma_long_curr):
                    candidates.append(symbol)
            # 死叉：短期均线从上方向下穿过长期均线
            elif self.mode == "death_cross":
                if (ma_short_prev > ma_long_prev) and (ma_short_curr < ma_long_curr):
                    candidates.append(symbol)

        if max_stocks is not None and len(candidates) > max_stocks:
            candidates = candidates[:max_stocks]

        return candidates

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(short={self.short_period}, long={self.long_period}, mode={self.mode})"


class MomentumSelectionStrategy(BaseSelectionStrategy):
    """动量选股策略

    选择过去 N 天涨幅最大的股票。

    适用场景：
    - 追逐市场热点和强势股
    - 短线爆发式行情
    - 牛市中的强者恒强逻辑

    核心思想：
    - 动量效应：过去表现好的股票在未来继续表现好的概率较高
    - 选择近期涨幅排名靠前的股票

    常用参数：
    - 5日/10日：超短线动量
    - 20日：短期动量
    - 60日：中期动量

    注意事项：
    - 涨幅过高时追涨风险大，需配合止损
    - 在震荡市中效果可能不佳
    - 需结合成交量判断资金真实性
    """

    def __init__(self, period: int = 20, top_n: Optional[int] = None):
        self.period = period
        self.top_n = top_n

    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        momentum = []

        for symbol in data["symbol"].unique():
            stock_data = data[data["symbol"] == symbol].sort_values("date")

            # 获取截至指定日期的数据
            stock_data = stock_data[stock_data["date"] <= date]

            if len(stock_data) < self.period + 1:
                continue

            # 计算过去 N 天的涨幅
            start_idx = max(0, len(stock_data) - self.period - 1)
            end_idx = len(stock_data) - 1

            start_price = stock_data.iloc[start_idx]["close"]
            end_price = stock_data.iloc[end_idx]["close"]

            return_rate = (end_price - start_price) / start_price * 100
            momentum.append((symbol, return_rate))

        # 按涨幅排序
        momentum.sort(key=lambda x: x[1], reverse=True)

        # 选择前 N 只
        selected = [sym for sym, _ in momentum]
        if self.top_n is not None:
            selected = selected[:self.top_n]
        if max_stocks is not None:
            selected = selected[:max_stocks]

        return selected

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(period={self.period}, top_n={self.top_n})"
