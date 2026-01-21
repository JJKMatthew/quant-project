"""策略模块

包含选股策略和交易策略的基类和实现。
"""

from .base import BaseSelectionStrategy, BaseTradingStrategy, Signal
from .selection import (
    PriceRangeSelectionStrategy,
    MovingAverageSelectionStrategy,
    MomentumSelectionStrategy,
)
from .trading import (
    FixedStopLossTradingStrategy,
    MovingAverageCrossTradingStrategy,
    HoldTradingStrategy,
    ATRStopLossTradingStrategy,
)
from .high_yield import (
    is_main_board,
    HighYieldShortTermSelectionStrategy,
    FiveDayHoldTradingStrategy,
    MomentumBreakthroughSelectionStrategy,
)
from .limit_up import (
    LimitUpSelectionStrategy,
    LimitUpTradingStrategy,
)

__all__ = [
    "BaseSelectionStrategy",
    "BaseTradingStrategy",
    "Signal",
    # 工具函数
    "is_main_board",
    # 选股策略
    "PriceRangeSelectionStrategy",
    "MovingAverageSelectionStrategy",
    "MomentumSelectionStrategy",
    "HighYieldShortTermSelectionStrategy",
    "MomentumBreakthroughSelectionStrategy",
    "LimitUpSelectionStrategy",
    # 交易策略
    "HoldTradingStrategy",
    "FixedStopLossTradingStrategy",
    "MovingAverageCrossTradingStrategy",
    "ATRStopLossTradingStrategy",
    "FiveDayHoldTradingStrategy",
    "LimitUpTradingStrategy",
]
