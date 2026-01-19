"""策略模块

包含选股策略和交易策略的基类和实现。
"""

from .base import BaseSelectionStrategy, BaseTradingStrategy
from .selection import (
    PriceRangeSelectionStrategy,
    MovingAverageSelectionStrategy,
)
from .trading import (
    FixedStopLossTradingStrategy,
    MovingAverageCrossTradingStrategy,
    HoldTradingStrategy,
)

__all__ = [
    "BaseSelectionStrategy",
    "BaseTradingStrategy",
    "PriceRangeSelectionStrategy",
    "MovingAverageSelectionStrategy",
    "FixedStopLossTradingStrategy",
    "MovingAverageCrossTradingStrategy",
    "HoldTradingStrategy",
]
