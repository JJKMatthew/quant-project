"""策略基类定义

所有自定义策略都应继承自这些基类。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd


class BaseSelectionStrategy(ABC):
    """选股策略基类

    选股策略负责在给定时间点选择股票池。

    子类需要实现 select 方法，该方法接收股票数据和日期，
    返回符合选股条件的股票代码列表。
    """

    @abstractmethod
    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        """选择股票

        Args:
            data: 包含所有股票历史数据的 DataFrame
                  必须包含列: date, symbol, open, high, low, close, volume, amount
            date: 选股日期
            max_stocks: 最多选择多少只股票，None 表示不限制

        Returns:
            符合条件的股票代码列表，如 ["sh601988", "sz000001"]
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseTradingStrategy(ABC):
    """交易策略基类

    交易策略负责在持有股票的过程中产生买卖信号。

    子类需要实现 generate_signals 方法，该方法接收单个股票的数据，
    返回买卖信号序列。
    """

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """生成交易信号

        Args:
            data: 单只股票的历史数据
            entry_date: 入场日期
            exit_date: 出场日期

        Returns:
            包含信号列的 DataFrame，格式如下：
            - 必须包含 date 列
            - 必须包含 signal 列: 1=买入/持有, 0=卖出/空仓, -1=卖出信号
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Signal:
    """交易信号常量"""

    HOLD = 1      # 持有/买入
    SELL = 0      # 卖出/空仓
    EXIT = -1     # 强制退出
