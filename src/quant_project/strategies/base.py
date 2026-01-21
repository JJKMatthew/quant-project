"""策略基类定义

所有自定义策略都应继承自这些基类。

架构说明：
- BaseSelectionStrategy: 选股策略，决定"买什么"
- BaseTradingStrategy: 交易策略，决定"何时买卖"

实现自己的策略时：
1. 选股策略：继承 BaseSelectionStrategy，实现 select() 方法
2. 交易策略：继承 BaseTradingStrategy，实现 generate_signals() 方法
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd


class BaseSelectionStrategy(ABC):
    """选股策略基类

    选股策略负责在给定时间点选择股票池。

    核心方法 select():
    - 输入：所有股票的历史数据、选股日期、最大选股数
    - 输出：符合选股条件的股票代码列表

    选股策略示例思路：
    - 价格区间筛选
    - 涨幅筛选（过去N天涨幅排名靠前）
    - 技术指标筛选（RSI、MACD、KDJ等）
    - 均线筛选（金叉、多头排列）
    - 量能筛选（成交量放大）
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
            date: 选股日期（通常选在调仓日）
            max_stocks: 最多选择多少只股票，None 表示不限制

        Returns:
            符合条件的股票代码列表，如 ["sh601988", "sz000001"]

        注意事项：
        - 策略只返回股票代码，不涉及资金分配
        - 应考虑数据充足性（如需要20日均线，需至少20天数据）
        - 可结合多因子综合评分排序
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseTradingStrategy(ABC):
    """交易策略基类

    交易策略负责在持有股票的过程中产生买卖信号。

    核心方法 generate_signals():
    - 输入：单只股票从入场到出场期间的数据
    - 输出：每日的交易信号序列

    交易策略示例思路：
    - 固定持有期策略
    - 止损/止盈策略
    - 均线交叉策略
    - ATR 动态止损策略
    - 技术指标信号策略
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
            data: 单只股票的历史数据（需包含从 entry_date 到 exit_date 的数据）
            entry_date: 入场日期（买入日）
            exit_date: 出场日期（计划卖出日，可提前触发止损止盈）

        Returns:
            包含信号列的 DataFrame，格式如下：
            - 必须包含 date 列
            - 必须包含 signal 列: 1=买入/持有(HOLD), 0=卖出/空仓(SELL), -1=强制退出(EXIT)

        信号说明：
        - HOLD (1): 继续持有
        - SELL (0): 卖出信号
        - EXIT (-1): 强制退出信号（如止损触发）

        注意事项：
        - 返回的日期范围应覆盖 entry_date 到 exit_date
        - 每日应有明确的信号
        - 信号为 SELL/EXIT 后，后续日期也应为 SELL
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Signal:
    """交易信号常量

    信号类型：
    - HOLD: 继续持有股票
    - SELL: 卖出信号（如技术指标看空）
    - EXIT: 强制退出信号（如止损、止盈触发）
    """

    HOLD = 1      # 持有/买入
    SELL = 0      # 卖出/空仓
    EXIT = -1     # 强制退出（止损/止盈）
