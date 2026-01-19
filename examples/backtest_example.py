"""回测器使用示例

演示如何使用选股策略回测器进行回测。
"""

# 示例 1: 简单的价格范围选股 + 持有策略
def example_1():
    """价格范围选股 + 持有策略"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import PriceRangeSelectionStrategy, HoldTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    # 获取数据
    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=50,  # 只获取前50只股票
    )

    # 创建策略
    selection_strategy = PriceRangeSelectionStrategy(
        min_price=5,      # 股价不低于5元
        max_price=50,     # 股价不高于50元
        min_amount=50000000,  # 成交额不低于5000万
    )
    trading_strategy = HoldTradingStrategy()  # 买入后一直持有

    # 创建回测配置
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,  # 初始10万
        max_stocks=10,           # 最多持仓10只
        rebalance_frequency="monthly",  # 每月调仓
    )

    # 运行回测
    engine = BacktestEngine(config)
    result = engine.run(data)

    # 分析结果
    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()


# 示例 2: 均线金叉选股 + 止损策略
def example_2():
    """均线金叉选股 + 固定止损策略"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import MovingAverageSelectionStrategy, FixedStopLossTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=50,
    )

    # 5日线上穿20日线时选入
    selection_strategy = MovingAverageSelectionStrategy(
        short_period=5,
        long_period=20,
        mode="golden_cross",
    )
    # 亏损5%时止损
    trading_strategy = FixedStopLossTradingStrategy(
        stop_loss_pct=0.05,
    )

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=10,
        rebalance_frequency="monthly",
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()
    analyzer.plot_equity_curve()


# 示例 3: 动量选股 + 均线交易
def example_3():
    """动量选股 + 均线交叉交易策略"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import MomentumSelectionStrategy, MovingAverageCrossTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=100,
    )

    # 选择过去20天涨幅最大的10只股票
    selection_strategy = MomentumSelectionStrategy(
        period=20,
        top_n=10,
    )
    # 5日线下穿20日线时卖出
    trading_strategy = MovingAverageCrossTradingStrategy(
        short_period=5,
        long_period=20,
    )

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=10,
        rebalance_frequency="monthly",
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()
    analyzer.generate_report()


# 示例 4: 自定义选股策略
def example_4():
    """自定义选股策略"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies.base import BaseSelectionStrategy
    from quant_project.strategies import HoldTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer
    from typing import List, Optional
    import pandas as pd

    # 自定义选股策略：选择换手率高于市场平均值的股票
    class HighTurnoverSelectionStrategy(BaseSelectionStrategy):
        def __init__(self, multiplier: float = 1.5):
            self.multiplier = multiplier

        def select(
            self,
            data: pd.DataFrame,
            date: pd.Timestamp,
            max_stocks: Optional[int] = None,
        ) -> List[str]:
            # 获取当天所有股票数据
            mask = data["date"] == date
            if not mask.any():
                return []

            candidates = data[mask].copy()

            # 计算市场平均换手率
            if "turnover" not in candidates.columns:
                return []
            avg_turnover = candidates["turnover"].mean()

            # 选择换手率高于市场平均的股票
            selected = candidates[
                candidates["turnover"] >= avg_turnover * self.multiplier
            ]["symbol"].tolist()

            # 按换手率降序排序
            selected_df = candidates[candidates["symbol"].isin(selected)].sort_values(
                "turnover", ascending=False
            )
            selected = selected_df["symbol"].tolist()

            if max_stocks is not None:
                selected = selected[:max_stocks]

            return selected

    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=50,
    )

    selection_strategy = HighTurnoverSelectionStrategy(multiplier=2.0)
    trading_strategy = HoldTradingStrategy()

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=10,
        rebalance_frequency="monthly",
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()


# 示例 5: 自定义交易策略
def example_5():
    """自定义交易策略：N天后无条件卖出"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import PriceRangeSelectionStrategy
    from quant_project.strategies.base import BaseTradingStrategy, Signal
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer
    import pandas as pd

    class NDaysExitTradingStrategy(BaseTradingStrategy):
        """持有N天后无条件卖出"""

        def __init__(self, hold_days: int = 20):
            self.hold_days = hold_days

        def generate_signals(
            self,
            data: pd.DataFrame,
            entry_date: pd.Timestamp,
            exit_date: pd.Timestamp,
        ) -> pd.DataFrame:
            mask = (data["date"] >= entry_date) & (data["date"] <= exit_date)
            trading_data = data[mask].copy().sort_values("date")

            if len(trading_data) == 0:
                return pd.DataFrame(columns=["date", "signal"])

            signals = []
            for i, row in trading_data.iterrows():
                # 计算持有天数
                days_held = (row["date"] - entry_date).days

                if days_held >= self.hold_days:
                    signals.append({"date": row["date"], "signal": Signal.EXIT})
                else:
                    signals.append({"date": row["date"], "signal": Signal.HOLD})

            return pd.DataFrame(signals)

    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=50,
    )

    selection_strategy = PriceRangeSelectionStrategy(min_price=5, max_price=50)
    trading_strategy = NDaysExitTradingStrategy(hold_days=30)  # 持有30天后卖出

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=10,
        rebalance_frequency="monthly",
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_1,
        "2": example_2,
        "3": example_3,
        "4": example_4,
        "5": example_5,
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = input(
            "请选择要运行的示例 (1-5):\n"
            "  1. 价格范围选股 + 持有策略\n"
            "  2. 均线金叉选股 + 止损策略\n"
            "  3. 动量选股 + 均线交易\n"
            "  4. 自定义选股策略\n"
            "  5. 自定义交易策略\n"
            "请输入: "
        )

    if example_num in examples:
        print(f"\n运行示例 {example_num}...\n")
        examples[example_num]()
    else:
        print("无效的示例编号")
