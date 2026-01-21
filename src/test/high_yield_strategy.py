"""
中短期高收益策略使用示例

目标：5个交易日内获得10%以上的收益率

股票范围：A股主板（非科创板、非创业板）
- 上海主板：600xxx、601xxx、603xxx、605xxx
- 深圳主板：000xxx、001xxx

策略说明：
1. HighYieldShortTermSelectionStrategy - 综合选股策略
   - 短期强势：过去3-5天涨幅筛选
   - 量价配合：成交量放大、换手率适中
   - 突破信号：突破近期高点
   - 技术指标：RSI不超买、MACD金叉
   - 价格过滤：排除ST股、超低价股、高价股

2. FiveDayHoldTradingStrategy - 5日持有交易策略
   - 固定持有5个交易日
   - 止损：亏损达到5%时止损
   - 移动止盈：盈利达到8%时启动移动止盈（最高价回撤10%时卖出）

3. MomentumBreakthroughSelectionStrategy - 动量突破策略（更激进）
   - 专注于捕捉强势股的突破机会
   - 选择过去N天涨幅排名靠前的股票
   - 突破确认+量能确认+MACD确认
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from quant_project.strategies import (
    is_main_board,
    HighYieldShortTermSelectionStrategy,
    FiveDayHoldTradingStrategy,
    MomentumBreakthroughSelectionStrategy,
)
from quant_project.backtest import BacktestEngine, BacktestConfig
from quant_project.data_loader import fetch_for_backtest


def run_high_yield_strategy():
    """运行中短期高收益策略回测"""

    print("=" * 60)
    print("中短期高收益策略回测 - 目标：5日10%+收益率")
    print("=" * 60)

    # 设置回测日期范围
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # 获取数据
    print(f"\n正在获取数据 ({start_date} 至 {end_date})...")
    data = fetch_for_backtest(
        start_date=start_date,
        end_date=end_date,
        max_stocks=100,  # 测试用，实际可以增加
        cache_dir="data/cache/high_yield"
    )

    if data.empty:
        print("没有获取到数据，请检查网络连接或日期范围")
        return

    print(f"数据获取完成，共 {len(data)} 条记录")

    # 显示主板过滤信息
    unique_symbols = data["symbol"].unique()
    main_board_count = sum(1 for s in unique_symbols if is_main_board(s))
    print(f"股票总数: {len(unique_symbols)} 只，主板: {main_board_count} 只")

    # 策略1：保守型高收益策略
    print("\n" + "=" * 60)
    print("策略1：HighYieldShortTermSelectionStrategy (综合选股)")
    print("=" * 60)

    selection_strategy1 = HighYieldShortTermSelectionStrategy(
        # 基础过滤
        min_price=3.0,
        max_price=50.0,
        min_amount=3000,       # 最小成交额（万元）
        min_turnover=2.0,      # 最小换手率（%）
        max_turnover=15.0,     # 最大换手率（%）

        # 涨幅筛选
        lookback_days=5,
        min_return=3.0,       # 过去5天至少涨3%
        max_return=15.0,       # 避免追太高

        # 量价配合
        volume_ratio_days=5,
        min_volume_ratio=1.3, # 成交量至少放大30%

        # 突破筛选
        use_breakout=True,
        breakout_days=20,

        # 技术指标
        use_rsi=True,
        rsi_min=30,
        rsi_max=70,
        use_macd=True,

        # 均线过滤
        use_ma=True,
        ma_short=5,
        ma_long=20,
        ma_trend="above",
    )

    trading_strategy1 = FiveDayHoldTradingStrategy(
        hold_days=5,           # 持有5个交易日
        stop_loss_pct=0.05,     # 5%止损
        trail_profit_pct=0.08,  # 8%启动移动止盈
        trail_stop_pct=0.10,    # 移动止盈回撤10%
    )

    config1 = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        selection_strategy=selection_strategy1,
        trading_strategy=trading_strategy1,
        initial_capital=100000,
        max_stocks=5,           # 最多持仓5只
        position_size="equal_weight",
        rebalance_frequency="weekly",  # 每周调仓
        commission_rate=0.0003,
        min_commission=5.0,
        slippage=0.001,        # 0.1%滑点
    )

    print(f"选股策略: {selection_strategy1}")
    print(f"交易策略: {trading_strategy1}")

    engine1 = BacktestEngine(config1)
    result1 = engine1.run(data)

    # 策略2：激进型动量突破策略
    print("\n" + "=" * 60)
    print("策略2：MomentumBreakthroughSelectionStrategy (动量突破)")
    print("=" * 60)

    selection_strategy2 = MomentumBreakthroughSelectionStrategy(
        momentum_days=10,
        top_n=5,               # 选择前5只
        breakout_days=20,
        min_volume_ratio=1.5,
        use_macd=True,
        min_price=3.0,
        max_price=100.0,
        min_amount=5000,
    )

    trading_strategy2 = FiveDayHoldTradingStrategy(
        hold_days=5,
        stop_loss_pct=0.05,
        trail_profit_pct=0.08,
        trail_stop_pct=0.10,
    )

    config2 = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        selection_strategy=selection_strategy2,
        trading_strategy=trading_strategy2,
        initial_capital=100000,
        max_stocks=5,
        position_size="equal_weight",
        rebalance_frequency="weekly",
        commission_rate=0.0003,
        min_commission=5.0,
        slippage=0.001,
    )

    print(f"选股策略: {selection_strategy2}")
    print(f"交易策略: {trading_strategy2}")

    engine2 = BacktestEngine(config2)
    result2 = engine2.run(data)

    # 输出结果
    print("\n" + "=" * 60)
    print("回测结果对比")
    print("=" * 60)

    def print_result(name, result):
        print(f"\n{name}:")
        print(f"  总收益率: {result.total_return*100:.2f}%")
        print(f"  年化收益率: {result.annual_return*100:.2f}%")
        print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  胜率: {result.win_rate*100:.2f}%")
        print(f"  交易次数: {len(result.trades)}")
        print(f"  持仓次数: {len(result.positions)}")

        # 分析5日收益率分布
        if result.positions:
            returns = [p.return_pct for p in result.positions if p.return_pct is not None]
            if returns:
                returns.sort(reverse=True)
                print(f"  最高单笔收益: {returns[0]:.2f}%")
                print(f"  最低单笔收益: {returns[-1]:.2f}%")
                print(f"  平均单笔收益: {sum(returns)/len(returns):.2f}%")
                print(f"  达标10%+次数: {len([r for r in returns if r >= 10])}/{len(returns)}")

    print_result("策略1: 综合选股", result1)
    print_result("策略2: 动量突破", result2)


def run_single_stock_test():
    """测试单只股票的选股逻辑"""
    from quant_project.data_loader import fetch_daily
    from quant_project.strategies import HighYieldShortTermSelectionStrategy
    import pandas as pd

    print("\n" + "=" * 60)
    print("单股选股测试")
    print("=" * 60)

    # 测试日期
    test_date = "2024-06-28"
    test_symbol = "sh600519"  # 贵州茅台

    # 获取数据
    data = fetch_daily(test_symbol, "2024-01-01", "2024-07-01")
    print(f"\n股票: {test_symbol}")
    print(f"日期: {test_date}")
    print(f"数据量: {len(data)} 行")

    # 创建选股策略
    strategy = HighYieldShortTermSelectionStrategy()

    # 测试选股
    selected = strategy.select(data, pd.Timestamp(test_date))
    print(f"选股结果: {'选中' if test_symbol in selected else '未选中'}")

    # 显示当天的技术指标
    stock_data = data[data["date"] == test_date]
    if not stock_data.empty:
        row = stock_data.iloc[0]
        print(f"\n当日数据:")
        print(f"  收盘价: {row['close']:.2f}")
        print(f"  最高价: {row['high']:.2f}")
        print(f"  最低价: {row['low']:.2f}")
        print(f"  成交量: {row['volume']:.0f}")
        print(f"  成交额: {row['amount']:.0f}")
        if 'turnover' in row.index:
            print(f"  换手率: {row['turnover']:.2f}%")
        if 'pct_chg' in row.index:
            print(f"  涨跌幅: {row['pct_chg']:.2f}%")


def test_board_filter():
    """测试主板股票过滤功能"""
    print("\n" + "=" * 60)
    print("主板股票过滤测试")
    print("=" * 60)

    # 测试各种股票代码
    test_symbols = [
        ("sh600000", "上海主板", True),
        ("sh601988", "上海主板(中国银行)", True),
        ("sh603259", "上海主板", True),
        ("sz000001", "深圳主板(平安银行)", True),
        ("sz001979", "深圳主板", True),
        ("sh688981", "科创板", False),
        ("sz300750", "创业板(宁德时代)", False),
        ("sz300059", "创业板", False),
    ]

    for symbol, name, expected in test_symbols:
        result = is_main_board(symbol)
        status = "通过" if result == expected else "失败"
        board_type = "主板" if result else "非主板"
        print(f"  {symbol:12s} ({name:15s}) -> {board_type} [{status}]")


if __name__ == "__main__":
    # 测试主板过滤
    test_board_filter()

    # 运行回测
    # run_high_yield_strategy()

    # 运行单股测试
    # run_single_stock_test()
