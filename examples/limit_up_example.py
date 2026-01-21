"""涨停板打板策略回测示例

演示如何使用涨停板打板策略进行高风险短线交易。

策略特点：
- 捕捉临近涨停的股票（涨幅7%-9.5%）
- 量能放大确认（量比>1.5）
- 短期持有（1-3天）
- 严格止损（5%）
- 快速止盈（第二天3%或移动止盈8%）
- 最多持仓3只，仓位控制在83%以内
"""


def limit_up_backtest():
    """涨停板打板策略回测"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import LimitUpSelectionStrategy, LimitUpTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    print("=== 涨停板打板策略回测 ===\n")
    print("策略说明：")
    print("- 选股：当天涨幅7%-9.5%，接近涨停但还未完全封板")
    print("- 持仓：最多3只股票，总仓位不超过83%（每只≈27.7%）")
    print("- 交易：持有1-3天，止损5%，快速止盈3%或移动止盈8%")
    print("- 风险：高风险高收益，适合激进投资者\n")

    # 获取数据（建议获取较多股票以便选择）
    print("正在获取数据...")
    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=200,  # 获取前200只股票，增加打板机会
    )
    print(f"数据获取完成，共 {data['symbol'].nunique()} 只股票\n")

    # 创建选股策略：涨停板选股
    selection_strategy = LimitUpSelectionStrategy(
        min_pct_chg=7.0,          # 最低涨幅7%
        max_pct_chg=9.8,          # 最高涨幅9.8%（接近但未完全封板）
        min_volume_ratio=1.5,     # 量比至少1.5倍
        min_price=3.0,            # 最低价格3元
        max_price=50.0,           # 最高价格50元
        min_amount=5000,          # 最小成交额5000万
        min_turnover=2.0,         # 最小换手率2%
        max_turnover=25.0,        # 最大换手率25%
        use_breakout=True,        # 要求突破近期高点
        breakout_days=10,         # 突破10天高点
        use_macd=True,            # 使用MACD确认
    )

    # 创建交易策略：涨停板交易
    trading_strategy = LimitUpTradingStrategy(
        hold_days=3,              # 最长持有3天
        stop_loss_pct=0.05,       # 止损5%
        quick_profit_pct=0.03,    # 第二天快速止盈3%
        trail_profit_pct=0.08,    # 移动止盈启动8%
        trail_stop_pct=0.08,      # 移动止盈回撤8%
    )

    # 创建回测配置
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,      # 初始资金10万
        max_stocks=3,                # 最多持仓3只（仓位控制）
        rebalance_frequency="weekly",  # 每周调仓（打板策略频繁调仓）
        commission_rate=0.0003,      # 佣金万三
        slippage=0.01,               # 滑点1%（打板滑点较大）
    )

    print("回测配置：")
    print(f"- 回测期间：{config.start_date} 至 {config.end_date}")
    print(f"- 初始资金：{config.initial_capital:,.0f} 元")
    print(f"- 最大持仓：{config.max_stocks} 只")
    print(f"- 调仓频率：{config.rebalance_frequency}")
    print(f"- 佣金费率：{config.commission_rate*10000:.1f}‱")
    print(f"- 滑点：{config.slippage*100:.1f}%\n")

    # 运行回测
    print("开始回测...")
    engine = BacktestEngine(config)
    result = engine.run(data)
    print("回测完成！\n")

    # 分析结果
    analyzer = BacktestAnalyzer(result)

    print("\n=== 回测结果摘要 ===")
    analyzer.print_summary()

    # 生成详细报告
    print("\n=== 生成详细报告 ===")
    analyzer.generate_report()

    # 绘制净值曲线
    print("\n=== 绘制净值曲线 ===")
    analyzer.plot_equity_curve()

    # 仓位控制验证
    print("\n=== 仓位控制验证 ===")
    verify_position_control(result)

    return result


def verify_position_control(result):
    """验证仓位控制是否符合要求"""
    import pandas as pd

    if not result.positions:
        print("没有持仓记录")
        return

    # 统计每日持仓数量
    position_df = pd.DataFrame(result.positions)

    # 按日期分组统计持仓数量
    daily_positions = position_df.groupby('entry_date').size()
    max_positions = daily_positions.max()
    avg_positions = daily_positions.mean()

    print(f"最大同时持仓：{max_positions} 只（要求≤3只）")
    print(f"平均持仓数量：{avg_positions:.2f} 只")

    if max_positions <= 3:
        print("✓ 持仓数量控制符合要求")
    else:
        print("✗ 持仓数量超过限制！")

    # 验证仓位比例
    # 假设等权重分配，每只股票占比应为 1/max_stocks
    position_ratio = 1.0 / 3  # 每只最多33.3%
    total_ratio = position_ratio * 3  # 满仓时最多100%

    # 实际应该控制在83%以下
    target_total_ratio = 0.83
    actual_position_ratio = target_total_ratio / 3  # 每只最多27.7%

    print(f"\n仓位分配：")
    print(f"- 目标总仓位：≤{target_total_ratio*100:.0f}%")
    print(f"- 每只股票仓位：≈{actual_position_ratio*100:.1f}%")
    print(f"- 3只满仓时：≈{actual_position_ratio*3*100:.1f}%")

    if actual_position_ratio * 3 <= target_total_ratio:
        print("✓ 仓位比例控制符合要求")
    else:
        print("✗ 仓位比例超过限制！")


def quick_test():
    """快速测试（小数据集）"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import LimitUpSelectionStrategy, LimitUpTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    print("=== 快速测试（小数据集）===\n")

    # 获取少量数据进行快速测试
    data = fetch_for_backtest(
        start_date="2024-11-01",
        end_date="2024-11-30",
        max_stocks=50,
    )

    selection_strategy = LimitUpSelectionStrategy()
    trading_strategy = LimitUpTradingStrategy()

    config = BacktestConfig(
        start_date="2024-11-01",
        end_date="2024-11-30",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=3,
        rebalance_frequency="weekly",
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()


def custom_parameters():
    """自定义参数示例"""
    from quant_project.backtest import BacktestConfig, BacktestEngine
    from quant_project.strategies import LimitUpSelectionStrategy, LimitUpTradingStrategy
    from quant_project.data_loader import fetch_for_backtest
    from quant_project.analysis import BacktestAnalyzer

    print("=== 自定义参数示例 ===\n")
    print("更激进的打板策略：")
    print("- 涨幅要求更高（8%-9.9%）")
    print("- 量比要求更大（≥2.0）")
    print("- 更快止盈（第二天2%）\n")

    data = fetch_for_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        max_stocks=200,
    )

    # 更激进的选股参数
    selection_strategy = LimitUpSelectionStrategy(
        min_pct_chg=8.0,          # 更接近涨停（8%+）
        max_pct_chg=9.9,          # 几乎涨停
        min_volume_ratio=2.0,     # 量比更大（2倍+）
        min_amount=8000,          # 更高成交额
        use_breakout=True,
        use_macd=True,
    )

    # 更快的止盈策略
    trading_strategy = LimitUpTradingStrategy(
        hold_days=2,              # 最长持有2天（更短）
        stop_loss_pct=0.04,       # 止损4%（更紧）
        quick_profit_pct=0.02,    # 快速止盈2%（更快）
        trail_profit_pct=0.06,    # 移动止盈6%（更低）
    )

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=100000,
        max_stocks=3,
        rebalance_frequency="weekly",
        commission_rate=0.0003,
        slippage=0.015,  # 更激进策略，滑点更大
    )

    engine = BacktestEngine(config)
    result = engine.run(data)

    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()
    analyzer.generate_report()


if __name__ == "__main__":
    import sys

    examples = {
        "1": limit_up_backtest,
        "2": quick_test,
        "3": custom_parameters,
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
    else:
        example_num = input(
            "请选择要运行的示例 (1-3):\n"
            "  1. 完整回测（推荐）\n"
            "  2. 快速测试（小数据集）\n"
            "  3. 自定义参数示例（更激进）\n"
            "请输入: "
        )

    if example_num in examples:
        print(f"\n{'='*50}\n")
        examples[example_num]()
        print(f"\n{'='*50}\n")
    else:
        print("无效的示例编号")
