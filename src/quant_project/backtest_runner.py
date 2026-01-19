"""回测器命令行接口

支持自定义选股策略和交易策略进行回测。
"""

import argparse
import sys
from typing import Optional, List

from .backtest import BacktestConfig, BacktestEngine
from .analysis import BacktestAnalyzer
from .data_loader import fetch_for_backtest, to_symbol, from_symbol

# 策略导入
from .strategies import (
    PriceRangeSelectionStrategy,
    MovingAverageSelectionStrategy,
    MomentumSelectionStrategy,
    HoldTradingStrategy,
    FixedStopLossTradingStrategy,
    MovingAverageCrossTradingStrategy,
    ATRStopLossTradingStrategy,
)


def get_selection_strategy(name: str, **kwargs) -> object:
    """根据名称获取选股策略"""
    if name == "price_range":
        return PriceRangeSelectionStrategy(
            min_price=kwargs.get("min_price"),
            max_price=kwargs.get("max_price"),
            min_amount=kwargs.get("min_amount"),
            max_amount=kwargs.get("max_amount"),
            min_pct_chg=kwargs.get("min_pct_chg"),
            max_pct_chg=kwargs.get("max_pct_chg"),
        )
    elif name == "ma_cross":
        return MovingAverageSelectionStrategy(
            short_period=kwargs.get("short_period", 5),
            long_period=kwargs.get("long_period", 20),
            mode=kwargs.get("ma_mode", "golden_cross"),
        )
    elif name == "momentum":
        return MomentumSelectionStrategy(
            period=kwargs.get("momentum_period", 20),
            top_n=kwargs.get("momentum_top_n", 20),
        )
    else:
        raise ValueError(f"未知的选股策略: {name}")


def get_trading_strategy(name: str, **kwargs) -> object:
    """根据名称获取交易策略"""
    if name == "hold":
        return HoldTradingStrategy()
    elif name == "fixed_stop_loss":
        return FixedStopLossTradingStrategy(
            stop_loss_pct=kwargs.get("stop_loss_pct", 0.05),
            take_profit_pct=kwargs.get("take_profit_pct"),
        )
    elif name == "ma_cross":
        return MovingAverageCrossTradingStrategy(
            short_period=kwargs.get("short_period", 5),
            long_period=kwargs.get("long_period", 20),
            stop_loss_pct=kwargs.get("stop_loss_pct"),
        )
    elif name == "atr_stop_loss":
        return ATRStopLossTradingStrategy(
            atr_period=kwargs.get("atr_period", 14),
            atr_multiplier=kwargs.get("atr_multiplier", 2.0),
        )
    else:
        raise ValueError(f"未知的交易策略: {name}")


def parse_symbols(symbols_str: str) -> List[str]:
    """解析股票代码字符串"""
    if not symbols_str:
        return []

    result = []
    for s in symbols_str.split(","):
        s = s.strip()
        if s:
            # 自动添加前缀
            if s.startswith("sh") or s.startswith("sz"):
                result.append(s)
            elif s.startswith("6"):
                result.append("sh" + s)
            else:
                result.append("sz" + s)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="选股策略回测器 - 自定义选股策略和交易策略进行回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用价格范围选股 + 持有策略
  python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection price_range --min-price 5 --max-price 50 \\
    --trading hold

  # 使用均线金叉选股 + 止损策略
  python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection ma_cross --short-ma 5 --long-ma 20 \\
    --trading fixed_stop_loss --stop-loss 0.05

  # 使用动量选股 + 均线交叉交易
  python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection momentum --momentum-period 20 --top-n 10 \\
    --trading ma_cross --short-ma 5 --long-ma 20

  # 指定股票池进行回测
  python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --symbols sh601988,sz000001,sz000002 \\
    --selection price_range --trading hold
        """
    )

    # 基本参数
    parser.add_argument("--start", required=True, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--symbols", help="指定股票池，逗号分隔，如 sh601988,sz000001")

    # 选股策略
    parser.add_argument(
        "--selection",
        choices=["price_range", "ma_cross", "momentum"],
        default="price_range",
        help="选股策略 (默认: price_range)",
    )

    # 价格范围选股参数
    parser.add_argument("--min-price", type=float, help="最低股价")
    parser.add_argument("--max-price", type=float, help="最高股价")
    parser.add_argument("--min-amount", type=float, help="最低成交额")
    parser.add_argument("--max-amount", type=float, help="最高成交额")
    parser.add_argument("--min-pct-chg", type=float, help="最低涨跌幅(%)")
    parser.add_argument("--max-pct-chg", type=float, help="最高涨跌幅(%)")

    # 均线选股参数
    parser.add_argument("--short-ma", type=int, default=5, help="短期均线周期")
    parser.add_argument("--long-ma", type=int, default=20, help="长期均线周期")
    parser.add_argument("--ma-mode", choices=["golden_cross", "death_cross"], default="golden_cross", help="均线模式")

    # 动量选股参数
    parser.add_argument("--momentum-period", type=int, default=20, help="动量周期")
    parser.add_argument("--top-n", type=int, default=20, help="动量选股数量")

    # 交易策略
    parser.add_argument(
        "--trading",
        choices=["hold", "fixed_stop_loss", "ma_cross", "atr_stop_loss"],
        default="hold",
        help="交易策略 (默认: hold)",
    )

    # 止损参数
    parser.add_argument("--stop-loss", type=float, default=0.05, help="止损百分比 (默认: 0.05)")
    parser.add_argument("--take-profit", type=float, help="止盈百分比")

    # ATR 参数
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 周期")
    parser.add_argument("--atr-multiplier", type=float, default=2.0, help="ATR 倍数")

    # 资金管理
    parser.add_argument("--capital", type=float, default=100000, help="初始资金 (默认: 100000)")
    parser.add_argument("--max-stocks", type=int, default=10, help="最多持仓股票数 (默认: 10)")
    parser.add_argument("--rebalance", choices=["once", "weekly", "monthly", "quarterly"], default="monthly", help="调仓频率 (默认: monthly)")

    # 费用
    parser.add_argument("--commission", type=float, default=0.0003, help="佣金费率 (默认: 0.0003)")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点 (默认: 0)")

    # 其他
    parser.add_argument("--max-fetch", type=int, default=50, help="自动获取股票时的最大数量 (默认: 50)")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--cache-dir", default="data/cache", help="缓存目录 (默认: data/cache)")
    parser.add_argument("--out-dir", default="reports", help="报告输出目录 (默认: reports)")
    parser.add_argument("--out-name", default="backtest_report", help="报告文件名前缀 (默认: backtest_report)")

    args = parser.parse_args()

    # 解析股票池
    symbols = parse_symbols(args.symbols) if args.symbols else None

    # 获取数据
    cache_dir = None if args.no_cache else args.cache_dir
    data = fetch_for_backtest(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        max_stocks=args.max_fetch,
        cache_dir=cache_dir,
    )

    if data.empty:
        print("错误: 没有获取到数据，请检查日期范围或网络连接")
        sys.exit(1)

    # 创建选股策略
    selection_strategy = get_selection_strategy(
        args.selection,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        max_amount=args.max_amount,
        min_pct_chg=args.min_pct_chg,
        max_pct_chg=args.max_pct_chg,
        short_period=args.short_ma,
        long_period=args.long_ma,
        mode=args.ma_mode,
        momentum_period=args.momentum_period,
        top_n=args.top_n,
    )

    # 创建交易策略
    trading_strategy = get_trading_strategy(
        args.trading,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        short_period=args.short_ma,
        long_period=args.long_ma,
        atr_period=args.atr_period,
        atr_multiplier=args.atr_multiplier,
    )

    # 创建回测配置
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        selection_strategy=selection_strategy,
        trading_strategy=trading_strategy,
        initial_capital=args.capital,
        max_stocks=args.max_stocks,
        rebalance_frequency=args.rebalance,
        commission_rate=args.commission,
        slippage=args.slippage,
    )

    # 运行回测
    print("\n" + "=" * 60)
    print("开始回测")
    print("=" * 60)
    engine = BacktestEngine(config)
    result = engine.run(data)

    # 分析结果
    analyzer = BacktestAnalyzer(result)
    analyzer.print_summary()

    # 生成报告
    print("\n生成回测报告...")
    report_path = analyzer.generate_report(
        output_dir=args.out_dir,
        report_name=args.out_name,
    )
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
