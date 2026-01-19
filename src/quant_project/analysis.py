"""回测结果分析和可视化

提供回测结果的详细分析和可视化功能。
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from .backtest import BacktestResult, Trade, Position


class BacktestAnalyzer:
    """回测结果分析器"""

    def __init__(self, result: BacktestResult):
        self.result = result

    def print_summary(self) -> None:
        """打印回测摘要"""
        print("=" * 60)
        print("回测摘要")
        print("=" * 60)
        print(f"选股策略: {self.result.config.selection_strategy}")
        print(f"交易策略: {self.result.config.trading_strategy}")
        print()
        print(f"回测期间: {self.result.config.start_date} ~ {self.result.config.end_date}")
        print(f"初始资金: ¥{self.result.config.initial_capital:,.0f}")
        print(f"最终资金: ¥{self.result.equity_curve.iloc[-1]['equity'] if not self.result.equity_curve.empty else 0:,.0f}")
        print()
        print("-" * 60)
        print("绩效指标")
        print("-" * 60)
        print(f"总收益率: {self.result.total_return * 100:.2f}%")
        print(f"年化收益率: {self.result.annual_return * 100:.2f}%")
        print(f"最大回撤: {self.result.max_drawdown * 100:.2f}%")
        print(f"夏普比率: {self.result.sharpe_ratio:.2f}")
        print(f"胜率: {self.result.win_rate * 100:.2f}%")
        print()
        print("-" * 60)
        print("交易统计")
        print("-" * 60)
        print(f"总交易次数: {len(self.result.trades)}")
        print(f"持仓次数: {len(self.result.positions)}")
        print(f"调仓频率: {self.result.config.rebalance_frequency}")
        print(f"佣金费率: {self.result.config.commission_rate * 100:.2f}%")
        print("=" * 60)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """获取交易记录的 DataFrame"""
        trades_data = []
        for trade in self.result.trades:
            trades_data.append({
                "date": trade.date,
                "symbol": trade.symbol,
                "action": trade.action,
                "price": trade.price,
                "shares": trade.shares,
                "amount": trade.amount,
                "commission": trade.commission,
            })
        return pd.DataFrame(trades_data)

    def get_positions_dataframe(self) -> pd.DataFrame:
        """获取持仓记录的 DataFrame"""
        positions_data = []
        for pos in self.result.positions:
            positions_data.append({
                "symbol": pos.symbol,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "shares": pos.shares,
                "exit_date": pos.exit_date,
                "exit_price": pos.exit_price,
                "return_pct": pos.return_pct,
                "amount": pos.entry_price * pos.shares,
            })
        return pd.DataFrame(positions_data)

    def print_best_positions(self, n: int = 10) -> None:
        """打印最佳持仓"""
        positions_df = self.get_positions_dataframe()
        if positions_df.empty or positions_df["return_pct"].isna().all():
            print("没有完成的持仓记录")
            return

        print("-" * 60)
        print(f"最佳 {min(n, len(positions_df))} 个持仓")
        print("-" * 60)
        best = positions_df.nlargest(n, "return_pct")
        print(best[["symbol", "entry_date", "exit_date", "return_pct", "amount"]].to_string(index=False))
        print()

    def print_worst_positions(self, n: int = 10) -> None:
        """打印最差持仓"""
        positions_df = self.get_positions_dataframe()
        if positions_df.empty or positions_df["return_pct"].isna().all():
            print("没有完成的持仓记录")
            return

        print("-" * 60)
        print(f"最差 {min(n, len(positions_df))} 个持仓")
        print("-" * 60)
        worst = positions_df.nsmallest(n, "return_pct")
        print(worst[["symbol", "entry_date", "exit_date", "return_pct", "amount"]].to_string(index=False))
        print()

    def plot_equity_curve(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """绘制净值曲线"""
        if self.result.equity_curve.empty:
            print("没有净值曲线数据")
            return None

        curve = self.result.equity_curve

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制净值曲线
        ax.plot(curve["date"], curve["equity"], label="策略净值", linewidth=2)

        # 标注最大回撤
        equity_values = curve["equity"].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak
        max_dd_idx = drawdown.argmin()

        if len(curve) > 0:
            ax.scatter(
                [curve.iloc[max_dd_idx]["date"]],
                [curve.iloc[max_dd_idx]["equity"]],
                color="red",
                s=100,
                zorder=5,
                label=f"最大回撤: {self.result.max_drawdown * 100:.2f}%",
            )

        # 设置坐标轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.set_xlabel("日期")
        ax.set_ylabel("净值")
        ax.set_title("策略净值曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_drawdown(
        self,
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """绘制回撤曲线"""
        if self.result.equity_curve.empty:
            print("没有净值曲线数据")
            return None

        curve = self.result.equity_curve
        equity_values = curve["equity"].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(
            curve["date"],
            drawdown * 100,
            0,
            alpha=0.3,
            color="red",
            label="回撤",
        )
        ax.plot(curve["date"], drawdown * 100, color="red", linewidth=1)

        # 设置坐标轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.set_xlabel("日期")
        ax.set_ylabel("回撤 (%)")
        ax.set_title("策略回撤曲线")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_returns_distribution(
        self,
        figsize: tuple = (10, 5),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """绘制收益率分布"""
        if self.result.equity_curve.empty:
            print("没有净值曲线数据")
            return None

        curve = self.result.equity_curve
        equity_values = curve["equity"].values
        daily_returns = np.diff(equity_values) / equity_values[:-1] * 100
        daily_returns = daily_returns[~np.isnan(daily_returns)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 直方图
        ax1.hist(daily_returns, bins=50, edgecolor="black", alpha=0.7)
        ax1.axvline(daily_returns.mean(), color="red", linestyle="--", linewidth=2, label=f"均值: {daily_returns.mean():.3f}%")
        ax1.axvline(daily_returns.median(), color="green", linestyle="--", linewidth=2, label=f"中位数: {daily_returns.median():.3f}%")
        ax1.set_xlabel("日收益率 (%)")
        ax1.set_ylabel("频数")
        ax1.set_title("日收益率分布")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 累积收益率
        cumulative_returns = np.cumprod(1 + daily_returns / 100) - 1
        ax2.plot(cumulative_returns * 100, linewidth=2)
        ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_xlabel("交易天数")
        ax2.set_ylabel("累积收益率 (%)")
        ax2.set_title("累积收益率")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_position_returns(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> plt.Figure:
        """绘制持仓收益率"""
        positions_df = self.get_positions_dataframe()
        if positions_df.empty or positions_df["return_pct"].isna().all():
            print("没有完成的持仓记录")
            return None

        # 过滤掉没有完成收益的持仓
        positions_df = positions_df.dropna(subset=["return_pct"]).copy()

        fig, ax = plt.subplots(figsize=figsize)

        colors = ["green" if r > 0 else "red" for r in positions_df["return_pct"]]
        ax.barh(range(len(positions_df)), positions_df["return_pct"], color=colors, alpha=0.7)

        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("收益率 (%)")
        ax.set_ylabel("持仓序号")
        ax.set_title(f"持仓收益率 (胜率: {self.result.win_rate * 100:.2f}%)")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def generate_report(
        self,
        output_dir: str = "reports",
        report_name: str = "backtest_report",
    ) -> str:
        """生成完整的回测报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        base_path = output_path / report_name

        # 生成图表
        self.plot_equity_curve(
            save_path=str(base_path.with_suffix("_equity.png")),
            show=False,
        )
        self.plot_drawdown(
            save_path=str(base_path.with_suffix("_drawdown.png")),
            show=False,
        )
        self.plot_returns_distribution(
            save_path=str(base_path.with_suffix("_returns.png")),
            show=False,
        )
        self.plot_position_returns(
            save_path=str(base_path.with_suffix("_positions.png")),
            show=False,
        )

        # 保存交易记录
        trades_df = self.get_trades_dataframe()
        if not trades_df.empty:
            trades_df.to_parquet(str(base_path.with_suffix("_trades.parquet")), index=False)

        # 保存持仓记录
        positions_df = self.get_positions_dataframe()
        if not positions_df.empty:
            positions_df.to_parquet(str(base_path.with_suffix("_positions.parquet")), index=False)

        # 保存净值曲线
        if not self.result.equity_curve.empty:
            self.result.equity_curve.to_parquet(str(base_path.with_suffix("_equity_curve.parquet")), index=False)

        return str(output_path)
