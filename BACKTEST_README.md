# 选股策略回测器

一个灵活的选股策略回测系统，支持自定义选股策略和交易策略。

## 功能特性

- **自定义选股策略**：支持多种选股策略（价格范围、均线交叉、动量等）
- **自定义交易策略**：支持多种交易策略（持有、止损、均线交叉等）
- **灵活的时间范围**：自由选择回测的起止日期
- **完整的绩效分析**：收益率、最大回撤、夏普比率、胜率等指标
- **可视化报告**：净值曲线、回撤曲线、收益率分布等图表

## 快速开始

### 命令行使用

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 基本回测：价格范围选股 + 持有策略
uv run python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection price_range --min-price 5 --max-price 50 \\
    --trading hold

# 均线金叉选股 + 止损策略
uv run python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection ma_cross --short-ma 5 --long-ma 20 \\
    --trading fixed_stop_loss --stop-loss 0.05

# 动量选股 + 均线交易
uv run python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --selection momentum --momentum-period 20 --top-n 10 \\
    --trading ma_cross --short-ma 5 --long-ma 20

# 指定股票池进行回测
uv run python -m quant_project.backtest_runner --start 2024-01-01 --end 2024-12-31 \\
    --symbols sh601988,sz000001,sz000002 \\
    --selection price_range --trading hold
```

### Python 代码使用

```python
from quant_project.backtest import BacktestConfig, BacktestEngine
from quant_project.strategies import PriceRangeSelectionStrategy, HoldTradingStrategy
from quant_project.data_loader import fetch_for_backtest
from quant_project.analysis import BacktestAnalyzer

# 获取数据
data = fetch_for_backtest(
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_stocks=50,
)

# 创建策略
selection_strategy = PriceRangeSelectionStrategy(
    min_price=5,
    max_price=50,
)
trading_strategy = HoldTradingStrategy()

# 创建回测配置
config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-12-31",
    selection_strategy=selection_strategy,
    trading_strategy=trading_strategy,
    initial_capital=100000,
    max_stocks=10,
    rebalance_frequency="monthly",
)

# 运行回测
engine = BacktestEngine(config)
result = engine.run(data)

# 分析结果
analyzer = BacktestAnalyzer(result)
analyzer.print_summary()
analyzer.plot_equity_curve()
analyzer.generate_report()
```

## 内置策略

### 选股策略

| 策略名称 | 说明 | 参数 |
|---------|------|------|
| `price_range` | 价格范围选股 | min_price, max_price, min_amount, max_amount, min_pct_chg, max_pct_chg |
| `ma_cross` | 均线交叉选股 | short_period, long_period, mode (golden_cross/death_cross) |
| `momentum` | 动量选股 | period, top_n |

### 交易策略

| 策略名称 | 说明 | 参数 |
|---------|------|------|
| `hold` | 持有策略 | 无 |
| `fixed_stop_loss` | 固定止损策略 | stop_loss_pct, take_profit_pct |
| `ma_cross` | 均线交叉交易 | short_period, long_period, stop_loss_pct |
| `atr_stop_loss` | ATR 动态止损 | atr_period, atr_multiplier |

## 自定义策略

### 自定义选股策略

```python
from quant_project.strategies.base import BaseSelectionStrategy
import pandas as pd
from typing import List, Optional

class MySelectionStrategy(BaseSelectionStrategy):
    def select(
        self,
        data: pd.DataFrame,
        date: pd.Timestamp,
        max_stocks: Optional[int] = None,
    ) -> List[str]:
        # 在这里实现你的选股逻辑
        # 返回符合条件的股票代码列表
        return ["sh601988", "sz000001"]
```

### 自定义交易策略

```python
from quant_project.strategies.base import BaseTradingStrategy, Signal
import pandas as pd

class MyTradingStrategy(BaseTradingStrategy):
    def generate_signals(
        self,
        data: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
    ) -> pd.DataFrame:
        # 在这里实现你的交易逻辑
        # 返回包含 signal 列的 DataFrame
        # signal: 1=持有, 0=卖出, -1=强制退出
        pass
```

## 命令行参数

### 基本参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--start` | 开始日期 (YYYY-MM-DD) | - |
| `--end` | 结束日期 (YYYY-MM-DD) | - |
| `--symbols` | 指定股票池，逗号分隔 | 自动获取 |
| `--max-fetch` | 自动获取时的最大股票数 | 50 |

### 选股策略参数

| 参数 | 说明 |
|-----|------|
| `--selection` | 选股策略 (price_range/ma_cross/momentum) |
| `--min-price` | 最低股价 |
| `--max-price` | 最高股价 |
| `--short-ma` | 短期均线周期 |
| `--long-ma` | 长期均线周期 |
| `--momentum-period` | 动量周期 |
| `--top-n` | 动量选股数量 |

### 交易策略参数

| 参数 | 说明 |
|-----|------|
| `--trading` | 交易策略 (hold/fixed_stop_loss/ma_cross/atr_stop_loss) |
| `--stop-loss` | 止损百分比 |
| `--take-profit` | 止盈百分比 |
| `--atr-period` | ATR 周期 |
| `--atr-multiplier` | ATR 倍数 |

### 资金管理参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--capital` | 初始资金 | 100000 |
| `--max-stocks` | 最多持仓股票数 | 10 |
| `--rebalance` | 调仓频率 (once/weekly/monthly/quarterly) | monthly |
| `--commission` | 佣金费率 | 0.0003 |
| `--slippage` | 滑点 | 0 |

## 输出报告

回测完成后，会在 `reports/` 目录下生成以下文件：

- `backtest_report_equity.png` - 净值曲线图
- `backtest_report_drawdown.png` - 回撤曲线图
- `backtest_report_returns.png` - 收益率分布图
- `backtest_report_positions.png` - 持仓收益率图
- `backtest_report_trades.parquet` - 交易记录
- `backtest_report_positions.parquet` - 持仓记录
- `backtest_report_equity_curve.parquet` - 净值曲线数据

## 运行示例

```powershell
# 运行示例代码
uv run python examples/backtest_example.py 1
```

## 项目结构

```
src/quant_project/
├── __init__.py
├── backtest.py              # 回测引擎
├── backtest_runner.py       # 命令行接口
├── analysis.py              # 结果分析和可视化
├── data_loader.py           # 数据获取
├── stock_selector.py        # 简单选股器（旧版）
└── strategies/              # 策略模块
    ├── __init__.py
    ├── base.py              # 策略基类
    ├── selection.py        # 选股策略
    └── trading.py           # 交易策略
```

## 注意事项

1. 数据使用 AkShare 获取，首次运行需要联网下载
2. 为了避免请求过于频繁，默认只获取前50只股票
3. 数据会缓存到 `data/cache/` 目录，后续运行会使用缓存
4. 回测结果仅供参考，不构成投资建议
