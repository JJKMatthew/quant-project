# 涨停板打板策略使用说明

## 策略概述

涨停板打板策略是一个**高风险高收益**的短线交易策略，专注于捕捉临近涨停的股票。

### 核心特点

- **选股：** 当天涨幅7%-9.5%，接近涨停但还未完全封板
- **量能：** 成交量放大（量比≥1.5倍）
- **突破：** 价格突破近期高点
- **持仓：** 最多3只股票，总仓位≤83%
- **周期：** 持有1-3天，快进快出
- **风控：** 止损5%，快速止盈3%或移动止盈8%

## 快速开始

### 1. 运行示例

```bash
# 完整回测（推荐）
uv run python examples/limit_up_example.py 1

# 快速测试（小数据集）
uv run python examples/limit_up_example.py 2

# 自定义参数示例（更激进）
uv run python examples/limit_up_example.py 3
```

### 2. 基本用法

```python
from quant_project.backtest import BacktestConfig, BacktestEngine
from quant_project.strategies import LimitUpSelectionStrategy, LimitUpTradingStrategy
from quant_project.data_loader import fetch_for_backtest
from quant_project.analysis import BacktestAnalyzer

# 获取数据
data = fetch_for_backtest(
    start_date="2024-01-01",
    end_date="2024-12-31",
    max_stocks=200,  # 建议获取较多股票以便选择
)

# 创建策略
selection_strategy = LimitUpSelectionStrategy(
    min_pct_chg=7.0,          # 最低涨幅7%
    max_pct_chg=9.8,          # 最高涨幅9.8%
    min_volume_ratio=1.5,     # 量比至少1.5倍
    min_amount=5000,          # 最小成交额5000万
)

trading_strategy = LimitUpTradingStrategy(
    hold_days=3,              # 最长持有3天
    stop_loss_pct=0.05,       # 止损5%
    quick_profit_pct=0.03,    # 第二天快速止盈3%
    trail_profit_pct=0.08,    # 移动止盈启动8%
)

# 配置回测（关键：max_stocks=3，控制仓位）
config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-12-31",
    selection_strategy=selection_strategy,
    trading_strategy=trading_strategy,
    initial_capital=100000,      # 初始资金10万
    max_stocks=3,                # 最多持仓3只（仓位控制）
    rebalance_frequency="weekly",  # 每周调仓
)

# 运行回测
engine = BacktestEngine(config)
result = engine.run(data)

# 分析结果
analyzer = BacktestAnalyzer(result)
analyzer.print_summary()
analyzer.generate_report()
```

## 策略参数说明

### 选股策略（LimitUpSelectionStrategy）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `min_pct_chg` | 7.0 | 最低涨幅（%），接近涨停但还未封板 |
| `max_pct_chg` | 9.8 | 最高涨幅（%），避免追已封板的 |
| `min_volume_ratio` | 1.5 | 最小量比，要求成交量放大 |
| `min_amount` | 5000 | 最小成交额（万元），保证流动性 |
| `min_price` | 3.0 | 最低价格（元） |
| `max_price` | 50.0 | 最高价格（元） |
| `min_turnover` | 2.0 | 最小换手率（%） |
| `max_turnover` | 25.0 | 最大换手率（%），避免过度炒作 |
| `use_breakout` | True | 是否要求突破近期高点 |
| `breakout_days` | 10 | 突破回溯天数 |
| `use_macd` | True | 是否使用MACD确认 |

### 交易策略（LimitUpTradingStrategy）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hold_days` | 3 | 最长持有天数 |
| `stop_loss_pct` | 0.05 | 止损百分比（5%） |
| `quick_profit_pct` | 0.03 | 快速止盈百分比（3%），适合第二天高开 |
| `trail_profit_pct` | 0.08 | 启动移动止盈的盈利百分比（8%） |
| `trail_stop_pct` | 0.08 | 移动止盈回撤百分比（8%） |

## 仓位控制

### 为什么要控制仓位？

打板策略风险极高，必须严格控制仓位：

- **最多持仓3只**：分散风险，避免单一股票波动过大
- **总仓位≤83%**：保留现金应对意外，每只股票约占27.7%

### 如何设置

在 `BacktestConfig` 中设置：

```python
config = BacktestConfig(
    max_stocks=3,  # 最多持仓3只
    # ... 其他参数
)
```

回测引擎会自动将资金平均分配给3只股票，每只占比约27.7%，总仓位约83%。

## 风险提示

⚠️ **高风险策略，请谨慎使用！**

1. **打板失败风险**：涨停板可能打开，导致次日低开
2. **流动性风险**：高位接盘，可能无法及时卖出
3. **波动风险**：短线波动剧烈，容易触发止损
4. **情绪风险**：追涨杀跌，容易受市场情绪影响

### 适用人群

- 有丰富交易经验的投资者
- 风险承受能力强
- 能够及时盯盘
- 心理素质好，能承受短期波动

### 建议

- **小资金试水**：先用小资金测试策略
- **严格止损**：不要心存侥幸，触发止损立即执行
- **控制频率**：不要频繁交易，注意手续费成本
- **回测验证**：充分回测后再实盘

## 参数调优建议

### 更保守的参数

适合风险厌恶型投资者：

```python
selection_strategy = LimitUpSelectionStrategy(
    min_pct_chg=6.0,          # 降低涨幅要求
    max_pct_chg=8.0,          # 避免追太高
    min_volume_ratio=1.8,     # 提高量比要求
)

trading_strategy = LimitUpTradingStrategy(
    hold_days=2,              # 更短持有期
    stop_loss_pct=0.04,       # 更紧止损
    quick_profit_pct=0.02,    # 更快止盈
)
```

### 更激进的参数

适合风险偏好型投资者：

```python
selection_strategy = LimitUpSelectionStrategy(
    min_pct_chg=8.0,          # 更接近涨停
    max_pct_chg=9.9,          # 几乎涨停
    min_volume_ratio=2.0,     # 更大量比
)

trading_strategy = LimitUpTradingStrategy(
    hold_days=3,              # 稍长持有期
    stop_loss_pct=0.06,       # 稍松止损
    trail_profit_pct=0.10,    # 追求更高收益
)
```

## 常见问题

### Q1: 为什么选股数量很少？

打板机会本身就少，符合条件的股票不多。可以：
- 增加 `max_stocks` 参数（数据获取时）
- 放宽涨幅范围（如6%-9.5%）
- 降低量比要求

### Q2: 如何提高胜率？

- 提高量比要求（如≥2.0）
- 增加技术指标确认（MACD、RSI等）
- 选择流动性更好的股票（提高min_amount）

### Q3: 如何降低风险？

- 减少持仓数量（如max_stocks=2）
- 缩短持有期（如hold_days=2）
- 更紧止损（如stop_loss_pct=0.04）

### Q4: 滑点和手续费如何设置？

打板策略滑点较大，建议：
```python
config = BacktestConfig(
    commission_rate=0.0003,  # 佣金万三
    slippage=0.01,           # 滑点1%（打板滑点较大）
)
```

## 文件位置

- **策略实现**：`src/quant_project/strategies/limit_up.py`
- **使用示例**：`examples/limit_up_example.py`
- **回测引擎**：`src/quant_project/backtest.py`

## 更多资源

- 查看 `examples/backtest_example.py` 了解其他策略示例
- 查看 `BACKTEST_README.md` 了解回测系统详细说明
- 查看 `src/quant_project/strategies/base.py` 了解如何自定义策略

## 联系与反馈

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

**免责声明**：本策略仅供学习和研究使用，不构成投资建议。实盘交易风险自负。
