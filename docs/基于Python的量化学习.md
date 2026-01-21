# 一·七计划： 股票预测与自动交易（Python）

## 目标简介  

使用 Python，把策略从数据准备 → 回测 → 简单机器学习信号 → 纸面交易 → 上线准备跑通。

券商环境：同花顺 / 银河证券（若券商提供 API，可在 Day5–Day7 进一步对接；如果暂时无 API，先用 paper trading 完成验证并与券商沟通接入）。

> 预估每日投入：3–6 小时（可根据实际调整）

---

## 环境与依赖（建议先做）
建议在虚拟环境（venv / conda）中执行：

安装（示例）：
```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scipy matplotlib scikit-learn lightgbm \
            backtesting backtrader tushare akshare sqlalchemy requests python-dotenv joblib
```

说明：
- `tushare` / `akshare`：国内常用行情与基本面数据（Tushare 需注册 token；AkShare 可免注册快速试用）。
- `backtesting`（backtesting.py）与 `backtrader`：回测框架。`backtesting.py` 轻量、上手快；`backtrader` 更灵活、容易对接券商实盘。
- `lightgbm` / `scikit-learn`：机器学习模型训练。
- `sqlalchemy` / `python-dotenv`：配置与凭证管理。
- `joblib`：保存模型。

建议项目目录（示例）
```
project/
├─ data/                # 历史数据（csv/parquet）
├─ notebooks/           # 分析与笔记本
├─ src/
│  ├─ adapters/         # 交易适配器（paper / broker）
│  ├─ strategies/       # 策略代码
│  ├─ models/           # 训练/预测脚本与模型
│  └─ utils/            # 公共工具
├─ reports/             # 回测与评估报告
├─ deployment/          # 部署脚本（docker, systemd）
└─ requirements.txt
```

---

## Day 1 — 环境搭建 + 数据接入（目标：拿到可用历史行情并绘图）
目标
- 建立虚拟环境并安装依赖；
- 注册并获取 Tushare token（可选），或直接用 AkShare；
- 下载一只或几只目标股票的日线 / 分钟数据并保存。

产出
- `data/` 下的示例数据文件（CSV/Parquet）；
- `notebooks/01_data_fetch_and_plot.ipynb`：绘制 K 线 / 成交量。

AkShare 示例（日线）：
```python
import akshare as ak
df = ak.stock_zh_a_daily(symbol="sh601988")  # 示例：工商银行
df.to_parquet("data/601988.parquet")
```

检查点
- 能用 pandas 显示收盘价曲线；
- 确认数据是否为复权或未复权（回测需决定）。

---

## Day 2 — 基线回测：SMA 交叉（目标：跑通 SMA 交叉回测）
目标
- 使用 `backtesting.py` 或 `backtrader` 实现 SMA(50,200) 或 EMA(20,50)；
- 在回测中加入手续费、滑点与头寸限制。

产出
- `notebooks/02_sma_backtest.ipynb` 或 `src/strategies/sma_cross.py`；
- 回测统计（年化收益、最大回撤、夏普比等），并保存交易明细 CSV。

backtesting.py 最小示例：
```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd

data = pd.read_parquet("data/601988.parquet").rename(columns={
    'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'
})

class SmaCross(Strategy):
    def init(self):
        self.s1 = self.I(pd.Series.rolling, self.data.Close, window=50).mean()
        self.s2 = self.I(pd.Series.rolling, self.data.Close, window=200).mean()
    def next(self):
        if crossover(self.s1, self.s2):
            self.buy()
        elif crossover(self.s2, self.s1):
            self.position.close()

bt = Backtest(data, SmaCross, cash=100000, commission=0.0003)
stats = bt.run()
bt.plot()
print(stats[['Return [%]','Sharpe Ratio','Max. Drawdown [%]']])
```

检查点
- 输出可读的回测统计并可以保存交易明细。

---

## Day 3 — 特征工程与简单 ML 信号（目标：训练一个短期上涨概率模型）
目标
- 构造技术指标特征（SMA/EMA、RSI、布林带、成交量相关）与滞后收益；
- 用 LogisticRegression 或 LightGBM 预测下一个交易日上涨概率（标签：次日收益 > 0）；
- 用滑动窗口（walk-forward）做交叉验证，避免未来信息泄露。

产出
- `src/models/train_model.py`、`models/` 下保存的模型（joblib）；
- `notebooks/03_ml_features_and_training.ipynb`：模型训练与性能评估（ROC/AUC、准确率、回测结合信号的表现对比）。

要点与示例
- 使用滚动式训练与测试（例如每次训练 2 年，测试 3 个月，滚动前移）；
- 把模型输出的概率作为信号阈值（如 prob > 0.55 买入）。

简单特征生成示例：
```python
import pandas as pd
import talib as ta  # 可选，或自己实现

df['rsi'] = ta.RSI(df['Close'].values, timeperiod=14)
df['sma20'] = df['Close'].rolling(20).mean()
df['ret1'] = df['Close'].pct_change(1).shift(-1)  # 未来标签（注意 shift 用法）
df = df.dropna()
```

检查点
- 模型在验证集上有超过随机基线的 AUC/信息系数；
- 将模型信号并入回测流程，比较有无提升。

---

## Day 4 — 真实交易规则与更严格回测（目标：模拟 A 股真实规则）
目标
- 在回测中模拟真实交易约束：T+1（当天买不能当天卖）、涨跌停限制、最小交易单位（手）、费用结构；
- 加入交易成本、滑点模型，评估策略在真实规则下表现。

产出
- `src/strategies/realistic_backtest.py`：包含 T+1 与手数约束的回测器；
- 回测报告（含手续费、滑点敏感性分析）。

注意事项
- A 股 T+1：当天买入的仓位不能当天卖出（策略、回测逻辑需体现）；
- 涨跌停：若进出场受限需考虑订单无法成交或延迟成交的逻辑。

检查点
- 回测能检测并阻止违反 T+1 的交易；
- 回测包含可配置的手续费与滑点参数。

---

## Day 5 — 纸面交易 / 交易适配器（目标：实现执行层接口并用 Paper Adapter 仿真）
目标
- 设计并实现交易适配器接口 `exchange_adapter`（方法示例：`place_order(symbol, side, qty, price)`、`cancel_order(id)`、`get_positions()`）；
- 实现 `paper_adapter`：把策略信号转化为本地模拟订单并在历史/实时 tick 上模拟成交。

产出
- `src/adapters/paper_adapter.py`（可序列化订单日志）；
- `notebooks/05_paper_trading_sim.ipynb`：在近实时数据上运行策略并记录 P&L。

关于同花顺 / 银河接入
- 与券商客服联系确认是否提供「量化/Algo API」或机构级接口；
- 若券商提供 Windows 客户端但无 API，有以下替代方案（均需券商配合与合规确认）：
  - 机构/企业接入 API（推荐与券商协商）；
  - 批量下单文件接口（生成券商指定格式文件供其系统读取）；
  - 第三方交易网关（需审查合规与稳定性）；
- 先使用 `PaperAdapter` 做至少 2 周的仿真，再推进真实接入。

检查点
- Paper Adapter 能记录每一笔模拟委托、成交与持仓历史（方便回放与审计）。

---

## Day 6 — 风控、压力测试与参数敏感性（目标：验证策略鲁棒性）
目标
- 实装风控模块（最大持仓比例、单股头寸上限、日内最大交易次数、账户最大回撤熔断）；
- 做参数敏感性/蒙特卡洛测试（对滑点、延迟、手续费扰动），并做极端情景模拟（闪崩、流动性枯竭）。

产出
- `src/utils/risk_manager.py`：集中管理风控规则；
- `reports/stress_test/`：不同情景的回测矩阵与图表。

检查点
- 当模拟大幅波动或延迟时，系统能触发熔断并记录告警；
- 风控生效前后回测统计对比，确认规则合理性。

---

## Day 7 — 部署、监控与上线准备（目标：把策略以服务化形式部署并准备小额实盘）
目标
- 实现监控与告警（日志、心跳、每日 P&L 报告；可集成邮件 / 微信 / 企业微信 / Telegram）；
- 编写上线清单（KYC/API key、最小下单量、资金限额、回滚流程）；
- 在 PaperAdapter 上稳定跑 1–2 周；若券商 API 准备就绪，准备小规模真金实验（建议 1–5% 初始资金）。

产出
- `deployment/README.md`：启动、配置、监控、回滚步骤；
- `deployment/docker-compose.yml` 或 `service` 启动脚本（systemd 示例）；
- 告警脚本（例如：当 P&L 日内下跌超过阈值发送告警）。

示例监控思路
- 心跳（每分钟写入状态到日志/数据库）；
- P&L 页面（简易 Web/静态 HTML 报表）；
- 异常告警（订单拒单、网络异常、API key 失效、超出回撤阈值）。

检查点
- 监控与告警已能触发测试警报；
- 已与券商沟通接入细节（或确认仍使用 Paper 继续验证）；
- 若准备实盘：小额真金上线时间窗口与风控生效证明文档。

---

## 常见风险与合规提示（必须重视）
- 避免过拟合：多用滚动验证、跨市场/跨时间窗口验证；
- 数据质量：复权、分红、拆股处理；分钟级别需注意补齐缺失数据；
- 交易成本：回测必须包含手续费与滑点建模；
- 合规：国内券商与交易监管对高频、撤单、异常交易有严格规则，务必与券商与合规团队沟通确认；
- 先用小资金验证真金实盘，逐步放大仓位。

---

## 可选后续（我可以继续帮你的内容）
- 生成完整的 starter repository（含上述目录、示例脚本：数据抓取、SMA 回测、Paper Adapter、简单 ML 模型、Docker 启动脚本、README）；
- 直接生成 Day1–Day3 的 Jupyter notebooks（数据拉取、SMA 回测、简单 ML）；
- 帮你拟定写给券商的接入问题清单（需要确认的 API、权限、证书、下单速率限制等）。

---

## 附：快速命令清单
- 建虚拟环境、安装依赖：
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- 运行回测（示例）：
```bash
python src/strategies/sma_cross_backtest.py
```
- 启动 paper trading 服务（示例）：
```bash
python src/adapters/paper_service.py --config deployment/config.yml
```

