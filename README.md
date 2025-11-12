## 量化项目 (quant-project)

使用 Python + uv 管理依赖的量化学习 / 实验仓库。

### 目录结构

```
pyproject.toml        # 项目与依赖声明 (PEP 621)
uv.lock               # 依赖锁定（需提交）
data/                 # 数据目录（raw/tmp 目录被忽略）
notebook/             # Jupyter 笔记本
src/                  # 源码 (建议: quant_project/ 包)
```

### 环境与依赖管理（使用 uv）

uv 是一个快速的 Python 包管理与构建工具，整合 venv、依赖解析和锁定。

#### 1. 安装 uv
Windows PowerShell:
```powershell
pip install uv --upgrade
# 或官方推荐方式（需要 curl）
# powershell -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
```

#### 2. 创建 / 同步虚拟环境
```powershell
# 在项目根目录执行
uv sync  # 会根据 pyproject.toml + uv.lock 创建 .venv 并安装依赖
```

#### 3. 激活虚拟环境
```powershell
.\.venv\Scripts\Activate.ps1
```

#### 4. 运行 Jupyter Lab
```powershell
uv run jupyter lab
```

#### 5. 安装新依赖
```powershell
uv add package_name
# 例如: uv add numpy
```

#### 6. 升级依赖
```powershell
uv lock --upgrade          # 重新解析版本
uv sync                    # 应用升级
```

### 代码组织建议
- 在 `src/` 下创建包目录：`src/quant_project/__init__.py`
- 函数与策略拆分：`data_loader.py`, `indicators.py`, `strategies/` 等。
- 保持 notebook 轻量：将逻辑下沉到 `src/`，notebook 只做调用与展示。

### 数据管理
- 原始/大文件放入 `data/raw/`（已在 .gitignore 中忽略）
- 中间文件放入 `data/tmp/`（忽略）
- 如需共享的示例数据（小）：放在 `data/` 根目录并提交。

### Git 提交流程
```powershell
git add .
git commit -m "feat: 初始量化项目结构"
git push origin main
```

### 创建 GitHub 仓库（两种方式）
1) 使用 GitHub CLI：
```powershell
gh repo create yourname/quant-project --private --source . --remote origin --push
```

2) 手动方式：
```powershell
git init            # 如果尚未初始化
git branch -M main
git remote add origin https://github.com/yourname/quant-project.git
git push -u origin main
```

### CI (可选) 思路
可用 GitHub Actions：安装 uv -> 同步依赖 -> 运行测试 / lint / notebook 检查。

### 常见问题 FAQ
Q: .venv 需要提交吗？
A: 不需要，已在 .gitignore。只提交 `pyproject.toml` 与 `uv.lock`。

Q: 没有 uv.lock？
A: 运行 `uv lock` 生成。首次 `uv sync` 也会生成。

Q: Python 版本？
A: `pyproject.toml` 设置为 `>=3.13`，建议本地安装 3.13 或以上。

---
欢迎扩展 README：添加策略说明、性能指标、回测截图等。

### 选股器使用示例
本项目新增模块：`quant_project.stock_selector`，基于 AkShare 日线数据进行简单条件筛选。

命令行运行（输出 parquet 与 HTML 到 `reports/selection.*`）：
```powershell
uv run python -m quant_project.stock_selector --start 2025-11-01 --end 2025-11-12 \
	--min-price 3 --max-price 30 --min-amount 5e7 --min-turnover 0.01 --min-pct -5 --max-pct 8
```

仅指定部分股票：
```powershell
uv run python -m quant_project.stock_selector --start 2025-11-01 --end 2025-11-12 \
	--symbols sh601988,sz000001 --min-turnover 0.02 --no-html
```

参数说明：
- `--start --end` 日期范围（将使用区间内最后一个交易日的截面数据）。
- `--symbols` 逗号分隔股票代码（不传则从行情接口获取全市场前若干只）。
- `--min-price / --max-price` 收盘价范围。
- `--min-amount / --max-amount` 成交额范围（单位：元）。
- `--min-pct / --max-pct` 当日涨跌幅范围（百分比）。
- `--min-turnover / --max-turnover` 换手率范围（支持填写 0.02 或 2 表示 2%）。
- `--no-parquet / --no-html` 关闭对应输出。

输出文件：
- `reports/selection.parquet`
- `reports/selection.html`（带简单样式，可浏览器查看）

后续可扩展：加入多日均线上穿/下穿、量能突增、波动率过滤、打分排序等。
