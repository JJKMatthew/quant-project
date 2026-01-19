from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, List
import pandas as pd


@dataclass
class SelectorConfig:
    # 日期范围及股票池
    start_date: str                      # 日期格式：YYYY-MM-DD
    end_date: str                        # 日期格式：YYYY-MM-DD
    symbols: Optional[List[str]] = None  # 例如 ["sh601988", "sz000001"]；为 None 时将从行情接口获取一个股票池
    # period: Optional[str] = None       
    period: str = "daily"                # 数据周期，akshare 支持 "daily", "weekly", "monthly" 等
    
    # 筛选条件（为 None 则不启用该条件）
    min_price: Optional[float] = None      # 股价，优先使用收盘价 close；若接口没有则回退 open
    max_price: Optional[float] = None      # 最高股价过滤
    min_amount: Optional[float] = None     # 最低成交额（元）
    max_amount: Optional[float] = None     # 最高成交额（元）
    min_pct_chg: Optional[float] = None    # 单位：百分比，例如 -5 表示 -5%
    max_pct_chg: Optional[float] = None    # 最高涨跌幅（%）
    min_turnover: Optional[float] = None   # 换手率，支持小数(0.05=5%)或百分数(5)；内部会做规范化
    max_turnover: Optional[float] = None   # 最高换手率

    # 输出设置
    output_dir: str = "reports"     
    output_name: str = "selection"
    to_parquet: bool = True  
    to_html: bool = True

# 工具函数【适配akshare的日期格式】
def _normalize_datestr(d: str) -> str: 
    return d.replace("-", "")  

# 确保目录存在
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_daily(symbol: str, period: str, start: str, end: str) -> pd.DataFrame:
    """通过 AkShare 获取日线数据并标准化列名。

    返回的列（若接口具备）：[date, symbol, name?, open, high, low, close, volume, amount, turnover, pct_chg]
    注意：name/turnover/pct_chg 的可用性取决于具体接口；若缺失 pct_chg 将按收盘价计算日涨跌幅。
    """
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol, 
        period=period, 
        start_date=start, 
        end_date=end,
        adjust="hfq"
        )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 统一日期为 pandas 时间类型
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # 可能原本是字符串
    else:
        df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])  # 兜底转换

    # 标准化数值列（若接口列名不同可在此扩展）
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover",
    }
    df = df.rename(columns=rename_map)

    # 添加代码列
    df["symbol"] = symbol

    # 若没有涨跌幅则根据收盘价计算（百分比）
    if "pct_chg" not in df.columns:
        df = df.sort_values("date")
        df["pct_chg"] = df["close"].pct_change() * 100.0

    # 将可能的 object 类型转为数值
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[[c for c in [
        "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"
    ] if c in df.columns]]


def select_stocks(cfg: SelectorConfig) -> pd.DataFrame:
    """抓取数据并在区间内的最后一个交易日做截面筛选。

    策略：对每个标的，在 [start, end] 区间内取最后一条记录，按条件进行过滤。
    """
    start = _normalize_datestr(cfg.start_date)
    end = _normalize_datestr(cfg.end_date)

    symbols = cfg.symbols
    if symbols is None:
        # 通过 AkShare 构建基础股票池（沪深 A 股列表）
        import akshare as ak
        stock_list = ak.stock_zh_a_spot_em()
        # 期望包含列：代码、名称、最新价、成交额、换手率…… 此处将代码映射为日线接口所需格式
        # 转换为 akshare 日线格式：shxxxxxx / szxxxxxx
        def to_symbol(code: str) -> str:
            code = str(code)
            if code.startswith("6"):
                return "sh" + code
            else:
                return "sz" + code
        symbols = [to_symbol(c) for c in stock_list["代码"].astype(str).tolist()]
        name_map = dict(zip([to_symbol(c) for c in stock_list["代码"].astype(str)], stock_list["名称"]))
    else:
        name_map = None

    rows: List[pd.DataFrame] = []
    for sym in symbols[:200]:  # 为避免频控，默认限制前 200 只；可按需调整
        try:
            df = fetch_daily(sym, start, end)
            if df.empty:
                continue
            last = df.sort_values("date").iloc[-1:]
            if name_map is not None:
                last["name"] = name_map.get(sym)
            rows.append(last)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # 若换手率看起来是百分数（>1），则规范化为小数（/100）
    if "turnover" in out.columns:
        # 经验规则：绝对值 > 1 视为百分数
        mask = out["turnover"].abs() > 1.5
        out.loc[mask, "turnover"] = out.loc[mask, "turnover"] / 100.0

    # 应用条件过滤
    def apply_range(col: str, lo: Optional[float], hi: Optional[float]):
        if col not in out.columns:
            return
        if lo is not None:
            out.loc[:, col] = out[col]
        if hi is not None:
            out.loc[:, col] = out[col]

    # 构建筛选掩码
    # 说明：以下按用户设定的最小/最大范围逐项叠加条件，缺省(None)则跳过该条件。
    # 变量选择：
    #   价格: 优先使用收盘价 close；若接口没有则回退 open。
    #   成交额(amount): 用于衡量资金活跃度与流动性。
    #   涨跌幅(pct_chg): 单日百分比变化，用于过滤异常波动或选强势/弱势股。
    #   换手率(turnover): 已转换为小数（例如 0.05 表示 5%），若用户输入 >1 视为百分比再除以 100。
    m = pd.Series(True, index=out.index)
    if cfg.min_price is not None:  # 最低股价过滤
        m &= out.get("close", out.get("open")).ge(cfg.min_price)
    if cfg.max_price is not None:  # 最高股价过滤
        m &= out.get("close", out.get("open")).le(cfg.max_price)
    if cfg.min_amount is not None and "amount" in out.columns:  # 最低成交额（元）
        m &= out["amount"].ge(cfg.min_amount)
    if cfg.max_amount is not None and "amount" in out.columns:  # 最高成交额（元）
        m &= out["amount"].le(cfg.max_amount)
    if cfg.min_pct_chg is not None and "pct_chg" in out.columns:  # 最低涨跌幅（%）
        m &= out["pct_chg"].ge(cfg.min_pct_chg)
    if cfg.max_pct_chg is not None and "pct_chg" in out.columns:  # 最高涨跌幅（%）
        m &= out["pct_chg"].le(cfg.max_pct_chg)
    if cfg.min_turnover is not None and "turnover" in out.columns:  # 最低换手率
        m &= out["turnover"].ge(cfg.min_turnover if cfg.min_turnover <= 1 else cfg.min_turnover/100.0)
    if cfg.max_turnover is not None and "turnover" in out.columns:  # 最高换手率
        m &= out["turnover"].le(cfg.max_turnover if cfg.max_turnover <= 1 else cfg.max_turnover/100.0)

    out = out[m].copy()

    # 默认按成交额降序，其次按涨跌幅降序
    sort_cols = [c for c in ["amount", "pct_chg"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    return out


def save_outputs(df: pd.DataFrame, cfg: SelectorConfig) -> List[str]:
    _ensure_dir(cfg.output_dir)
    saved: List[str] = []
    base = os.path.join(cfg.output_dir, cfg.output_name)

    if cfg.to_parquet and not df.empty:
        pq = f"{base}.parquet"
        df.to_parquet(pq, index=False)
        saved.append(pq)

    if cfg.to_html:
        html = f"{base}.html"
        # 生成带样式的 HTML 表格（预览前 500 行）
        styled = df.head(500).style.format({
            "close": "{:.2f}",
            "amount": "{:.0f}",
            "pct_chg": "{:.2f}",
            "turnover": "{:.2%}",
        }, na_rep="-")
        styled.to_html(html)
        saved.append(html)

    return saved


def run_selector(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    min_pct_chg: Optional[float] = None,
    max_pct_chg: Optional[float] = None,
    min_turnover: Optional[float] = None,
    max_turnover: Optional[float] = None,
    output_dir: str = "reports",
    output_name: str = "selection",
    to_parquet: bool = True,
    to_html: bool = True,
) -> List[str]:
    cfg = SelectorConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        min_price=min_price,
        max_price=max_price,
        min_amount=min_amount,
        max_amount=max_amount,
        min_pct_chg=min_pct_chg,
        max_pct_chg=max_pct_chg,
        min_turnover=min_turnover,
        max_turnover=max_turnover,
        output_dir=output_dir,
        output_name=output_name,
        to_parquet=to_parquet,
        to_html=to_html,
    )

    df = select_stocks(cfg)
    return save_outputs(df, cfg)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="基于 AkShare 日线数据的简单选股器")
    p.add_argument("--start", required=True, help="开始日期，格式 YYYY-MM-DD")
    p.add_argument("--end", required=True, help="结束日期，格式 YYYY-MM-DD")
    p.add_argument("--symbols", help="逗号分隔的标的代码，例如 sh601988,sz000001；缺省则自动获取股票池")
    p.add_argument("--min-price", type=float, help="最低股价（收盘价）")
    p.add_argument("--max-price", type=float, help="最高股价（收盘价）")
    p.add_argument("--min-amount", type=float, help="最低成交额（元）")
    p.add_argument("--max-amount", type=float, help="最高成交额（元）")
    p.add_argument("--min-pct", type=float, dest="min_pct_chg", help="最低涨跌幅（百分比，例如 2 表示 2%）")
    p.add_argument("--max-pct", type=float, dest="max_pct_chg", help="最高涨跌幅（百分比，例如 8 表示 8%）")
    p.add_argument("--min-turnover", type=float, help="最低换手率（支持 0.02 或 2 表示 2%）")
    p.add_argument("--max-turnover", type=float, help="最高换手率（支持 0.05 或 5 表示 5%）")
    p.add_argument("--out-dir", default="reports", help="输出目录")
    p.add_argument("--out-name", default="selection", help="输出文件名前缀")
    p.add_argument("--no-parquet", action="store_true", help="不生成 parquet 文件")
    p.add_argument("--no-html", action="store_true", help="不生成 HTML 文件")

    args = p.parse_args()
    syms = args.symbols.split(",") if args.symbols else None

    saved = run_selector(
        start_date=args.start,
        end_date=args.end,
        symbols=syms,
        min_price=args.min_price,
        max_price=args.max_price,
        min_amount=args.min_amount,
        max_amount=args.max_amount,
        min_pct_chg=args.min_pct_chg,
        max_pct_chg=args.max_pct_chg,
        min_turnover=args.min_turnover,
        max_turnover=args.max_turnover,
        output_dir=args.out_dir,
        output_name=args.out_name,
        to_parquet=not args.no_parquet,
        to_html=not args.no_html,
    )
    print("已保存:", saved)
