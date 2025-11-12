from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, List
import pandas as pd


@dataclass
class SelectorConfig:
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    symbols: Optional[List[str]] = None  # e.g. ["sh601988", "sz000001"]. If None, fetch a universe.

    # Filters (any can be None -> skip)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    min_pct_chg: Optional[float] = None   # in percent, e.g., -5, 3
    max_pct_chg: Optional[float] = None
    min_turnover: Optional[float] = None  # ratio (0.05=5%) or percent depending on data; we will normalize
    max_turnover: Optional[float] = None

    # Output
    output_dir: str = "reports"
    output_name: str = "selection"
    to_parquet: bool = True
    to_html: bool = True


def _normalize_datestr(d: str) -> str:
    return d.replace("-", "")  # akshare daily often uses YYYYMMDD


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily kline by akshare and standardize columns.

    Returns columns: [date, symbol, name?, open, high, low, close, volume, amount, turnover, pct_chg]
    Note: availability of name/turnover/pct_chg depends on API; we derive pct_chg if missing.
    """
    import akshare as ak

    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # unify date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # may be date or str
    else:
        df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])  # safety

    # standardize numeric cols presence
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

    # Add symbol
    df["symbol"] = symbol

    # Compute pct_chg if missing
    if "pct_chg" not in df.columns:
        df = df.sort_values("date")
        df["pct_chg"] = df["close"].pct_change() * 100.0

    # Some akshare fields may be object; coerce
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[[c for c in [
        "date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"
    ] if c in df.columns]]


def select_stocks(cfg: SelectorConfig) -> pd.DataFrame:
    """Fetch data and apply per-day filters on last available day in range.

    Strategy: for each symbol, take the last row within [start,end] and evaluate filters.
    """
    start = _normalize_datestr(cfg.start_date)
    end = _normalize_datestr(cfg.end_date)

    symbols = cfg.symbols
    if symbols is None:
        # Build a basic universe via akshare (沪深 A 股列表)
        import akshare as ak
        stock_list = ak.stock_zh_a_spot_em()
        # Expect columns like: 代码, 名称, 最新价, 成交额, 换手率 ... we will map code to symbol style
        # Convert to akshare daily symbol format: shxxxxxx / szxxxxxx
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
    for sym in symbols[:200]:  # limit to first 200 to avoid rate limits; adjust as needed
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

    # Normalize turnover to ratio if it looks like percentage > 1
    if "turnover" in out.columns:
        # Heuristic: values > 1 likely represent percent, convert to fraction
        mask = out["turnover"].abs() > 1.5
        out.loc[mask, "turnover"] = out.loc[mask, "turnover"] / 100.0

    # Apply filters
    def apply_range(col: str, lo: Optional[float], hi: Optional[float]):
        if col not in out.columns:
            return
        if lo is not None:
            out.loc[:, col] = out[col]
        if hi is not None:
            out.loc[:, col] = out[col]

    # Build mask
    m = pd.Series(True, index=out.index)
    if cfg.min_price is not None:
        m &= out.get("close", out.get("open")).ge(cfg.min_price)
    if cfg.max_price is not None:
        m &= out.get("close", out.get("open")).le(cfg.max_price)
    if cfg.min_amount is not None and "amount" in out.columns:
        m &= out["amount"].ge(cfg.min_amount)
    if cfg.max_amount is not None and "amount" in out.columns:
        m &= out["amount"].le(cfg.max_amount)
    if cfg.min_pct_chg is not None and "pct_chg" in out.columns:
        m &= out["pct_chg"].ge(cfg.min_pct_chg)
    if cfg.max_pct_chg is not None and "pct_chg" in out.columns:
        m &= out["pct_chg"].le(cfg.max_pct_chg)
    if cfg.min_turnover is not None and "turnover" in out.columns:
        m &= out["turnover"].ge(cfg.min_turnover if cfg.min_turnover <= 1 else cfg.min_turnover/100.0)
    if cfg.max_turnover is not None and "turnover" in out.columns:
        m &= out["turnover"].le(cfg.max_turnover if cfg.max_turnover <= 1 else cfg.max_turnover/100.0)

    out = out[m].copy()

    # Sort by amount desc then pct_chg desc
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
        # Format a styled table
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

    p = argparse.ArgumentParser(description="Simple stock selector based on daily metrics via akshare")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--symbols", help="Comma-separated symbols like sh601988,sz000001")
    p.add_argument("--min-price", type=float)
    p.add_argument("--max-price", type=float)
    p.add_argument("--min-amount", type=float)
    p.add_argument("--max-amount", type=float)
    p.add_argument("--min-pct", type=float, dest="min_pct_chg")
    p.add_argument("--max-pct", type=float, dest="max_pct_chg")
    p.add_argument("--min-turnover", type=float)
    p.add_argument("--max-turnover", type=float)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--out-name", default="selection")
    p.add_argument("--no-parquet", action="store_true")
    p.add_argument("--no-html", action="store_true")

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
    print("Saved:", saved)
