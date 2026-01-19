"""数据获取模块

从 AkShare 获取股票历史数据，用于回测。
"""

from typing import List, Optional
import pandas as pd
import os
from pathlib import Path


def _normalize_datestr(d: str) -> str:
    """将 YYYY-MM-DD 转换为 YYYYMMDD"""
    return d.replace("-", "")


def _ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def to_symbol(code: str) -> str:
    """将股票代码转换为 AkShare 格式"""
    code = str(code)
    if code.startswith("6"):
        return "sh" + code
    else:
        return "sz" + code


def from_symbol(symbol: str) -> str:
    """将 AkShare 格式转换为纯代码"""
    return symbol[2:]


def fetch_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = "hfq",
) -> pd.DataFrame:
    """获取单只股票的日线数据

    Args:
        symbol: 股票代码，如 "sh601988" 或 "sz000001"
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        period: 周期，支持 "daily", "weekly", "monthly"
        adjust: 复权方式，"hfq"=前复权, "qfq"=后复权, ""=不复权

    Returns:
        包含历史数据的 DataFrame
    """
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period=period,
        start_date=_normalize_datestr(start_date),
        end_date=_normalize_datestr(end_date),
        adjust=adjust,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # 标准化日期列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    # 标准化数值列
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 添加代码列
    df["symbol"] = symbol

    # 计算涨跌幅（如果接口没有提供）
    if "pct_chg" not in df.columns:
        df = df.sort_values("date")
        df["pct_chg"] = df["close"].pct_change() * 100.0

    # 选择需要的列
    columns = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    if "turnover" in df.columns:
        columns.append("turnover")
    if "pct_chg" in df.columns:
        columns.append("pct_chg")

    return df[[c for c in columns if c in df.columns]]


def get_stock_list(max_stocks: Optional[int] = None) -> pd.DataFrame:
    """获取股票列表

    Args:
        max_stocks: 最多返回多少只股票，None 表示全部

    Returns:
        包含股票列表的 DataFrame
    """
    import akshare as ak

    stock_list = ak.stock_zh_a_spot_em()

    # 添加 AkShare 格式的代码列
    stock_list["symbol"] = stock_list["代码"].apply(to_symbol)

    # 限制数量
    if max_stocks is not None:
        stock_list = stock_list.head(max_stocks)

    return stock_list[["代码", "名称", "symbol"]]


def fetch_multiple_stocks(
    symbols: List[str],
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = "hfq",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """获取多只股票的历史数据

    Args:
        symbols: 股票代码列表
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        period: 周期，支持 "daily", "weekly", "monthly"
        adjust: 复权方式，"hfq"=前复权, "qfq"=后复权, ""=不复权
        cache_dir: 缓存目录，None 表示不缓存

    Returns:
        合并后的 DataFrame
    """
    all_data = []

    for i, symbol in enumerate(symbols):
        print(f"正在获取 {symbol} ({i+1}/{len(symbols)})...")

        # 尝试从缓存读取
        cache_file = None
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_file = cache_path / f"{symbol}_{start_date}_{end_date}_{period}_{adjust}.parquet"
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    all_data.append(df)
                    continue
                except Exception:
                    pass

        # 从网络获取
        try:
            df = fetch_daily(symbol, start_date, end_date, period, adjust)
            if not df.empty:
                all_data.append(df)

                # 缓存到文件
                if cache_dir and cache_file:
                    _ensure_dir(cache_dir)
                    df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"  获取失败: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    return result


def fetch_for_backtest(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    max_stocks: Optional[int] = 50,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """为回测获取数据

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        symbols: 指定股票代码列表，None 则自动获取
        max_stocks: 自动获取时的最大股票数
        cache_dir: 缓存目录

    Returns:
        合并后的 DataFrame
    """
    if symbols is None:
        print("正在获取股票列表...")
        stock_list = get_stock_list(max_stocks=max_stocks)
        symbols = stock_list["symbol"].tolist()
        print(f"已选择 {len(symbols)} 只股票")

    print(f"开始获取 {len(symbols)} 只股票的数据...")
    data = fetch_multiple_stocks(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )

    if not data.empty:
        print(f"获取完成，共 {len(data)} 条记录")
    else:
        print("没有获取到任何数据")

    return data
