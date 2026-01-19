# test_akshare_simple.py
import akshare as ak
import pandas as pd

print("开始测试 akshare...")

# 测试1：基础接口
try:
    trade_dates = ak.tool_trade_date_hist_sina()
    print(f"✅ 基础接口成功: {len(trade_dates)} 行")
except Exception as e:
    print(f"❌ 基础接口失败: {e}")

# 测试2：股票列表
try:
    stock_list = ak.stock_info_a_code_name()
    print(f"✅ 股票列表成功: {len(stock_list)} 只股票")
except Exception as e:
    print(f"❌ 股票列表失败: {e}")

# 测试3：少量数据
try:
    small_data = ak.stock_zh_a_hist(symbol='sh601988', period="daily", start_date="20240101", end_date="20240110")
    print(f"✅ 少量数据成功: {len(small_data)} 行")
except Exception as e:
    print(f"❌ 少量数据失败: {e}")

print("测试完成")