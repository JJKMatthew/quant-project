# 修改后的测试代码 - test_akshare_fixed.py
import akshare as ak
import pandas as pd
from datetime import datetime

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

# 测试3：调整日期范围获取数据
try:
    # 使用更合理的日期范围
    small_data = ak.stock_zh_a_hist(
        symbol='sh601988', 
        period="daily", 
        start_date="20240101", 
        end_date="20241231",  # 改为全年数据
        adjust="hfq"  # 添加复权参数
    )
    print(f"✅ 股票数据成功: {len(small_data)} 行")
    if len(small_data) > 0:
        print("数据前5行:")
        print(small_data.head())
    else:
        print("⚠️ 获取到0行数据，尝试其他接口...")
        
        # 尝试其他接口
        try:
            data2 = ak.stock_zh_a_daily(symbol='sh601988')
            print(f"✅ stock_zh_a_daily 成功: {len(data2)} 行")
            print(data2.head())
        except Exception as e2:
            print(f"❌ stock_zh_a_daily 失败: {e2}")
            
except Exception as e:
    print(f"❌ 股票数据失败: {e}")

print("测试完成")