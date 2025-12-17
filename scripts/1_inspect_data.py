# 檔案路徑: scripts/1_inspect_data.py
from datasets import load_dataset
import pandas as pd

# 設定顯示選項
pd.set_option('display.max_colwidth', 50)

def inspect_dataset(name, split="train"):
    print(f"\n{'='*20} 正在檢查: {name} {'='*20}")
    try:
        # 只下載前 5 筆
        ds = load_dataset(name, split=f"{split}[:5]")
        
        print(f"✅ 欄位名稱 (Columns): {ds.column_names}")
        print(f"✅ 範例資料 (第一筆):")
        df = ds.to_pandas()
        print(df.iloc[0])
        return ds.column_names
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return []

# 檢查兩個資料集
inspect_dataset("bnadimi/PyraNet-Verilog")
inspect_dataset("sonyashijin/RTL_verilog_synthetic_Claude_3.7_verified_to_compile")