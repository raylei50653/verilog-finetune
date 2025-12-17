# 檔案路徑: scripts/2_process_data.py
from datasets import load_dataset, concatenate_datasets
import json
import os
import shutil

# 定義 Alpaca 格式模板
alpaca_prompt = """Below is an instruction that describes a hardware design task. Write the corresponding Verilog code.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- 1. 過濾函數：只保留 Rank >= 18 的高品質資料 ---
def filter_high_quality(example):
    try:
        # PyraNet 的 description 是一個 JSON 字串
        raw_desc = example['description']
        if not isinstance(raw_desc, str):
            return False
            
        data = json.loads(raw_desc)
        
        # 取得 rank，預設為 0
        rank = data.get('rank', 0)
        
        # 嘗試轉換為 float (因為可能是字串 "20" 或數字 20)
        try:
            rank = float(rank)
        except (ValueError, TypeError):
            rank = 0

        # 【關鍵條件】 Rank 必須 >= 18
        # 你也可以順便檢查 compile_status 是否為 "No error"，但 rank 通常已包含此隱含意義
        if rank >= 18:
            return True
            
    except Exception:
        # 如果 JSON 解析失敗，視為壞資料，過濾掉
        return False
        
    return False

# --- 2. 格式化函數 ---
def format_pyranet(example):
    # 因為我們在 filter 階段已經解析過一次，但為了 map 方便，這裡再解析一次取 description
    # 效能影響微乎其微
    instruction = ""
    try:
        data = json.loads(example['description'])
        instruction = data.get('description', "")
    except:
        pass

    output = example['code']
    
    # 確保有內容
    if not instruction: 
        instruction = "Implement the Verilog module based on the code structure."

    text = alpaca_prompt.format(instruction, "", output) + "<|end_of_text|>"
    return {"text": text}

def format_synthetic(example):
    # Synthetic 資料已經是高品質，直接使用
    instruction = example['evolved_nl']
    output = example['rtl']
    text = alpaca_prompt.format(instruction, "", output) + "<|end_of_text|>"
    return {"text": text}

def main():
    print("🧹 正在清理舊資料...")
    output_path = "./data/processed"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # 1. 處理 PyraNet (加入過濾步驟)
    print("⬇️  正在處理 PyraNet...")
    ds1 = load_dataset("bnadimi/PyraNet-Verilog", split="train")
    original_count = len(ds1)
    
    print("🔍 正在執行品質過濾 (Rank >= 18)...")
    # 使用 filter 函數
    ds1 = ds1.filter(filter_high_quality)
    filtered_count = len(ds1)
    
    print(f"   📉 過濾結果: {original_count} -> {filtered_count} 筆 (保留率: {filtered_count/original_count:.1%})")

    print("📝 正在格式化 PyraNet...")
    ds1 = ds1.map(format_pyranet, remove_columns=ds1.column_names)

    # 2. 處理 Synthetic
    print("⬇️  正在處理 Synthetic (全部保留)...")
    ds2 = load_dataset("sonyashijin/RTL_verilog_synthetic_Claude_3.7_verified_to_compile", split="train")
    ds2 = ds2.map(format_synthetic, remove_columns=ds2.column_names)
    print(f"   Synthetic 處理完成: {len(ds2)} 筆")

    # 3. 合併與打亂
    print("🔄 正在合併資料集...")
    combined = concatenate_datasets([ds1, ds2])
    combined = combined.shuffle(seed=3407)

    # 4. 存檔
    print(f"💾 正在儲存至 {output_path}...")
    combined.save_to_disk(output_path)
    print(f"✅ 資料準備完成！總筆數: {len(combined)}")
    print("   👉 請執行 scripts/3_train_from_local.py 開始訓練")

if __name__ == "__main__":
    main()