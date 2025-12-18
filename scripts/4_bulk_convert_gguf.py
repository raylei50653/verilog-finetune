import os
import glob
import shutil
import torch
import gc

# 1. FORCE USE OF STANDARD HOME CACHE
# Since we verified that ~/.cache/huggingface has 21GB of data, we should use it.
# This avoids partial/incomplete local caches triggering re-downloads.
active_cache = os.path.expanduser("~/.cache/huggingface")
print(f"📦 Using STANDARD HOME cache: {active_cache}")

os.environ["HF_HOME"] = active_cache
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(active_cache, "hub")

# Now import ML libraries
from unsloth import FastLanguageModel

def get_latest_checkpoint(base_dir):
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    final_adapter = os.path.join(base_dir, "final_adapter")
    
    if os.path.exists(final_adapter):
        return final_adapter
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def convert_adapter_to_gguf(adapter_path, output_base_name):
    if not adapter_path or not os.path.exists(adapter_path):
        print(f"❌ Skip: Path {adapter_path} not found.")
        return

    print(f"\n🚀 Found Adapter: {adapter_path}")
    
    # 最終輸出目錄
    gguf_models_dir = "gguf_models"
    os.makedirs(gguf_models_dir, exist_ok=True)

    # 暫存工作目錄 (絕對路徑)
    temp_base_dir = os.path.abspath("temp_gguf_process")

    # 1. 定義 FP16 永久存放路徑 (在 HF cache 內)
    fp16_cache_dir = os.path.join(active_cache, "merged_models", output_base_name + "-fp16")
    
    is_merged_exists = os.path.exists(os.path.join(fp16_cache_dir, "config.json"))

    try:
        if not is_merged_exists:
            print(f"🔄 [Step 1] Merging Adapter to Base (First time only)...")
            os.makedirs(fp16_cache_dir, exist_ok=True)
            # 只有不存在時才載入 Adapter 並合併
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = adapter_path,
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True, # 改回 True 以適應 12GB VRAM
            )
            print(f"💾 Saving merged FP16 to cache: {fp16_cache_dir}")
            model.save_pretrained_merged(fp16_cache_dir, tokenizer, save_method = "merged_16bit")
            print("✅ FP16 model saved.")
            
            # 釋放記憶體，確保後續步驟有乾淨環境
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"⏩ [Step 1] Found existing FP16 in cache. Skipping merge & write.")

    except Exception as e:
        print(f"❌ Failed in Step 1: {e}")
        return

    # 定義我們需要的量化版本
    target_quants = [
        ("q4_k_m", "Q4_K_M"),
        ("q3_k_m", "Q3_K_M")
    ]

    try:
        for q_method, suffix in target_quants:
            output_filename = f"{output_base_name}.{suffix}.gguf"
            final_path = os.path.join(gguf_models_dir, output_filename)
            
            if os.path.exists(final_path):
                print(f"⏩ Skip: {final_path} already exists.")
                continue

            print(f"\n⚙️  Processing {suffix} -> {final_path}")
            
            # 為每個量化任務建立乾淨的子目錄
            current_temp_dir = os.path.join(temp_base_dir, q_method)
            if os.path.exists(current_temp_dir):
                shutil.rmtree(current_temp_dir)
            os.makedirs(current_temp_dir)
            
            try:
                # 每次轉換前重新載入模型，確保狀態乾淨
                # 回退到使用 adapter_path，因為從 merged_fp16 + load_in_4bit 進行 GGUF 轉換不穩定
                print(f"🔄 Loading adapter for {q_method}...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = adapter_path,
                    max_seq_length = 2048,
                    dtype = None,
                    load_in_4bit = True,
                )

                # 使用 Unsloth 內建轉換
                model.save_pretrained_gguf(
                    current_temp_dir, 
                    tokenizer, 
                    quantization_method = q_method
                )
                
                # 尋找並移動檔案
                found = False
                for f in os.listdir(current_temp_dir):
                    if f.endswith(".gguf"):
                        src = os.path.join(current_temp_dir, f)
                        shutil.move(src, final_path)
                        print(f"✅ Saved: {final_path}")
                        found = True
                        break
                
                if not found:
                    print(f"❌ Error: No GGUF generated for {q_method}")

            except Exception as e:
                print(f"❌ Failed processing {q_method}: {e}")
            
            finally:
                # 每個迴圈結束都釋放記憶體
                if 'model' in locals(): del model
                if 'tokenizer' in locals(): del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
            
    finally:
        # 【關鍵】無論發生什麼事，最後一定強制刪除整個暫存資料夾
        # 這會把裡面任何自動產生的 .cache 都一併清掉
        if os.path.exists(temp_base_dir):
            print("🧹 Cleaning up temporary directories...")
            try:
                shutil.rmtree(temp_base_dir)
            except OSError as e:
                print(f"⚠️ Warning: Failed to fully clean temp dir: {e}")

    # 清理記憶體
    print("🧹 Cleaning up memory...")
    if 'model' in locals(): del model
    if 'tokenizer' in locals(): del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    verilog_adapter = get_latest_checkpoint("outputs")
    if verilog_adapter:
        convert_adapter_to_gguf(verilog_adapter, "verilog-llama-3-8b")
    else:
        print("⚠️ No checkpoints found.")
