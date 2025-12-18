import os

# Force HF Cache to local directory to avoid re-downloading
# Must be set before importing transformers/unsloth
cache_dir = os.path.join(os.getcwd(), "hf_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")

import shutil
import glob
from unsloth import FastLanguageModel

def get_latest_checkpoint(base_dir):
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    final_adapter = os.path.join(base_dir, "final_adapter")
    
    if os.path.exists(final_adapter):
        return final_adapter
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def convert_model(adapter_path, output_name, quantization="q4_k_m"):
    if not adapter_path or not os.path.exists(adapter_path):
        print(f"❌ Skip: Path {adapter_path} not found.")
        return

    print(f"\n🚀 Found Adapter: {adapter_path}")
    print(f"📦 Using Cache: {cache_dir}")
    print(f"📦 Converting to: gguf_models/{output_name}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        cache_dir = cache_dir, # Explicitly pass cache_dir
    )

    temp_dir = "temp_gguf_export"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        # Save as GGUF (Unsloth uses the base model from cache automatically)
        model.save_pretrained_gguf(temp_dir, tokenizer, quantization_method = quantization)
        
        # Find the generated gguf file and move it to final destination
        gguf_models_dir = "gguf_models"
        os.makedirs(gguf_models_dir, exist_ok=True)
        
        found = False
        # Search in temp_dir and current working directory
        search_dirs = [temp_dir, os.getcwd()]
        
        for search_dir in search_dirs:
            for f in os.listdir(search_dir):
                if f.endswith(".gguf") and "llama-3-8b" in f: # Filter for likely candidates
                    src = os.path.join(search_dir, f)
                    dst = os.path.join(gguf_models_dir, output_name)
                    print(f"📦 Moving {src} to {dst}...")
                    shutil.move(src, dst)
                    print(f"✅ Success! GGUF saved as {dst}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print("❌ Error: No GGUF file generated in temp_dir or root.")
    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # 1. 普通 Verilog 模型
    verilog_adapter = get_latest_checkpoint("outputs")
    convert_model(verilog_adapter, "verilog-llama-3-8b.Q4_K_M.gguf")

    # 2. MG Verilog 模型
    mg_adapter = get_latest_checkpoint("outputs_mg")
    convert_model(mg_adapter, "verilog-llama-3-8b-mg.Q4_K_M.gguf")