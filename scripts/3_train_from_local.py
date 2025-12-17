# 檔案路徑: scripts/3_train_from_local.py
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_from_disk 

# ==========================================
# 1. 設定與載入
# ==========================================
max_seq_length = 2048
output_dir = "models/verilog_llama3"

print("🔥 正在載入模型 (Llama-3-8B 4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 加入 LoRA 適配器
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 如果顯存還有剩，可以試著改為 32，但 16 對 Verilog 已經很夠用
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# ==========================================
# 2. 載入處理好的資料
# ==========================================
print("📂 讀取本地資料集 (data/processed)...")
try:
    dataset = load_from_disk("./data/processed")
    print(f"   ✅ 成功載入: {len(dataset)} 筆資料")
except Exception as e:
    print(f"   ❌ 載入失敗，請確認是否已執行 scripts/2_process_data.py")
    raise e

# ==========================================
# 3. 訓練參數設定
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # 對應處理腳本中的 key
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2, # 12GB VRAM 建議值
        gradient_accumulation_steps = 4, # 累積梯度，模擬 Batch Size = 8
        warmup_steps = 5,
        num_train_epochs = 1, # 跑完一輪即可
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # 節省顯存的關鍵
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # 關閉 wandb 上傳
    ),
)

# ==========================================
# 4. 開始訓練
# ==========================================
print("🚀 開始訓練...")
# 顯示顯存資訊
gpu_stats = torch.cuda.get_device_properties(0)
print(f"   GPU: {gpu_stats.name}. Max Memory: {gpu_stats.total_memory / 1024**3:.2f} GB")

trainer_stats = trainer.train()

# ==========================================
# 5. 儲存與轉檔
# ==========================================
print("💾 儲存 GGUF 模型中 (這會花一點時間)...")
# 儲存 LoRA adapter
model.save_pretrained("models/lora_adapters")

# 轉換並儲存為 GGUF (q4_k_m)
model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m")
print(f"✅ 全部完成！模型已儲存至 {output_dir}")