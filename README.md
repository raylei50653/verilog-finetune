# verilog-finetune

本專案提供了一個完整的流程，用於指令微調 (instruction fine-tuning) Llama-3 8B 模型，使其專門用於 Verilog 程式碼生成。

## 專案概覽

該專案旨在透過一個多階段的管線 (pipeline)，利用高品質的 Verilog 資料集對 Llama-3 8B 模型進行微調，最終生成可攜式的 GGUF 格式模型，並透過 Ollama 進行本地推論與自動化評測。

## 工作流程

整個流程分為以下四個主要腳本：

### 1. 資料檢查 (1_inspect_data.py)

**目的:** 初步檢查用於微調的 Hugging Face 資料集，了解其結構和內容。
**操作:** 該腳本主要用於資料探索，不涉及資料修改。它載入以下兩個資料集：
*   `bnadimi/PyraNet-Verilog`
*   `sonyashijin/RTL_verilog_synthetic_Claude_3.7_verified_to_compile`

### 2. 資料處理 (2_process_data.py)

**目的:** 對原始資料集進行預處理、品質過濾和格式化，以準備進行模型訓練。
**操作:**
*   應用嚴格的品質過濾器，僅保留高品質條目。
*   將資料樣本統一格式化為 Alpaca 指令遵循格式。
*   將處理後的資料集合併並保存到本地的 `data/processed` 目錄中。

**重要提示:** 在執行 `3_train_from_local.py` 之前，必須先執行此腳本來準備訓練資料。

### 3. 模型訓練 (3_train_from_local.py)

**目的:** 使用處理後的本地資料集對 Llama-3 8B 模型進行高效微調，並導出為 GGUF 格式。
**操作:**
*   利用 `unsloth` 進行 4 位元量化與 LoRA 微調。
*   訓練完成後，將模型導出為 `gguf_models/verilog-llama-3-8b.Q4_K_M.gguf`，可直接供 llama.cpp 或 Ollama 使用。

### 4. 模型評測 (4_benchmark.py)

**目的:** 自動化評估微調後模型的 Verilog 生成能力。
**操作:**
*   透過 Ollama 執行一系列基準測試（涵蓋語法、時序邏輯、複雜架構）。
*   自動提取生成的 Verilog 程式碼並進行基礎結構檢查。
*   將結果輸出至 `benchmark_results/` 目錄。

## 如何使用

### 訓練流程

1.  **準備資料:**
    ```bash
    python scripts/2_process_data.py
    ```

2.  **模型微調:**
    ```bash
    python scripts/3_train_from_local.py
    ```

### 推論與使用 (Ollama)

本專案包含一個預先配置好的 `Modelfile`，方便快速部署。

1.  **建立模型:**
    ```bash
    ollama create verilog-llama3 -f Modelfile
    ```

2.  **互動式對話:**
    ```bash
    ollama run verilog-llama3 "Write a Verilog module for a 4-bit counter."
    ```

3.  **執行基準測試:**
    ```bash
    python scripts/4_benchmark.py
    ```

## 關鍵技術

*   **Unsloth:** 提供高效的 Llama 模型 4 位元量化和 LoRA 微調實現。
*   **LoRA (Low-Rank Adaptation):** 一種參數高效的微調技術。
*   **GGUF & Ollama:** 用於本地推論與部署。
