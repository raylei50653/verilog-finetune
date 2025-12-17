# verilog-finetune

本项目提供了一个完整的流程，用于指令微调 (instruction fine-tuning) Llama-3 8B 模型，使其专门用于 Verilog 代码生成。

## 项目概览

该项目旨在通过一个三阶段的流水线，利用高质量的 Verilog 数据集对 Llama-3 8B 模型进行微调，最终生成可移植的 GGUF 格式模型，以便在本地设备上进行推理。

## 工作流程

整个流程分为以下三个主要脚本：

### 1. 数据检查 (1_inspect_data.py)

**目的:** 初步检查用于微调的 Hugging Face 数据集，了解其结构和内容。
**操作:** 该脚本主要用于数据探索，不涉及数据修改。它加载以下两个数据集：
*   `bnadimi/PyraNet-Verilog`
*   `sonyashijin/RTL_verilog_synthetic_Claude_3.7_verified_to_compile`

### 2. 数据处理 (2_process_data.py)

**目的:** 对原始数据集进行预处理、质量过滤和格式化，以准备进行模型训练。
**操作:**
*   加载 `bnadimi/PyraNet-Verilog` 和 `sonyashijin/RTL_verilog_synthetic_Claude_3.7_verified_to_compile` 数据集。
*   对 `PyraNet-Verilog` 数据集应用严格的质量过滤器，仅保留 `rank >= 18` 的高质量条目。
*   将两个数据集的样本统一格式化为 Alpaca 指令遵循格式，以便模型理解指令和响应的结构。
*   将处理后的数据集合并、打乱，并保存到本地的 `data/processed` 目录中。

**重要提示:** 在运行 `3_train_from_local.py` 之前，必须先运行此脚本来准备训练数据。

### 3. 模型训练 (3_train_from_local.py)

**目的:** 使用处理后的本地数据集对 Llama-3 8B 模型进行高效微调，并导出为 GGUF 格式。
**操作:**
*   利用 `unsloth` 库加载 4 比特量化版本的 Llama-3 8B 模型，实现高效内存使用和更快的训练速度。
*   应用 LoRA (Low-Rank Adaptation) 适配器，这是一种参数高效的微调技术，可以显著减少训练时间和计算资源。
*   使用 `SFTTrainer` (Supervised Fine-tuning Trainer) 在本地处理好的数据集上进行模型训练。
*   训练完成后，保存 LoRA 适配器。
*   最重要的是，将微调后的模型导出为 GGUF 格式。GGUF 是一种用于在本地设备上运行大型语言模型的通用格式，支持多种推理引擎（如 `llama.cpp`）。

## 如何使用

1.  **准备数据:**
    运行数据处理脚本以准备训练数据：
    ```bash
    python scripts/2_process_data.py
    ```

2.  **模型微调:**
    数据准备好后，运行模型训练脚本：
    ```bash
    python scripts/3_train_from_local.py
    ```
    此脚本将执行微调过程，并最终在项目目录下生成 GGUF 格式的模型文件。

## 关键技术

*   **Hugging Face `datasets` & `transformers`:** 用于数据加载和模型交互。
*   **Unsloth:** 提供高效的 Llama 模型 4 比特量化和 LoRA 微调实现。
*   **LoRA (Low-Rank Adaptation):** 一种参数高效的微调技术。
*   **GGUF:** 一种用于本地推理大型语言模型的通用格式。