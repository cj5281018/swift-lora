# 基于 Qwen2.5-7B 的多领域通用客服 LoRA 微调系统

基于 Qwen2.5-7B-Instruct 基座模型，使用 ms-swift 框架进行 QLoRA 微调，构建覆盖**电商、酒店、餐饮、旅游、交通**等多领域的通用智能客服助手。

## 项目亮点

- **多领域混合训练**：融合 ECD 电商对话、CrossWOZ 任务型对话（酒店/餐饮/景点/交通）、BELLE 通用指令三大数据集
- **4bit QLoRA 高效微调**：单卡 RTX 3090 24G 显存仅占用 ~12GB，训练 1 万条数据约 6 小时
- **完整评估体系**：BERTScore 语义匹配 + ROUGE-L 字面重叠 + 违规安全检测 + 人工抽样打分
- **迭代优化验证**：V1(2k) → V2(5k) → V3(10k)，BERTScore 从 0.20 提升至 0.49，eval_loss 降低 35%

## 评估结果

| 指标 | V1 (ECD 2k) | V2 (ECD 5k) | **V3 (10k 混合)** |
|------|:--:|:--:|:--:|
| eval_loss | 2.13 | 1.83 | **1.38** |
| BERTScore F1 | 0.20 | 0.23 | **0.49** |
| ROUGE-L F1 | 0.27 | 0.08 | **0.24** |
| 违规率 | 0% | 0.2% | **0%** |
| 回复长度比 | 1.06 | 1.19 | **1.11** |

> V3 模型在保持 0% 违规率的前提下，BERTScore 较 V1 提升 140%，回复质量从"模板化敷衍"进化为"针对性的专业客服回复"。

## 项目结构

```
├── ecd2swift.py              # ECD 原始数据 → ms-swift JSONL 格式转换
├── mix_datasets.py           # 三数据集混合脚本 (ECD + CrossWOZ + BELLE)
├── eval_metrics.py           # 自动化评估 (BERTScore + ROUGE-L + 违规检测)
├── train.jsonl               # 训练集 (9000 条)
├── val.jsonl                 # 验证集 (1000 条)
├── eval_results_v3/          # V3 评估报告
├── 结果记录.txt              # 基座 vs V1 vs V2 vs V3 推理对比
└── 基于 Qwen2.5-7B+ms-swift 多领域通用客服 LoRA 微调系统文档.txt
```

## 数据说明

| 数据集 | 采样量 | 覆盖领域 |
|------|:--:|------|
| ECD E-commerce Dialogue | 5,000 | 电商商品咨询、物流、售后 |
| CrossWOZ | 3,000 | 酒店预订、餐饮推荐、景点查询、交通出行 |
| BELLE 1.5M | 2,000 | 通用中文问答、写作、推理 |

三数据集按 5:3:2 混合，统一转换为 ms-swift ShareGPT 格式的 `messages` 结构。

## 环境准备

### 1. 安装 ms-swift 框架

```bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

### 2. 安装依赖

```bash
pip install transformers peft bitsandbytes bert-score rouge datasets
```

### 3. 下载基座模型

```bash
# 从 ModelScope 或 HuggingFace 下载 Qwen2.5-7B-Instruct
# 放置到本地路径，例如 ./models/Qwen2.5-7B-Instruct/
```

### 4. 下载数据集

- ECD: [E-commerce Dialogue Corpus](https://github.com/...)
- CrossWOZ: `huggingface-cli download GEM/CrossWOZ`
- BELLE: `huggingface-cli download BelleGroup/train_1M_CN`

## 训练流程

### Step 1: 数据预处理

```bash
# ECD 格式转换（修改 MAX_SAMPLES 控制数据量）
python ecd2swift.py

# 三数据集混合
python mix_datasets.py
```

### Step 2: QLoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model /path/to/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset train.jsonl \
    --val_dataset val.jsonl \
    --torch_dtype bfloat16 \
    --quant_bits 4 \
    --quant_method bnb \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --max_length 2048 \
    --output_dir output_v3 \
    --metric_for_best_model loss
```

### Step 3: 推理测试

```bash
swift infer \
    --model /path/to/Qwen2.5-7B-Instruct \
    --adapters output_v3/checkpoint-2200 \
    --infer_backend transformers
```

### Step 4: 自动化评估

```bash
python eval_metrics.py
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-7B-Instruct (Alibaba) |
| 微调框架 | ms-swift (ModelScope) |
| 微调方法 | QLoRA (4bit量化 + LoRA rank=16) |
| 训练优化 | Gradient Checkpointing、Gradient Accumulation |
| 评估指标 | BERTScore (bert-base-chinese)、ROUGE-L、违规关键词检测 |
| 数据集 | ECD、CrossWOZ、BELLE 1.5M |
| 硬件 | NVIDIA RTX 3090 24GB ×1 |

## License

本项目代码遵循 MIT License。使用的数据集和预训练模型遵循各自的许可协议。
