# 基于 Qwen2.5-7B 的多领域通用客服 LoRA 微调系统

基于 Qwen2.5-7B-Instruct + QLoRA 构建的轻量化多领域智能客服助手，单卡 RTX 3090 24G 即可完成全流程训练与推理。

## 项目概述

使用 4bit QLoRA 技术在 7B 大模型上高效微调，融合 **电商、酒店、餐饮、旅游、交通、通用指令** 六大领域数据，构建能根据问题类型给出差异化回复的通用客服助手。无需全量微调，训练时间 ~6 小时，推理显存仅需 ~12GB。

## 评估结果

| 指标 | V1 (2k) | V2 (5k) | **V3 (10k 混合)** | 提升 |
|------|:--:|:--:|:--:|:--:|
| eval_loss | 2.13 | 1.83 | **1.38** | ↓35% |
| BERTScore F1 | 0.20 | 0.23 | **0.49** | ↑140% |
| ROUGE-L F1 | 0.27 | 0.08 | **0.24** | — |
| 违规率 | 0% | 0.2% | **0%** | ✅ |

> 完整推理对比见 [结果记录.txt](customer_service_lora/结果记录.txt)，基座 vs V1 vs V2 vs V3 对 14 题测试的可视化对比。

## 快速开始

### 环境安装

```bash
git clone https://github.com/cj5281018/swift-lora.git
cd swift-lora/customer_service_lora
pip install ms-swift transformers peft bitsandbytes bert-score rouge
```

### 数据预处理

```bash
# 下载 ECD、CrossWOZ、BELLE 数据集到 datasets/ 目录
# 运行格式转换 + 多领域混合
python ecd2swift.py      # ECD → JSONL
python mix_datasets.py   # 三数据集 5:3:2 混合 → train.jsonl/val.jsonl
```

### QLoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tuner_type lora \
    --dataset train.jsonl \
    --val_dataset val.jsonl \
    --torch_dtype bfloat16 \
    --quant_bits 4 \
    --quant_method bnb \
    --num_train_epochs 2 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --max_length 2048 \
    --output_dir output \
    --metric_for_best_model loss
```

### 推理测试

```bash
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapters output/checkpoint-xxx \
    --infer_backend transformers
```

### 自动化评估

```bash
python eval_metrics.py     # BERTScore + ROUGE-L + 违规检测 + 人工抽样
```

## 项目结构

```
customer_service_lora/
├── ecd2swift.py              # ECD 数据清洗 → JSONL 格式转换
├── mix_datasets.py           # 三数据集混合 (5:3:2)
├── eval_metrics.py           # 自动化评估流水线
├── train.jsonl               # 训练集 9000 条
├── val.jsonl                 # 验证集 1000 条
├── eval_results_v3/          # V3 评估报告
├── 结果记录.txt              # 基座 vs V1/V2/V3 推理对比
└── 基于 Qwen2.5-7B+ms-swift 多领域通用客服 LoRA 微调系统文档.txt
```

## 数据集

| 数据集 | 采样量 | 领域 |
|------|:--:|------|
| ECD E-commerce Dialogue | 5,000 | 电商商品咨询、物流、售后 |
| CrossWOZ | 3,000 | 酒店、餐饮、景点、交通 |
| BELLE 1.5M | 2,000 | 通用中文问答 |

## 技术栈

`PyTorch 2.5` `Transformers` `PEFT` `QLoRA (4bit)` `BERTScore` `ROUGE-L` `Qwen2.5-7B` `ms-swift`

- **微调方法**: QLoRA — 4bit BitsAndBytes 量化 + LoRA rank=16，可训练参数仅 0.92%
- **训练硬件**: RTX 3090 24GB ×1，显存占用 ~12GB
- **训练效率**: 1 万条数据 2 epoch 约 6 小时
- **评估体系**: BERTScore (语义) + ROUGE-L (字面) + 违规正则检测 (安全)

## 迭代历程

| 版本 | 数据量 | 数据集 | key insight |
|------|:--:|------|------|
| V1 | 2,000 | ECD 电商 | 基线跑通，但所有回复都是"亲亲实在抱歉呢" |
| V2 | 5,000 | ECD 电商 + 优化 prompt | 回复开始差异化，但仍有幻觉（编造投诉电话） |
| V3 | 10,000 | ECD + CrossWOZ + BELLE | 多领域长对话使 BERTScore 翻倍，幻觉消失 |

## License

MIT License. 基于 [ms-swift](https://github.com/modelscope/swift) 框架构建。
