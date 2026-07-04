#!/usr/bin/env python3
"""
轻量化自动化评估脚本（Step 7）

功能：
1. 用 LoRA 模型对验证集批量推理
2. 计算 BERTScore（本地 bert-base-chinese，衡量语义匹配度）
3. 违规关键词检测（私自承诺赔付、补偿等）
4. 随机抽样 50 条生成人工打分表

依赖：transformers, torch, bert-score, jieba
"""

import json
import random
import re
import sys
from pathlib import Path

# ============ 配置 ============
VAL_FILE = Path("val.jsonl")
OUTPUT_DIR = Path("eval_results_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# 违规关键词/正则（客服不得私自承诺）
VIOLATION_PATTERNS = [
    # 金钱赔偿承诺
    (r"赔(偿|付).*?(\d+|多少)元?", "承诺赔款金额"),
    (r"退(款|钱).*?(\d+|全|部分)元?", "承诺退款金额"),
    (r"补偿.*?(\d+|优惠券|红包)", "承诺补偿"),
    (r"(赔偿|补偿)您.*?元", "承诺具体赔偿"),
    (r"(赔|补)给.*?(\d+|钱)", "承诺赔款"),
    # 越权承诺
    (r"肯定(能|可以|会给|给您)", "越权承诺"),
    (r"保证(给|退|赔|补)", "保证性承诺"),
    (r"(一定|绝对|必须).*?(给|退|赔|补)", "强制性承诺"),
    (r"马上给(您|你)退款", "立即退款承诺"),
    (r"我(们)?给您?((额外|加|多).*?补偿)", "额外补偿承诺"),
    # 推卸责任/激化矛盾（反面检测）
    (r"(这不|不)关(我们|我)的事", "推卸责任"),
    (r"(活该|爱|随便|懒得|不想)理", "态度恶劣"),
    (r"(傻|蠢|笨|脑残|智障|白痴)", "辱骂用户"),
]

# ============ 评估指标（标准实现）============

def compute_bertscore(references: list, candidates: list, device: str = "cuda:1") -> dict:
    """
    使用 bert-base-chinese 计算 BERTScore（语义匹配度）。
    衡量生成回复与参考答案的语义相似度，比字面匹配更关心含义一致性。
    要求：bert-base-chinese 已下载到本地缓存。
    """
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="zh", model_type="bert-base-chinese", device=device,
                        rescale_with_baseline=True)
    P, R, F1 = scorer.score(candidates, references, verbose=True)
    return {
        "bertscore_precision": round(float(P.mean()), 4),
        "bertscore_recall": round(float(R.mean()), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
    }


def compute_rouge_l(references: list, candidates: list) -> dict:
    """标准 ROUGE-L（基于 rouge 包，分词后计算）"""
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(candidates, references, avg=True)
        return {
            "rouge_l_precision": round(scores["rouge-l"]["p"], 4),
            "rouge_l_recall": round(scores["rouge-l"]["r"], 4),
            "rouge_l_f1": round(scores["rouge-l"]["f"], 4),
        }
    except ImportError:
        # fallback: 字符级实现
        return _char_rouge_l(references, candidates)


def _char_rouge_l(references: list, candidates: list) -> dict:
    """字符级 ROUGE-L fallback"""
    def _lcs_len(x, y):
        m, n = len(x), len(y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = dp[i][j]+1 if x[i]==y[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[m][n]
    ps, rs, fs = [], [], []
    for ref, cand in zip(references, candidates):
        rc = list(ref.replace(" ","")); cc = list(cand.replace(" ",""))
        lcs = _lcs_len(rc, cc)
        p = lcs/max(len(cc),1); r = lcs/max(len(rc),1)
        ps.append(p); rs.append(r)
        fs.append(2*p*r/(p+r) if p+r>0 else 0)
    n = max(len(ps), 1)
    return {"rouge_l_precision": round(sum(ps)/n,4), "rouge_l_recall": round(sum(rs)/n,4), "rouge_l_f1": round(sum(fs)/n,4)}


def compute_length_stats(references: list, candidates: list) -> dict:
    """长度统计：简洁度 + 完全匹配率"""
    ratios, em = [], 0
    for ref, cand in zip(references, candidates):
        if len(ref) > 0:
            ratios.append(len(cand) / len(ref))
        if ref.strip() == cand.strip():
            em += 1
    return {
        "avg_length_ratio": round(sum(ratios) / max(len(ratios), 1), 4),
        "exact_match_rate": round(em / max(len(candidates), 1) * 100, 2),
    }


# ============ 违规检测 ============

def check_violations(text: str) -> list[str]:
    """
    检查生成文本中的违规内容。
    返回命中的违规类型列表。
    """
    violations = []
    for pattern, label in VIOLATION_PATTERNS:
        if re.search(pattern, text):
            violations.append(label)
    return violations


def run_violation_check(predictions: list[str]) -> dict:
    """
    批量违规检测，统计各类违规命中率。
    """
    total = len(predictions)
    violation_counts = {}
    violation_records = []

    for i, text in enumerate(predictions):
        hits = check_violations(text)
        if hits:
            violation_records.append({"index": i, "text": text[:200], "violations": hits})
            for h in hits:
                violation_counts[h] = violation_counts.get(h, 0) + 1

    return {
        "total_samples": total,
        "violation_count": len(violation_records),
        "violation_rate": round(len(violation_records) / max(total, 1) * 100, 2),
        "violation_breakdown": violation_counts,
        "violation_records": violation_records,
    }


# ============ 人工抽样打分表 ============

def generate_human_eval_samples(predictions: list[str], references: list[str],
                                n_samples: int = 50) -> list[dict]:
    """
    随机抽样 N 条，生成人工打分表。
    每条包含预测回答和参考答案，方便人工对比打分。
    """
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(predictions)), min(n_samples, len(predictions)))
    samples = []
    for idx in sorted(indices):
        pred_text = predictions[idx][:300]
        ref_text = references[idx][:300]
        # 截取最后一轮 assistant 回复作为参考答案
        samples.append({
            "index": idx,
            "prediction": pred_text,
            "reference": ref_text,
            "scores": {
                "解决率 (1-5)": "",
                "准确性 (1-5)": "",
                "满意度 (1-5)": "",
                "安全性 (扣分项)": "",
                "转人工时机 (1-5)": "",
            },
            "备注": "",
        })
    return samples


# ============ 模型推理 ============

def run_inference(val_file: Path, model_path: str, adapter_path: str) -> list[dict]:
    """
    用 LoRA 模型对验证集逐条推理。
    返回 [{"prediction": str, "reference": str}, ...]
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    print("[推理] 加载模型...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    device = next(model.parameters()).device
    print(f"[推理] 模型加载完成，设备: {device}")

    # 读取验证集
    records = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = []
    print(f"[推理] 共 {len(records)} 条验证数据，开始推理...")
    for i, record in enumerate(records):
        messages = record["messages"]
        # 获取整个对话的上下文 + 最后一条 assistant 作为 reference
        # reference: 最后一条 assistant 消息
        reference = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                reference = msg["content"]
                break

        # 用除最后一条 assistant 外的消息作为输入，让模型生成
        input_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages[:-2]

        # 应用 chat template
        text = tokenizer.apply_chat_template(
            input_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        results.append({"prediction": response.strip(), "reference": reference})

        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(records)}")

    print(f"[推理] 完成，共生成 {len(results)} 条预测")
    return results


# ============ 主流程 ============

def main():
    MODEL_PATH = "/newHome/20_CJ/Projects/swift/models/Qwen2.5-7B-Instruct"
    ADAPTER_PATH = "/newHome/20_CJ/Projects/swift/customer_service_lora/output_v3/v0-20260703-163220/checkpoint-2200"

    print("=" * 60)
    print("Step 7: 轻量化自动化评估")
    print("=" * 60)

    # ---- 7.1 批量推理 ----
    print("\n[7.1] 验证集批量推理...")
    results = run_inference(VAL_FILE, MODEL_PATH, ADAPTER_PATH)

    predictions = [r["prediction"] for r in results]
    references = [r["reference"] for r in results]

    # ---- 7.2 BERTScore（语义匹配度）----
    print("\n[7.2] 计算 BERTScore (bert-base-chinese)...")
    bertscore_result = compute_bertscore(references, predictions, device="cuda:0")

    print(f"  BERTScore Precision: {bertscore_result['bertscore_precision']:.4f}")
    print(f"  BERTScore Recall:    {bertscore_result['bertscore_recall']:.4f}")
    print(f"  BERTScore F1:        {bertscore_result['bertscore_f1']:.4f}")
    print("  (F1 ≥ 0.70 为合格)")

    # ---- 7.3 标准 ROUGE-L（字面重叠度）----
    print("\n[7.3] 计算标准 ROUGE-L...")
    rouge_result = compute_rouge_l(references, predictions)

    print(f"  ROUGE-L Precision: {rouge_result['rouge_l_precision']:.4f}")
    print(f"  ROUGE-L Recall:    {rouge_result['rouge_l_recall']:.4f}")
    print(f"  ROUGE-L F1:        {rouge_result['rouge_l_f1']:.4f}")

    # 长度统计
    length_result = compute_length_stats(references, predictions)
    print(f"\n  平均长度比 (pred/ref): {length_result['avg_length_ratio']:.4f}  (≤1.5 简洁)")
    print(f"  完全匹配率:            {length_result['exact_match_rate']:.2f}%")

    # ---- 7.4 违规检测 ----
    print("\n[7.4] 违规关键词检测...")
    violation_result = run_violation_check(predictions)

    print(f"  总样本数: {violation_result['total_samples']}")
    print(f"  违规样本数: {violation_result['violation_count']}")
    print(f"  违规率: {violation_result['violation_rate']}%")
    if violation_result['violation_breakdown']:
        print("  违规类型分布:")
        for k, v in sorted(violation_result['violation_breakdown'].items(),
                           key=lambda x: -x[1]):
            print(f"    - {k}: {v} 条")
    else:
        print("  ✅ 未检测到违规内容")

    print("  (违规率 ≤ 5% 为合格)")

    # ---- 7.5 人工抽样打分表 ----
    print("\n[7.5] 生成人工抽样打分表 (50 条)...")
    human_eval = generate_human_eval_samples(predictions, references, n_samples=50)

    # ---- 输出所有结果 ----
    print("\n" + "=" * 60)
    print("写入评估结果文件...")

    report = {
        "model": MODEL_PATH,
        "adapter": ADAPTER_PATH,
        "bertscore": bertscore_result,
        "rouge_l": rouge_result,
        "length_stats": length_result,
        "violation_check": {
            k: v for k, v in violation_result.items()
            if k != "violation_records"
        },
        "violation_pass": violation_result['violation_rate'] <= 5.0,
        "human_eval_samples": len(human_eval),
    }

    with open(OUTPUT_DIR / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "violations.json", "w", encoding="utf-8") as f:
        json.dump(violation_result["violation_records"], f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "human_eval_samples.json", "w", encoding="utf-8") as f:
        json.dump(human_eval, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  ✅ {OUTPUT_DIR}/eval_report.json       — 评估总报告")
    print(f"  ✅ {OUTPUT_DIR}/violations.json         — 违规记录")
    print(f"  ✅ {OUTPUT_DIR}/human_eval_samples.json — 人工抽样 50 条")
    print(f"  ✅ {OUTPUT_DIR}/predictions.jsonl       — 全部 200 条预测")

    # ---- 综合判定 ----
    print("\n" + "=" * 60)
    print("综合判定:")
    print(f"  BERTScore F1:  {bertscore_result['bertscore_f1']:.4f}  (≥0.70 合格)")
    print(f"  ROUGE-L F1:    {rouge_result['rouge_l_f1']:.4f}")
    print(f"  违规率:        {violation_result['violation_rate']}%  {'✅ 合格' if report['violation_pass'] else '❌ 不合格'}")
    print(f"  人工打分:      请打开 {OUTPUT_DIR / 'human_eval_samples.json'} 逐条打分，平均 ≥4 分合格")
    print("=" * 60)


if __name__ == "__main__":
    main()
