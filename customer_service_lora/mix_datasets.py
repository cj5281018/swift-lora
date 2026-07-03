#!/usr/bin/env python3
"""
多领域数据集混合脚本 — 统一转为 ms-swift JSONL 格式

数据源:
  1. ECD 电商客服        → 电商
  2. CrossWOZ            → 酒店 / 餐饮 / 旅游 / 交通
  3. BELLE 通用指令      → 通用中文能力

混合比例:  ECD 5 : CrossWOZ 3 : BELLE 2
总量: ~10,000 条 (MVP)
"""

import json
import random
import zipfile
from pathlib import Path
from collections import Counter

# ============ 配置 ============
BASE_DIR = Path("/newHome/20_CJ/Projects/swift")
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "customer_service_lora"

TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val.jsonl"

# 各数据集采样数量
ECD_SAMPLES = 5000       # 电商核心
CROSSWOZ_SAMPLES = 3000  # 酒店/餐饮/旅游/交通
BELLE_SAMPLES = 2000     # 通用能力

TOTAL_SAMPLES = ECD_SAMPLES + CROSSWOZ_SAMPLES + BELLE_SAMPLES
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# ============ 多领域通用客服 System Prompt ============
SYSTEM_PROMPT = """你是遇见公司的多领域智能客服助手，能够处理电商、酒店、餐饮、旅游、交通等各类服务咨询。请根据用户问题的具体领域和内容，给出针对性的回复。

回复原则：
1)【领域适配】电商问题→商品/物流/售后；酒店问题→预订/入住/设施；餐饮问题→推荐/预订/口味；旅游问题→景点/门票/攻略；其他领域类推。
2)【针对性回答】根据用户具体问题回答，不要使用千篇一律的模板。
3)【简洁专业】用1-3句话解决问题。
4)【语气亲切】适度使用"亲""呢""哦"等客服口语，但不滥用。
5)【安全边界】不得承诺具体赔偿金额，遇到无法处理的问题引导用户联系人工客服。"""


# ============ 各数据集转换函数 ============

def load_ecd(max_samples: int) -> list[dict]:
    """加载 ECD 电商数据 → ms-swift messages 格式"""
    print(f"[ECD] 加载电商数据 (目标 {max_samples} 条)...")

    records = []
    for split_file in ["train.txt", "dev.txt", "test.txt"]:
        filepath = DATASETS_DIR / "E-commerce dataset" / split_file
        if not filepath.exists():
            continue
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                label = parts[0]
                if label != "1":  # 只保留高质量回复
                    continue
                # 去分字空格
                turns = [p.replace(" ", "").strip() for p in parts[1:] if p.replace(" ", "").strip()]
                if len(turns) < 2:
                    continue
                records.append(turns)

    # 去重
    seen = set()
    unique = []
    for turns in records:
        key = "|||".join(turns)
        if key not in seen:
            seen.add(key)
            unique.append(turns)

    random.seed(RANDOM_SEED)
    random.shuffle(unique)
    sampled = unique[:max_samples]

    # 转 messages 格式
    results = []
    for turns in sampled:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, turn in enumerate(turns):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": turn})
        # 确保以 assistant 结尾
        if msgs[-1]["role"] != "assistant":
            msgs = msgs[:-1]
        if len(msgs) >= 3:
            results.append({"messages": msgs})

    print(f"  → {len(results)} 条")
    return results


def load_crosswoz(max_samples: int) -> list[dict]:
    """加载 CrossWOZ → ms-swift messages 格式
    领域: 酒店/餐馆/景点/出租/地铁
    """
    print(f"[CrossWOZ] 加载多领域对话 (目标 {max_samples} 条)...")

    all_data = {}
    with zipfile.ZipFile(DATASETS_DIR / "CrossWOZ" / "data.zip") as z:
        for split_name in ["train.json", "val.json", "test.json"]:
            with z.open(split_name) as f:
                all_data.update(json.load(f))

    records = list(all_data.values())
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    sampled = records[:max_samples]

    results = []
    domain_count = Counter()
    for rec in sampled:
        # 提取领域
        domains = set()
        for g in rec.get("goal", []):
            if len(g) >= 2:
                domains.add(g[1])

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in rec["messages"]:
            role = "user" if msg["role"] == "usr" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        if msgs[-1]["role"] != "assistant":
            msgs = msgs[:-1]
        if len(msgs) >= 3:
            results.append({"messages": msgs})
            for d in domains:
                domain_count[d] += 1

    print(f"  → {len(results)} 条")
    print(f"  领域分布: {dict(domain_count)}")
    return results


def load_belle(max_samples: int) -> list[dict]:
    """加载 BELLE 通用指令 → ms-swift messages 格式"""
    print(f"[BELLE] 加载通用指令 (目标 {max_samples} 条)...")

    filepath = DATASETS_DIR / "Belle_open_source_1M.json"
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # 过滤掉太短或太长的样本
                output = d.get("output", "")
                if len(output) < 10 or len(output) > 500:
                    continue
                records.append(d)
            except json.JSONDecodeError:
                continue

    print(f"  有效样本: {len(records)} 条")
    random.seed(RANDOM_SEED)
    random.shuffle(records)
    sampled = records[:max_samples]

    results = []
    for d in sampled:
        instruction = d.get("instruction", "")
        inp = d.get("input", "")
        output = d.get("output", "")

        # 构建 user query
        if inp:
            user_text = f"{instruction}\n{inp}"
        else:
            user_text = instruction

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ]
        results.append({"messages": msgs})

    print(f"  → {len(results)} 条")
    return results


# ============ 主流程 ============

def main():
    print("=" * 60)
    print("多领域数据集混合 → ms-swift JSONL")
    print(f"目标: ECD {ECD_SAMPLES} + CrossWOZ {CROSSWOZ_SAMPLES} + BELLE {BELLE_SAMPLES}")
    print(f"     = {TOTAL_SAMPLES} 条")
    print("=" * 60)

    # ---- 1. 加载各数据集 ----
    print("\n[1/4] 加载数据集...")
    ecd_data = load_ecd(ECD_SAMPLES)
    crosswoz_data = load_crosswoz(CROSSWOZ_SAMPLES)
    belle_data = load_belle(BELLE_SAMPLES)

    # ---- 2. 混合打乱 ----
    print(f"\n[2/4] 混合打乱...")
    all_data = ecd_data + crosswoz_data + belle_data
    random.shuffle(all_data)
    print(f"  总计: {len(all_data)} 条")

    # 统计来源分布
    print(f"  ECD: {len(ecd_data)} | CrossWOZ: {len(crosswoz_data)} | BELLE: {len(belle_data)}")

    # ---- 3. 划分训练/验证集 ----
    print(f"\n[3/4] 划分训练/验证集 ({int(TRAIN_RATIO*100)}:{int((1-TRAIN_RATIO)*100)})...")
    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    print(f"  训练: {len(train_data)} | 验证: {len(val_data)}")

    # ---- 4. 写入 JSONL ----
    print(f"\n[4/4] 写入 JSONL...")
    for filename, data in [(TRAIN_OUTPUT, train_data), (VAL_OUTPUT, val_data)]:
        with open(filename, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  {TRAIN_OUTPUT}: {len(train_data)} 条")
    print(f"  {VAL_OUTPUT}:   {len(val_data)} 条")

    # ---- 展示样本 ----
    print("\n" + "=" * 60)
    print("各数据源样本预览:")
    for label, data in [("ECD电商", ecd_data[:1]), ("CrossWOZ多领域", crosswoz_data[:1]), ("BELLE通用", belle_data[:1])]:
        print(f"\n--- [{label}] ---")
        for msg in data[0]["messages"]:
            print(f"  [{msg['role']}] {msg['content'][:120]}" + ("..." if len(msg['content']) > 120 else ""))

    print("\n" + "=" * 60)
    print(f"✅ 混合完成! {len(all_data)} 条 → {TRAIN_OUTPUT} / {VAL_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
