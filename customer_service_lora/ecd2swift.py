#!/usr/bin/env python3
"""
ECD E-commerce Dataset → ms-swift JSONL 格式转换脚本

功能:
1. 从 ECD 原始 tsv 文件中读取对话数据
2. 过滤低质量样本 (label=0)
3. 去除汉字间的空格（分字格式 → 正常文本）
4. 重建多轮对话历史，添加通用客服 system prompt
5. 输出 ms-swift 标准格式的 train.jsonl / val.jsonl（9:1 划分）
6. 控制总样本量 ≤ MAX_SAMPLES（MVP 阶段推荐 2000 条）
"""

import json
import os
import random
from pathlib import Path

# ============ 配置参数 ============

# ECD 原始文件路径
ECD_DIR = Path("../datasets/E-commerce dataset")
TRAIN_FILE = ECD_DIR / "train.txt"
DEV_FILE = ECD_DIR / "dev.txt"
TEST_FILE = ECD_DIR / "test.txt"

# 输出目录（当前脚本所在目录）
OUTPUT_DIR = Path(".")
TRAIN_OUTPUT = OUTPUT_DIR / "train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val.jsonl"

# 最大样本数（MVP 阶段建议 2000 条，全量训练耗时太长）
MAX_SAMPLES = 5000

# 训练/验证集划分比例
TRAIN_RATIO = 0.9

# 随机种子，保证可复现
RANDOM_SEED = 42

# ============ 通用客服 System Prompt ============
# 核心目标：针对不同问题类型给出差异化回复，而非千篇一律的模板
SYSTEM_PROMPT = """你是遇见公司的智能客服助手，请根据用户问题的具体内容，给出针对性的回复，不要使用千篇一律的模板。

回复原则：
1)【针对性回答】针对用户的具体问题回答，而非模糊敷衍。物流问题→告知发货时间/快递公司/查询方式；商品咨询→介绍规格/价格/优惠；售后问题→说明退换货流程/质保政策；投诉不满→先表达歉意，再给出具体解决方案。
2)【简洁专业】用1-3句话解决问题，不啰嗦不列步骤。
3)【语气亲切】适度使用"亲""亲亲""呢""哦"等客服口语，但不滥用。
4)【安全边界】不得承诺具体赔偿金额、不得泄露公司内部信息、遇到无法处理的问题引导用户联系人工客服。
5)【主动解决】不要总说"我帮您问问""联系下掌柜"，能直接回答的就直接回答。"""


# ============ 工具函数 ============

def desegment_text(text: str) -> str:
    """
    去除汉字之间的空格（分字/分词格式 → 正常中文文本）。

    ECD 数据中每个汉字/词之间用空格分开，例如:
        "亲亲 真的 不好意思 我们 已经 是 优惠价 了 呢"
    →   "亲亲真的不好意思我们已经是优惠价了呢"

    同时保留英文单词、数字之间的空格。

    但实际观察 ECD 数据，基本是纯中文+数字内容，且所有字符间都有空格。
    处理策略：去掉所有空格，因为中文/数字/英文单词之间在客服语境中无需空格。
    """
    # 直接去掉所有空格即可还原正常中文文本
    return text.replace(" ", "")


def parse_ecd_line(line: str) -> dict | None:
    """
    解析 ECD 单行数据。

    输入格式: label \\t turn1 \\t turn2 \\t ... \\t turnN
    - label: 0（差回复）/ 1（好回复）
    - turn1..turnN: 交替的对话轮次

    输出: {
        "label": int,
        "turns": ["utterance1", "utterance2", ...],  # 按顺序排列的对话
    }
    如果格式异常则返回 None。
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split("\t")
    if len(parts) < 3:
        # 至少要有 label + 1 个用户发言 + 1 个客服回复
        return None

    try:
        label = int(parts[0])
    except ValueError:
        return None

    # 解析所有对话轮次，并去除分字空格
    turns = []
    for turn in parts[1:]:
        clean = desegment_text(turn).strip()
        if clean:
            turns.append(clean)

    if len(turns) < 2:
        return None

    return {"label": label, "turns": turns}


def build_messages(turns: list[str]) -> list[dict]:
    """
    将对话轮次列表转换为 ms-swift 的 messages 格式。

    ECD 数据的轮次是交替的 user/assistant/user/assistant/...
    即：索引 0=user, 索引 1=assistant, 索引 2=user, 索引 3=assistant, ...

    对于客服对话，第一句通常是用户提问，负责第一句由客服回答。

    返回: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    messages = []
    for i, turn in enumerate(turns):
        if i % 2 == 0:
            # 偶数索引：用户发言
            messages.append({"role": "user", "content": turn})
        else:
            # 奇数索引：客服回复
            messages.append({"role": "assistant", "content": turn})
    return messages


def build_full_record(turns: list[str], system_prompt: str = SYSTEM_PROMPT) -> dict:
    """
    构建完整的训练样本。

    在 messages 最前面插入 system prompt，
    返回 {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    messages = build_messages(turns)
    # 在最前面插入 system prompt
    messages.insert(0, {"role": "system", "content": system_prompt})
    return {"messages": messages}


# ============ 主流程 ============

def load_and_parse(filepath: Path) -> list[dict]:
    """
    加载 ECD 文件，只保留 label=1（高质量回复）的样本。
    """
    records = []
    skipped = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parsed = parse_ecd_line(line)
            if parsed is None:
                skipped += 1
                continue
            if parsed["label"] != 1:
                # 只保留高质量回复（label=1）
                skipped += 1
                continue
            records.append(parsed)

    print(f"  文件: {filepath.name}")
    print(f"  读取行数: {line_num}, 有效高质量样本: {len(records)}, 跳过: {skipped}")
    return records


def main():
    print("=" * 60)
    print("ECD → ms-swift 格式转换")
    print("=" * 60)
    print()

    # ---------- Step 2.1: 加载所有数据源 ----------
    print("[Step 1] 加载 ECD 原始数据...")

    all_parsed = []

    for filepath in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
        if filepath.exists():
            records = load_and_parse(filepath)
            all_parsed.extend(records)
            print(f"        累计样本数: {len(all_parsed)}")
        else:
            print(f"  ⚠ 文件不存在，跳过: {filepath}")

    print(f"\n  高质量样本总数: {len(all_parsed)}")

    # ---------- Step 2.2: 去重（基于对话内容） ----------
    print("\n[Step 2] 去重处理...")
    seen = set()
    unique_records = []
    for record in all_parsed:
        # 用所有拼接文本的 hash 做去重 key
        key = "|||".join(record["turns"])
        if key not in seen:
            seen.add(key)
            unique_records.append(record)
    print(f"  去重后样本数: {len(unique_records)} (移除 {len(all_parsed) - len(unique_records)} 条重复)")

    # ---------- Step 2.3: 打乱并采样 ----------
    print(f"\n[Step 3] 随机采样 MAX_SAMPLES={MAX_SAMPLES} 条...")
    random.seed(RANDOM_SEED)
    random.shuffle(unique_records)

    if len(unique_records) > MAX_SAMPLES:
        sampled = unique_records[:MAX_SAMPLES]
        print(f"  采样后: {len(sampled)} 条")
    else:
        sampled = unique_records
        print(f"  样本不足 {MAX_SAMPLES} 条，全部保留: {len(sampled)} 条")

    # ---------- Step 2.4: 9:1 划分训练/验证集 ----------
    print("\n[Step 4] 划分训练集/验证集 (9:1)...")
    split_idx = int(len(sampled) * TRAIN_RATIO)
    train_records = sampled[:split_idx]
    val_records = sampled[split_idx:]
    print(f"  训练集: {len(train_records)} 条")
    print(f"  验证集: {len(val_records)} 条")

    # ---------- Step 2.5: 转换为 ms-swift JSONL 并写入 ----------
    print("\n[Step 5] 转换格式并写入 JSONL...")

    # 写入训练集
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for record in train_records:
            swift_record = build_full_record(record["turns"])
            # 确保最后一条消息是 assistant（ms-swift 要求）
            # 如果最后一条是 user（奇数轮次），则去掉最后一条
            if swift_record["messages"][-1]["role"] != "assistant":
                # 去掉最后一条 user 消息，使对话以 assistant 结束
                swift_record["messages"] = swift_record["messages"][:-1]
                # 如果去掉后只剩下 system，跳过这条
                if len(swift_record["messages"]) < 3:
                    continue
            f.write(json.dumps(swift_record, ensure_ascii=False) + "\n")

    # 重新统计实际写入的条数
    with open(TRAIN_OUTPUT, "r", encoding="utf-8") as f:
        train_count = sum(1 for _ in f)

    # 写入验证集
    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for record in val_records:
            swift_record = build_full_record(record["turns"])
            if swift_record["messages"][-1]["role"] != "assistant":
                swift_record["messages"] = swift_record["messages"][:-1]
                if len(swift_record["messages"]) < 3:
                    continue
            f.write(json.dumps(swift_record, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "r", encoding="utf-8") as f:
        val_count = sum(1 for _ in f)

    print(f"  实际写入:")
    print(f"    {TRAIN_OUTPUT}: {train_count} 条")
    print(f"    {VAL_OUTPUT}:   {val_count} 条")

    # ---------- Step 2.6: 展示样本 ----------
    print("\n" + "=" * 60)
    print("样本预览 (train.jsonl 前 1 条，格式化展示):")
    print("=" * 60)
    with open(TRAIN_OUTPUT, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        for msg in sample["messages"]:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 120:
                content = content[:120] + "..."
            print(f"  [{role}]: {content}")

    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print(f"   训练数据: {TRAIN_OUTPUT.resolve()}")
    print(f"   验证数据: {VAL_OUTPUT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
