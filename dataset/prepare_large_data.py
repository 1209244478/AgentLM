"""
大规模数据集准备脚本
支持从多个开源数据源下载、清洗、合并数据，为不同规模的 MiniMind 模型准备训练数据。

推荐数据规模:
  - minimind-6 (~500M):  pretrain ~50B tokens,  SFT ~2M 条
  - minimind-7 (~1.5B):  pretrain ~200B tokens, SFT ~5M 条
  - minimind-8 (~6B):    pretrain ~1T tokens,   SFT ~10M 条

用法:
  python dataset/prepare_large_data.py --scale medium --output_dir dataset/
  python dataset/prepare_large_data.py --scale large --output_dir dataset/ --pretrain_sources wikipedia books --sft_sources openhermes
"""
import json
import os
import sys
import argparse
import hashlib
import random
import re
from pathlib import Path

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SCALE_CONFIG = {
    "small": {
        "desc": "minimind-6 (~500M) 数据规模",
        "pretrain_target_tokens": 50_000_000_000,
        "sft_target_samples": 2_000_000,
        "pretrain_output": "pretrain_large.jsonl",
        "sft_output": "sft_large.jsonl",
    },
    "medium": {
        "desc": "minimind-7 (~1.5B) 数据规模",
        "pretrain_target_tokens": 200_000_000_000,
        "sft_target_samples": 5_000_000,
        "pretrain_output": "pretrain_xl.jsonl",
        "sft_output": "sft_xl.jsonl",
    },
    "large": {
        "desc": "minimind-8 (~6B) 数据规模",
        "pretrain_target_tokens": 1_000_000_000_000,
        "sft_target_samples": 10_000_000,
        "pretrain_output": "pretrain_xxlarge.jsonl",
        "sft_output": "sft_xxlarge.jsonl",
    },
}

PRETRAIN_SOURCES = {
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.zh",
        "split": "train",
        "text_field": "text",
        "desc": "维基百科中文数据",
    },
    "wikipedia_en": {
        "dataset": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "split": "train",
        "text_field": "text",
        "desc": "维基百科英文数据",
    },
    "books": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "desc": "FineWeb-Edu 高质量网页数据",
    },
    "starcoder": {
        "dataset": "bigcode/starcoderdata",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "desc": "StarCoder 代码数据",
    },
    "mint_arxiv": {
        "dataset": "mlfoundations/MINT-1T-ArXiv",
        "subset": None,
        "split": "train",
        "text_field": None,
        "desc": "MINT-1T ArXiv 论文数据（需使用 convert_mint_arxiv.py 转换）",
    },
}

SFT_SOURCES = {
    "openhermes": {
        "dataset": "teknium/OpenHermes-2.5",
        "subset": None,
        "split": "train",
        "desc": "OpenHermes 2.5 多轮对话数据",
    },
    "alpaca_zh": {
        "dataset": "silk-road/alpaca-data-gpt4-chinese",
        "subset": None,
        "split": "train",
        "desc": "Alpaca 中文 GPT4 数据",
    },
    "sharegpt": {
        "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "subset": None,
        "split": "train",
        "desc": "ShareGPT 对话数据",
    },
    "magpie": {
        "dataset": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "subset": None,
        "split": "train",
        "desc": "Magpie Pro 高质量指令数据",
    },
}


def deduplicate_texts(texts, hash_set=None):
    if hash_set is None:
        hash_set = set()
    unique = []
    for text in texts:
        h = hashlib.md5(text.strip().encode()).hexdigest()
        if h not in hash_set:
            hash_set.add(h)
            unique.append(text)
    return unique, hash_set


def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()
    return text


def download_pretrain_source(source_name, max_samples=None, output_path=None):
    from datasets import load_dataset
    source = PRETRAIN_SOURCES[source_name]
    print(f"[Pretrain] 下载 {source_name}: {source['desc']}")

    if source_name == "mint_arxiv":
        print(f"  跳过 {source_name}，请使用 convert_mint_arxiv.py 单独转换")
        return 0

    kwargs = {"path": source["dataset"], "split": source["split"]}
    if source.get("subset"):
        kwargs["name"] = source["subset"]

    ds = load_dataset(**kwargs, streaming=True, trust_remote_code=True)
    count = 0
    hash_set = set()

    mode = "w" if output_path else None
    f = open(output_path, "w", encoding="utf-8") if output_path else None

    try:
        for sample in ds:
            text = sample.get(source["text_field"], "")
            text = clean_text(text)
            if len(text) < 100:
                continue
            unique, hash_set = deduplicate_texts([text], hash_set)
            if unique and f:
                json.dump({"text": unique[0]}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if count % 10000 == 0:
                    print(f"  已处理 {count} 条")
            if max_samples and count >= max_samples:
                break
    except KeyboardInterrupt:
        print(f"  用户中断，已处理 {count} 条")
    finally:
        if f:
            f.close()

    print(f"  完成: {count} 条")
    return count


def download_sft_source(source_name, max_samples=None, output_path=None):
    from datasets import load_dataset
    source = SFT_SOURCES[source_name]
    print(f"[SFT] 下载 {source_name}: {source['desc']}")

    kwargs = {"path": source["dataset"], "split": source["split"]}
    if source.get("subset"):
        kwargs["name"] = source["subset"]

    ds = load_dataset(**kwargs, streaming=True, trust_remote_code=True)
    count = 0
    hash_set = set()

    mode = "w" if output_path else None
    f = open(output_path, "w", encoding="utf-8") if output_path else None

    def normalize_conversation(sample):
        conversations = []
        if "conversations" in sample:
            for turn in sample["conversations"]:
                role = turn.get("from", turn.get("role", "user"))
                content = turn.get("value", turn.get("content", ""))
                role_map = {"human": "user", "gpt": "assistant", "bing": "assistant"}
                role = role_map.get(role, role)
                conversations.append({"role": role, "content": content})
        elif "instruction" in sample:
            conversations.append({"role": "user", "content": sample["instruction"]})
            if sample.get("input"):
                conversations[-1]["content"] += "\n" + sample["input"]
            conversations.append({"role": "assistant", "content": sample.get("output", "")})
        return conversations

    try:
        for sample in ds:
            conversations = normalize_conversation(sample)
            if len(conversations) < 2:
                continue
            conv_text = json.dumps(conversations, ensure_ascii=False)
            h = hashlib.md5(conv_text.encode()).hexdigest()
            if h in hash_set:
                continue
            hash_set.add(h)

            if f:
                json.dump({"conversations": conversations}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if count % 10000 == 0:
                    print(f"  已处理 {count} 条")
            if max_samples and count >= max_samples:
                break
    except KeyboardInterrupt:
        print(f"  用户中断，已处理 {count} 条")
    finally:
        if f:
            f.close()

    print(f"  完成: {count} 条")
    return count


def merge_jsonl_files(input_paths, output_path, deduplicate=True):
    print(f"[Merge] 合并 {len(input_paths)} 个文件 -> {output_path}")
    hash_set = set()
    count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in input_paths:
            if not os.path.exists(path):
                print(f"  跳过不存在的文件: {path}")
                continue
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    if deduplicate:
                        h = hashlib.md5(line.encode()).hexdigest()
                        if h in hash_set:
                            continue
                        hash_set.add(h)
                    out_f.write(line + "\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"  已合并 {count} 条")

    print(f"  合并完成: {count} 条 (去重后)")
    return count


def main():
    parser = argparse.ArgumentParser(description="MiniMind 大规模数据集准备")
    parser.add_argument("--scale", type=str, default="small", choices=["small", "medium", "large"],
                        help="数据规模: small(~500M), medium(~1.5B), large(~6B)")
    parser.add_argument("--output_dir", type=str, default=DATA_DIR, help="输出目录")
    parser.add_argument("--pretrain_sources", nargs="+", default=None,
                        choices=list(PRETRAIN_SOURCES.keys()),
                        help="预训练数据源（默认使用全部）")
    parser.add_argument("--sft_sources", nargs="+", default=None,
                        choices=list(SFT_SOURCES.keys()),
                        help="SFT数据源（默认使用全部）")
    parser.add_argument("--max_pretrain_per_source", type=int, default=None,
                        help="每个预训练数据源的最大样本数（None=不限制）")
    parser.add_argument("--max_sft_per_source", type=int, default=None,
                        help="每个SFT数据源的最大样本数（None=不限制）")
    parser.add_argument("--merge_existing", action="store_true",
                        help="是否与现有数据合并")
    args = parser.parse_args()

    config = SCALE_CONFIG[args.scale]
    print(f"=== MiniMind 数据准备 ===")
    print(f"规模: {config['desc']}")
    print(f"预训练目标: ~{config['pretrain_target_tokens'] / 1e9:.0f}B tokens")
    print(f"SFT目标: ~{config['sft_target_samples'] / 1e6:.1f}M 条")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    pretrain_sources = args.pretrain_sources or list(PRETRAIN_SOURCES.keys())
    sft_sources = args.sft_sources or list(SFT_SOURCES.keys())

    pretrain_files = []
    for source_name in pretrain_sources:
        output_path = os.path.join(args.output_dir, f"pretrain_{source_name}.jsonl")
        download_pretrain_source(source_name, max_samples=args.max_pretrain_per_source, output_path=output_path)
        if os.path.exists(output_path):
            pretrain_files.append(output_path)

    sft_files = []
    for source_name in sft_sources:
        output_path = os.path.join(args.output_dir, f"sft_{source_name}.jsonl")
        download_sft_source(source_name, max_samples=args.max_sft_per_source, output_path=output_path)
        if os.path.exists(output_path):
            sft_files.append(output_path)

    if args.merge_existing:
        existing_pretrain = os.path.join(args.output_dir, "pretrain_combined.jsonl")
        if os.path.exists(existing_pretrain):
            pretrain_files.insert(0, existing_pretrain)
        existing_sft = os.path.join(args.output_dir, "sft_combined.jsonl")
        if os.path.exists(existing_sft):
            sft_files.insert(0, existing_sft)

    if pretrain_files:
        pretrain_output = os.path.join(args.output_dir, config["pretrain_output"])
        merge_jsonl_files(pretrain_files, pretrain_output)
        print(f"\n预训练数据: {pretrain_output}")

    if sft_files:
        sft_output = os.path.join(args.output_dir, config["sft_output"])
        merge_jsonl_files(sft_files, sft_output)
        print(f"SFT数据: {sft_output}")

    print("\n=== 数据准备完成 ===")
    print(f"训练命令示例:")
    print(f"  python trainer/train_pretrain.py --config minimind-3/config_{{'6' if args.scale=='small' else '7' if args.scale=='medium' else '8'}}.json \\")
    print(f"    --data_path {os.path.join(args.output_dir, config['pretrain_output'])} \\")
    print(f"    --epochs 2 --batch_size 16 --accumulation_steps 8")


if __name__ == "__main__":
    main()
