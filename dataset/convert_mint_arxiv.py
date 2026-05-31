"""
MINT-1T ArXiv -> MiniMind 格式转换（带超时重试版）
目标: 30K pretrain + 10K SFT
"""
import json, os, re, sys, random, tarfile, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "mlfoundations/MINT-1T-ArXiv"
MAX_PRETRAIN = 30000
MAX_SFT = 10000
DOWNLOAD_TIMEOUT = 180
MAX_RETRIES = 2

PRETRAIN_OUT = os.path.join(DATA_DIR, "pretrain_mint_arxiv.jsonl")
SFT_OUT = os.path.join(DATA_DIR, "sft_mint_arxiv.jsonl")


def count_lines(path):
    if os.path.exists(path):
        return sum(1 for _ in open(path, "r", encoding="utf-8"))
    return 0


def clean_latex(text):
    if text is None:
        return ""
    text = re.sub(r'\\cite\s*\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\s*\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_abstract(texts):
    for t in texts:
        if t is None:
            continue
        lower = t.lower()
        if any(kw in lower for kw in ['abstract', 'we present', 'we propose',
                                        'we introduce', 'this paper', 'in this paper']):
            return t
    return texts[0] if texts else ""


SFT_TEMPLATES = [
    ("Please summarize the main contribution of this paper.",
     "This paper presents {abstract}"),
    ("What is this academic paper about?",
     "This paper is about {abstract}"),
    ("请用中文总结一下这篇论文的核心贡献。",
     "这篇论文的核心贡献是：{abstract}"),
]


def download_with_timeout(tar_name):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(hf_hub_download, REPO, tar_name, repo_type="dataset")
        return fut.result(timeout=DOWNLOAD_TIMEOUT)


def download_tar(tar_name):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return download_with_timeout(tar_name)
        except FutureTimeout:
            if attempt < MAX_RETRIES:
                print(f"  Timeout ({DOWNLOAD_TIMEOUT}s), retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(5)
            else:
                raise TimeoutError(f"Download timeout after {MAX_RETRIES} retries")
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  Error: {e}, retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(10)
            else:
                raise


def main():
    print("Fetching tar list...")
    all_files = list_repo_files(REPO, repo_type="dataset")
    tar_files = sorted([f for f in all_files if f.endswith(".tar")])
    print(f"Found {len(tar_files)} tars")

    pretrain_count = count_lines(PRETRAIN_OUT)
    sft_count = count_lines(SFT_OUT)
    total_done = max(pretrain_count, sft_count)
    print(f"Existing: pretrain={pretrain_count} sft={sft_count}")

    PAPERS_PER_TAR = 400
    start_shard = max(0, total_done // PAPERS_PER_TAR)
    print(f"Start at shard {start_shard}")

    f_p = open(PRETRAIN_OUT, "a", encoding="utf-8")
    f_s = open(SFT_OUT, "a", encoding="utf-8")
    skipped_empty = 0
    failed = 0

    try:
        for i, tar_name in enumerate(tqdm(tar_files[start_shard:], desc="Shards", unit="shard")):
            if pretrain_count >= MAX_PRETRAIN and sft_count >= MAX_SFT:
                break

            try:
                local_path = download_tar(tar_name)
            except Exception as e:
                print(f"\n  Failed {tar_name}: {e}")
                failed += 1
                if failed > 10:
                    print("  Too many failures, stopping")
                    break
                continue

            try:
                with tarfile.open(local_path, "r:") as tar:
                    for member in tar.getmembers():
                        if pretrain_count >= MAX_PRETRAIN and sft_count >= MAX_SFT:
                            break
                        if not member.isfile() or not member.name.endswith(".json"):
                            continue
                        try:
                            content = tar.extractfile(member).read()
                            json_data = json.loads(content.decode("utf-8"))
                        except Exception:
                            continue

                        texts = json_data.get("texts", [])
                        if not texts:
                            skipped_empty += 1
                            continue

                        valid_texts = [clean_latex(t) for t in texts if t]
                        valid_texts = [t for t in valid_texts if len(t) > 20]
                        if not valid_texts:
                            skipped_empty += 1
                            continue

                        abstract = extract_abstract(valid_texts)

                        if pretrain_count < MAX_PRETRAIN:
                            combined = " ".join(valid_texts)
                            if len(combined) > 100:
                                f_p.write(json.dumps({"text": combined}, ensure_ascii=False) + "\n")
                                f_p.flush()
                                pretrain_count += 1

                        if sft_count < MAX_SFT and abstract and len(abstract) > 80:
                            user_q, asst_a = random.choice(SFT_TEMPLATES)
                            content = clean_latex(abstract)
                            if len(content) > 2000:
                                content = content[:2000] + "..."
                            sample = {
                                "conversations": [
                                    {"role": "user", "content": user_q},
                                    {"role": "assistant", "content": asst_a.format(abstract=content)}
                                ]
                            }
                            f_s.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            f_s.flush()
                            sft_count += 1
            finally:
                try:
                    os.remove(local_path)
                except OSError:
                    pass
    finally:
        f_p.close()
        f_s.close()

    print(f"\nDone! pretrain={pretrain_count} sft={sft_count} skipped={skipped_empty} failed={failed}")
    print(f"  pretrain: {os.path.getsize(PRETRAIN_OUT)/1024/1024:.1f} MB")
    print(f"  sft:     {os.path.getsize(SFT_OUT)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
