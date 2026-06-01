"""
MiniMind 性能基准测试
覆盖: 前向传播吞吐、生成吞吐、内存占用、KV Cache、TTT 开销、
      注意力类型对比、MoE vs Dense、参数共享、序列长度缩放、高级架构(MHC+CSA)
"""
import sys
import os
import time
import math
import json
import gc
import torch
import torch.nn.functional as F
from collections import OrderedDict
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM, MiniMindModel, MiniMindBlock,
    Attention, LinearAttention, ALiBiAttention, FeedForward, TTTFeedForward,
    MTPHead, precompute_freqs_cis, apply_rotary_pos_emb, KVCache
)
from model.model_advanced import MHCConnection, CompressedSparseAttention, MHC_CSABlock, MiniMaxSparseAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
WARMUP_ITERS = 3
BENCH_ITERS = 10
RESULTS = OrderedDict()


def get_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


@contextmanager
def cuda_memory_tracker(label=""):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        yield
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_after = torch.cuda.memory_allocated()
        yield {
            "allocated_mb": (mem_after - mem_before) / 1024 / 1024,
            "peak_mb": mem_peak / 1024 / 1024,
        }
    else:
        yield
        yield {"allocated_mb": 0, "peak_mb": 0}


def benchmark_fn(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_config(**overrides):
    defaults = dict(
        hidden_size=128, num_hidden_layers=4, vocab_size=320,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        flash_attn=False, dropout=0.0
    )
    defaults.update(overrides)
    return MiniMindConfig(**defaults)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(name, value, unit="", detail=""):
    print(f"  {name:<45s} {value:>12.4f} {unit:<8s} {detail}")


# ============================================================
print_section("MiniMind 性能基准测试")
print(f"  设备: {get_device_name()}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"  Warmup: {WARMUP_ITERS} iters, Benchmark: {BENCH_ITERS} iters")

# ============================================================
print_section("[1] 前向传播吞吐量 — 不同模型规模")

model_configs = OrderedDict([
    ("Tiny  (h=64,  L=2)",  dict(hidden_size=64,  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32)),
    ("Small (h=128, L=4)",  dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32)),
    ("Base  (h=256, L=6)",  dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2, head_dim=32)),
    ("Large (h=512, L=8)",  dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4, head_dim=64)),
])

seq_len = 64
batch_size = 2

fwd_results = OrderedDict()
for name, cfg_overrides in model_configs.items():
    config = make_config(**cfg_overrides)
    model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).eval()
    total_p, train_p = count_params(model)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=DEVICE)

    with torch.no_grad():
        t_avg = benchmark_fn(lambda: model(x))

    tokens_per_sec = (batch_size * seq_len) / t_avg
    fwd_results[name] = {
        "params_M": total_p / 1e6,
        "time_ms": t_avg * 1000,
        "tokens_per_sec": tokens_per_sec,
    }
    print_result(name, t_avg * 1000, "ms", f"params={total_p/1e6:.2f}M, throughput={tokens_per_sec:.0f} tok/s")

    del model, x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["forward_throughput"] = fwd_results

# ============================================================
print_section("[2] 生成吞吐量 — 自回归解码")

gen_config = make_config(hidden_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32)
gen_model = MiniMindForCausalLM(gen_config).to(DEVICE).to(DTYPE).eval()
prompt_len = 8
gen_tokens_target = 32

x_gen = torch.randint(0, gen_config.vocab_size, (1, prompt_len), device=DEVICE)

gen_cache_time = None
gen_cache_throughput = None
try:
    with torch.no_grad():
        t0 = time.perf_counter()
        gen_ids = gen_model.generate(inputs=x_gen, max_new_tokens=gen_tokens_target,
                                     do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_cache_time = time.perf_counter() - t0
    actual_gen_tokens = gen_ids.shape[1] - prompt_len
    gen_cache_throughput = actual_gen_tokens / gen_cache_time
    print_result("生成 (use_cache=True, KVCache)", gen_cache_time * 1000, "ms",
                 f"tokens={actual_gen_tokens}, throughput={gen_cache_throughput:.1f} tok/s")
except Exception as e:
    print_result("生成 (use_cache=True)", 0, "ms", f"[ERROR] {str(e)[:60]}")

gen_nocache_time = None
gen_nocache_throughput = None
try:
    with torch.no_grad():
        t0 = time.perf_counter()
        gen_ids_nc = gen_model.generate(inputs=x_gen, max_new_tokens=gen_tokens_target,
                                        do_sample=False, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_nocache_time = time.perf_counter() - t0
    actual_gen_tokens_nc = gen_ids_nc.shape[1] - prompt_len
    gen_nocache_throughput = actual_gen_tokens_nc / gen_nocache_time
    print_result("生成 (use_cache=False)", gen_nocache_time * 1000, "ms",
                 f"tokens={actual_gen_tokens_nc}, throughput={gen_nocache_throughput:.1f} tok/s")
except Exception as e:
    print_result("生成 (use_cache=False)", 0, "ms", f"[ERROR] {str(e)[:60]}")

if gen_cache_time is not None and gen_nocache_time is not None:
    speedup = gen_nocache_time / gen_cache_time
    print_result("KV Cache 加速比", speedup, "x", "")
else:
    speedup = None
    print_result("KV Cache 加速比", 0, "x", "[部分测试不可用]")

RESULTS["generation_throughput"] = {
    "with_cache_ms": gen_cache_time * 1000 if gen_cache_time else None,
    "with_cache_tok_s": gen_cache_throughput,
    "without_cache_ms": gen_nocache_time * 1000 if gen_nocache_time else None,
    "without_cache_tok_s": gen_nocache_throughput,
    "cache_speedup": speedup,
}

del gen_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[3] 序列长度缩放 — 前向传播时间 vs 序列长度")

scale_config = make_config(hidden_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32)
scale_model = MiniMindForCausalLM(scale_config).to(DEVICE).to(DTYPE).eval()

seq_lengths = [16, 32, 64, 128, 256, 512]
scale_results = OrderedDict()

for sl in seq_lengths:
    x_sl = torch.randint(0, scale_config.vocab_size, (1, sl), device=DEVICE)
    with torch.no_grad():
        t_avg = benchmark_fn(lambda: scale_model(x_sl), warmup=2, iters=5)
    scale_results[str(sl)] = t_avg * 1000
    print_result(f"seq_len={sl}", t_avg * 1000, "ms", f"{(sl)/t_avg:.0f} tok/s")
    del x_sl

if len(scale_results) >= 3:
    keys = list(scale_results.keys())
    ratio = scale_results[keys[-1]] / scale_results[keys[0]]
    seq_ratio = int(keys[-1]) / int(keys[0])
    print_result(f"缩放比 ({keys[-1]}/{keys[0]})", ratio, f"x (理论~{seq_ratio}x)", "")

RESULTS["seq_length_scaling"] = scale_results

del scale_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[4] 注意力类型对比 — Standard vs Linear vs ALiBi")

attn_types = OrderedDict([
    ("Standard (SDPA)", dict(attention_type="standard")),
    ("Linear (ELU+1)", dict(attention_type="linear")),
    ("ALiBi",          dict(attention_type="alibi")),
])

attn_results = OrderedDict()
for name, overrides in attn_types.items():
    config = make_config(**overrides)
    model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).eval()
    total_p, _ = count_params(model)
    x = torch.randint(0, config.vocab_size, (2, 64), device=DEVICE)

    with torch.no_grad():
        t_avg = benchmark_fn(lambda: model(x))

    attn_results[name] = {"time_ms": t_avg * 1000, "params_M": total_p / 1e6}
    print_result(name, t_avg * 1000, "ms", f"params={total_p/1e6:.2f}M")

    del model, x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["attention_type_comparison"] = attn_results

# ============================================================
print_section("[5] MoE vs Dense 对比")

moe_config = make_config(use_moe=True, num_experts=4, num_experts_per_tok=2)
dense_config = make_config(use_moe=False)

moe_model = MiniMindForCausalLM(moe_config).to(DEVICE).to(DTYPE).eval()
dense_model = MiniMindForCausalLM(dense_config).to(DEVICE).to(DTYPE).eval()

moe_params, _ = count_params(moe_model)
dense_params, _ = count_params(dense_model)

x_cmp = torch.randint(0, 320, (2, 64), device=DEVICE)

with torch.no_grad():
    t_moe = benchmark_fn(lambda: moe_model(x_cmp))
    t_dense = benchmark_fn(lambda: dense_model(x_cmp))

print_result("Dense 前向", t_dense * 1000, "ms", f"params={dense_params/1e6:.2f}M")
print_result("MoE 前向", t_moe * 1000, "ms", f"params={moe_params/1e6:.2f}M")
print_result("MoE/Dense 时间比", t_moe / t_dense, "x", "")
print_result("MoE/Dense 参数比", moe_params / dense_params, "x", "")

RESULTS["moe_vs_dense"] = {
    "dense_time_ms": t_dense * 1000,
    "dense_params_M": dense_params / 1e6,
    "moe_time_ms": t_moe * 1000,
    "moe_params_M": moe_params / 1e6,
    "time_ratio": t_moe / t_dense,
    "param_ratio": moe_params / dense_params,
}

del moe_model, dense_model, x_cmp
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[6] 跨层参数共享 — 参数压缩 vs 速度")

share_configs = OrderedDict([
    ("无共享 (factor=1)", dict(layer_share_factor=1)),
    ("2层共享 (factor=2)", dict(layer_share_factor=2)),
    ("4层共享 (factor=4)", dict(layer_share_factor=4)),
])

share_results = OrderedDict()
for name, overrides in share_configs.items():
    config = make_config(num_hidden_layers=8, **overrides)
    model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).eval()
    total_p, _ = count_params(model)
    x = torch.randint(0, 320, (2, 64), device=DEVICE)

    with torch.no_grad():
        t_avg = benchmark_fn(lambda: model(x))

    share_results[name] = {"time_ms": t_avg * 1000, "params_M": total_p / 1e6}
    print_result(name, t_avg * 1000, "ms", f"params={total_p/1e6:.2f}M")

    del model, x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["layer_sharing"] = share_results

# ============================================================
print_section("[7] TTT 推理时训练 — 开销分析")

ttt_config = make_config(ttt_enabled=True, ttt_chunk_size=32)
ttt_model = MiniMindForCausalLM(ttt_config).to(DEVICE).to(DTYPE).eval()
no_ttt_config = make_config()
no_ttt_model = MiniMindForCausalLM(no_ttt_config).to(DEVICE).to(DTYPE).eval()

x_ttt = torch.randint(0, 320, (1, 64), device=DEVICE)

with torch.no_grad():
    t_no_ttt = benchmark_fn(lambda: no_ttt_model(x_ttt), warmup=2, iters=5)

ttt_model.enable_ttt()
with torch.enable_grad():
    t_with_ttt = benchmark_fn(lambda: ttt_model(x_ttt, use_ttt=True), warmup=1, iters=3)

print_result("无 TTT 前向", t_no_ttt * 1000, "ms", "")
print_result("有 TTT 前向", t_with_ttt * 1000, "ms", "")
print_result("TTT 开销倍数", t_with_ttt / t_no_ttt, "x", "")

RESULTS["ttt_overhead"] = {
    "no_ttt_ms": t_no_ttt * 1000,
    "with_ttt_ms": t_with_ttt * 1000,
    "overhead_ratio": t_with_ttt / t_no_ttt,
}

ttt_model.disable_ttt()
del ttt_model, no_ttt_model, x_ttt
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[8] MTP 多 Token 预测 — 参数与速度")

mtp_configs = OrderedDict([
    ("MTP heads=0", dict(mtp_num_heads=0)),
    ("MTP heads=1", dict(mtp_num_heads=1)),
    ("MTP heads=2", dict(mtp_num_heads=2)),
    ("MTP heads=4", dict(mtp_num_heads=4)),
])

mtp_results = OrderedDict()
for name, overrides in mtp_configs.items():
    config = make_config(**overrides)
    model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).eval()
    total_p, _ = count_params(model)
    x = torch.randint(0, 320, (2, 64), device=DEVICE)
    labels = torch.randint(0, 320, (2, 64), device=DEVICE)

    with torch.no_grad():
        t_fwd = benchmark_fn(lambda: model(x, labels=labels), warmup=2, iters=5)

    mtp_results[name] = {"time_ms": t_fwd * 1000, "params_M": total_p / 1e6}
    print_result(name, t_fwd * 1000, "ms", f"params={total_p/1e6:.2f}M")

    del model, x, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["mtp_comparison"] = mtp_results

# ============================================================
print_section("[9] 高级架构模块 — MHC + CSA 性能")

adv_config = make_config(hidden_size=128, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=32, num_hidden_layers=2)
freqs_c, freqs_s = precompute_freqs_cis(dim=32, end=512)
pos_emb = (freqs_c[:64], freqs_s[:64])

# MHC Connection
mhc = MHCConnection(hidden_size=128, num_streams=2).to(DEVICE).to(DTYPE)
streams = torch.randn(2, 64, 2, 128, device=DEVICE, dtype=DTYPE)

t_mhc = benchmark_fn(lambda: mhc(streams))
print_result("MHC Connection (2 streams)", t_mhc * 1000, "ms", "")

# CSA Attention
csa = CompressedSparseAttention(adv_config, compress_factor=4).to(DEVICE).to(DTYPE).eval()
x_csa = torch.randn(1, 64, 128, device=DEVICE, dtype=DTYPE)

with torch.no_grad():
    t_csa = benchmark_fn(lambda: csa(x_csa, pos_emb), warmup=2, iters=5)
print_result("CSA Attention (seq=64)", t_csa * 1000, "ms", "")

# Standard Attention for comparison
std_attn = Attention(adv_config).to(DEVICE).to(DTYPE).eval()
with torch.no_grad():
    t_std = benchmark_fn(lambda: std_attn(x_csa, pos_emb), warmup=2, iters=5)
print_result("Standard Attention (seq=64)", t_std * 1000, "ms", "")
print_result("CSA/Standard 时间比", t_csa / t_std, "x", "")

# MHC_CSABlock
block = MHC_CSABlock(layer_id=0, config=adv_config, use_csa=True, num_streams=2).to(DEVICE).to(DTYPE).eval()
x_block = torch.randn(2, 64, 128, device=DEVICE, dtype=DTYPE)

with torch.no_grad():
    t_block = benchmark_fn(lambda: block(x_block, pos_emb), warmup=2, iters=5)
print_result("MHC_CSABlock (seq=64)", t_block * 1000, "ms", "")

# CSA 长序列对比
long_seq_lengths = [64, 128, 256]
csa_scale = OrderedDict()
for sl in long_seq_lengths:
    x_long = torch.randn(1, sl, 128, device=DEVICE, dtype=DTYPE)
    pos_long = (freqs_c[:sl], freqs_s[:sl])
    with torch.no_grad():
        t_csa_l = benchmark_fn(lambda: csa(x_long, pos_long), warmup=1, iters=3)
    with torch.no_grad():
        t_std_l = benchmark_fn(lambda: std_attn(x_long, pos_long), warmup=1, iters=3)
    csa_scale[str(sl)] = {
        "csa_ms": t_csa_l * 1000,
        "std_ms": t_std_l * 1000,
        "ratio": t_csa_l / t_std_l,
    }
    print_result(f"CSA seq={sl}", t_csa_l * 1000, "ms",
                 f"Std={t_std_l*1000:.2f}ms, ratio={t_csa_l/t_std_l:.2f}x")

RESULTS["advanced_modules"] = {
    "mhc_ms": t_mhc * 1000,
    "csa_ms": t_csa * 1000,
    "std_attn_ms": t_std * 1000,
    "csa_std_ratio": t_csa / t_std,
    "mhc_csa_block_ms": t_block * 1000,
    "csa_scaling": csa_scale,
}

del mhc, csa, std_attn, block, streams, x_csa, x_block
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[10] LoRA-FFN 与 Parallel Attn+FFN")

lora_config = make_config(lora_ffn=True, lora_ffn_r=4)
par_config = make_config(parallel_attn_ffn=True)
base_config = make_config()

lora_model = MiniMindForCausalLM(lora_config).to(DEVICE).to(DTYPE).eval()
par_model = MiniMindForCausalLM(par_config).to(DEVICE).to(DTYPE).eval()
base_model = MiniMindForCausalLM(base_config).to(DEVICE).to(DTYPE).eval()

x_ffn = torch.randint(0, 320, (2, 64), device=DEVICE)

with torch.no_grad():
    t_base = benchmark_fn(lambda: base_model(x_ffn), warmup=2, iters=5)
    t_lora = benchmark_fn(lambda: lora_model(x_ffn), warmup=2, iters=5)
    t_par = benchmark_fn(lambda: par_model(x_ffn), warmup=2, iters=5)

base_p, _ = count_params(base_model)
lora_p, _ = count_params(lora_model)
par_p, _ = count_params(par_model)

print_result("Base FFN", t_base * 1000, "ms", f"params={base_p/1e6:.2f}M")
print_result("LoRA-FFN (r=4)", t_lora * 1000, "ms", f"params={lora_p/1e6:.2f}M, +{(lora_p-base_p)/1e3:.1f}K")
print_result("Parallel Attn+FFN", t_par * 1000, "ms", f"params={par_p/1e6:.2f}M")

RESULTS["ffn_variants"] = {
    "base_ms": t_base * 1000,
    "base_params_M": base_p / 1e6,
    "lora_ms": t_lora * 1000,
    "lora_params_M": lora_p / 1e6,
    "lora_extra_K": (lora_p - base_p) / 1e3,
    "parallel_ms": t_par * 1000,
    "parallel_params_M": par_p / 1e6,
}

del lora_model, par_model, base_model, x_ffn
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[11] Mamba Hybrid 注意力")

mamba_config = make_config(mamba_hybrid=True, mamba_ratio=0.5)
mamba_model = MiniMindForCausalLM(mamba_config).to(DEVICE).to(DTYPE).eval()
mamba_p, _ = count_params(mamba_model)

x_mamba = torch.randint(0, 320, (2, 64), device=DEVICE)

with torch.no_grad():
    t_mamba = benchmark_fn(lambda: mamba_model(x_mamba), warmup=2, iters=5)

print_result("Mamba Hybrid (ratio=0.5)", t_mamba * 1000, "ms", f"params={mamba_p/1e6:.2f}M")

RESULTS["mamba_hybrid"] = {
    "time_ms": t_mamba * 1000,
    "params_M": mamba_p / 1e6,
}

del mamba_model, x_mamba
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[12] GPU 内存占用分析 (仅 CUDA)")

if torch.cuda.is_available():
    mem_configs = OrderedDict([
        ("Tiny  (h=64,  L=2)",  dict(hidden_size=64,  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=32)),
        ("Small (h=128, L=4)",  dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32)),
        ("Base  (h=256, L=6)",  dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2, head_dim=32)),
    ])

    mem_results = OrderedDict()
    for name, cfg_overrides in mem_configs.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        config = make_config(**cfg_overrides)
        model = MiniMindForCausalLM(config).to(DEVICE).half().eval()
        model_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        x = torch.randint(0, config.vocab_size, (1, 64), device=DEVICE)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        fwd_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

        total_p, _ = count_params(model)
        mem_results[name] = {
            "model_mb": model_mem,
            "fwd_peak_mb": fwd_peak,
            "params_M": total_p / 1e6,
        }
        print_result(name, fwd_peak, "MB", f"model={model_mem:.1f}MB, params={total_p/1e6:.2f}M")

        del model, x
        gc.collect()
        torch.cuda.empty_cache()

    RESULTS["gpu_memory"] = mem_results
else:
    print("  跳过 (无 CUDA 设备)")
    RESULTS["gpu_memory"] = "N/A - No CUDA"

# ============================================================
print_section("[13] 训练吞吐量 — 前向+反向")

train_config = make_config(hidden_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32)
train_model = MiniMindForCausalLM(train_config).to(DEVICE).to(DTYPE).train()
optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-4)

x_train = torch.randint(0, 320, (4, 64), device=DEVICE)
labels_train = torch.randint(0, 320, (4, 64), device=DEVICE)


def train_step():
    optimizer.zero_grad()
    out = train_model(x_train, labels=labels_train)
    out.loss.backward()
    optimizer.step()


t_train = benchmark_fn(train_step, warmup=2, iters=5)
train_throughput = (4 * 64) / t_train
print_result("训练步 (fwd+bwd+opt)", t_train * 1000, "ms", f"throughput={train_throughput:.0f} tok/s")

RESULTS["training_throughput"] = {
    "step_ms": t_train * 1000,
    "throughput_tok_s": train_throughput,
    "batch_size": 4,
    "seq_len": 64,
}

del train_model, optimizer, x_train, labels_train
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[14] Muon 优化器 vs AdamW 步进时间")

from trainer.trainer_utils import create_muon_optimizer

opt_config = make_config(hidden_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32)
opt_model = MiniMindForCausalLM(opt_config).to(DEVICE).to(DTYPE).train()

adamw_opt = torch.optim.AdamW(opt_model.parameters(), lr=1e-3)
muon_opt = create_muon_optimizer(opt_model, lr=1e-3)

x_opt = torch.randint(0, 320, (2, 32), device=DEVICE)
labels_opt = torch.randint(0, 320, (2, 32), device=DEVICE)


def adamw_step():
    adamw_opt.zero_grad()
    out = opt_model(x_opt, labels=labels_opt)
    out.loss.backward()
    adamw_opt.step()


def muon_step():
    muon_opt.zero_grad()
    out = opt_model(x_opt, labels=labels_opt)
    out.loss.backward()
    muon_opt.step()


t_adamw = benchmark_fn(adamw_step, warmup=2, iters=5)
t_muon = benchmark_fn(muon_step, warmup=2, iters=5)

print_result("AdamW 步进", t_adamw * 1000, "ms", "")
print_result("Muon 步进", t_muon * 1000, "ms", "")
print_result("Muon/AdamW 时间比", t_muon / t_adamw, "x", "")

RESULTS["optimizer_comparison"] = {
    "adamw_ms": t_adamw * 1000,
    "muon_ms": t_muon * 1000,
    "ratio": t_muon / t_adamw,
}

del opt_model, adamw_opt, muon_opt, x_opt, labels_opt
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[15] 长上下文专项压力测试 — MSA vs Standard vs CSA")

long_ctx_config_base = dict(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=4, head_dim=32, vocab_size=320,
    flash_attn=False, dropout=0.0,
)

long_seq_lengths = [64, 128, 256, 512, 1024]

long_ctx_results = OrderedDict()

attn_variants = OrderedDict([
    ("Standard", dict(msa_enabled=False, use_csa=False)),
    ("CSA", dict(use_csa=True, csa_compress_factor=4)),
    ("MSA (topk=0.25)", dict(msa_enabled=True, msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64)),
    ("MSA (topk=0.10)", dict(msa_enabled=True, msa_block_size=64, msa_topk_ratio=0.10, msa_fallback_len=128)),
])

for variant_name, variant_overrides in attn_variants.items():
    cfg = {**long_ctx_config_base, **variant_overrides}
    config = MiniMindConfig(**cfg)
    model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).eval()
    total_p, _ = count_params(model)

    variant_results = OrderedDict()
    for sl in long_seq_lengths:
        x = torch.randint(0, config.vocab_size, (1, sl), device=DEVICE)
        try:
            with torch.no_grad():
                t_avg = benchmark_fn(lambda: model(x), warmup=1, iters=3)
            variant_results[str(sl)] = {"time_ms": t_avg * 1000, "ok": True}
            print_result(f"{variant_name} seq={sl}", t_avg * 1000, "ms",
                         f"{sl / t_avg:.0f} tok/s, params={total_p / 1e6:.2f}M")
        except Exception as e:
            variant_results[str(sl)] = {"time_ms": None, "ok": False, "error": str(e)[:60]}
            print_result(f"{variant_name} seq={sl}", 0, "ms", f"[ERROR] {str(e)[:50]}")
        del x
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    long_ctx_results[variant_name] = {
        "params_M": total_p / 1e6,
        "scaling": variant_results,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if len(long_ctx_results) >= 2:
    std_key = "Standard"
    if std_key in long_ctx_results:
        std_scaling = long_ctx_results[std_key]["scaling"]
        for vname in long_ctx_results:
            if vname == std_key:
                continue
            v_scaling = long_ctx_results[vname]["scaling"]
            ratios = []
            for sl_key in std_scaling:
                if sl_key in v_scaling and std_scaling[sl_key]["ok"] and v_scaling[sl_key]["ok"]:
                    ratios.append(v_scaling[sl_key]["time_ms"] / std_scaling[sl_key]["time_ms"])
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                print_result(f"{vname}/Standard 平均比", avg_ratio, "x", "")

RESULTS["long_context_variants"] = long_ctx_results

# ============================================================
print_section("[16] MSA 长上下文生成压力测试 — KVCache 加速")

msa_gen_config = MiniMindConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=4, head_dim=32, vocab_size=320,
    flash_attn=False, dropout=0.0,
    msa_enabled=True, msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64,
)
msa_gen_model = MiniMindForCausalLM(msa_gen_config).to(DEVICE).to(DTYPE).eval()

std_gen_config = MiniMindConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=4, head_dim=32, vocab_size=320,
    flash_attn=False, dropout=0.0,
)
std_gen_model = MiniMindForCausalLM(std_gen_config).to(DEVICE).to(DTYPE).eval()

gen_prompt_lengths = [8, 32, 64, 128]
gen_new_tokens = 16

msa_gen_results = OrderedDict()
std_gen_results = OrderedDict()

for prompt_len in gen_prompt_lengths:
    x_gen = torch.randint(0, msa_gen_config.vocab_size, (1, prompt_len), device=DEVICE)

    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            gen_ids = msa_gen_model.generate(inputs=x_gen, max_new_tokens=gen_new_tokens,
                                             do_sample=False, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            msa_cache_time = time.perf_counter() - t0
        msa_actual = gen_ids.shape[1] - prompt_len
        msa_tps = msa_actual / msa_cache_time
        msa_gen_results[str(prompt_len)] = {"time_ms": msa_cache_time * 1000, "tok_s": msa_tps}
        print_result(f"MSA prompt={prompt_len}", msa_cache_time * 1000, "ms",
                     f"gen={msa_actual}tok, {msa_tps:.1f} tok/s")
    except Exception as e:
        msa_gen_results[str(prompt_len)] = {"time_ms": None, "error": str(e)[:60]}
        print_result(f"MSA prompt={prompt_len}", 0, "ms", f"[ERROR] {str(e)[:50]}")

    try:
        with torch.no_grad():
            t0 = time.perf_counter()
            gen_ids = std_gen_model.generate(inputs=x_gen, max_new_tokens=gen_new_tokens,
                                             do_sample=False, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            std_cache_time = time.perf_counter() - t0
        std_actual = gen_ids.shape[1] - prompt_len
        std_tps = std_actual / std_cache_time
        std_gen_results[str(prompt_len)] = {"time_ms": std_cache_time * 1000, "tok_s": std_tps}
        print_result(f"Standard prompt={prompt_len}", std_cache_time * 1000, "ms",
                     f"gen={std_actual}tok, {std_tps:.1f} tok/s")
    except Exception as e:
        std_gen_results[str(prompt_len)] = {"time_ms": None, "error": str(e)[:60]}
        print_result(f"Standard prompt={prompt_len}", 0, "ms", f"[ERROR] {str(e)[:50]}")

    del x_gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["long_context_generation"] = {
    "msa": msa_gen_results,
    "standard": std_gen_results,
}

del msa_gen_model, std_gen_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[17] MSA 稀疏度分析 — TopK 选择统计")

msa_sparse_config = MiniMindConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=4, head_dim=32, vocab_size=320,
    flash_attn=False, dropout=0.0,
    msa_enabled=True, msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64,
)
msa_sparse_model = MiniMindForCausalLM(msa_sparse_config).to(DEVICE).to(DTYPE).eval()

sparse_analysis = OrderedDict()
for sl in [128, 256, 512, 1024]:
    x = torch.randint(0, msa_sparse_config.vocab_size, (1, sl), device=DEVICE)
    try:
        with torch.no_grad():
            h = msa_sparse_model.model.embed_tokens(x)
            freqs_c, freqs_s = precompute_freqs_cis(msa_sparse_config.head_dim, sl)
            pos_emb = (freqs_c[:sl], freqs_s[:sl])

            block = msa_sparse_model.model.layers[0]
            h_norm = block.input_layernorm(h)
            attn = block.self_attn

            xq = attn.q_proj(h_norm).view(1, sl, attn.n_local_heads, attn.head_dim)
            xk = attn.k_proj(h_norm).view(1, sl, attn.n_local_kv_heads, attn.head_dim)
            xq, xk = attn.q_norm(xq), attn.k_norm(xk)
            xq, xk = apply_rotary_pos_emb(xq, xk, freqs_c[:sl], freqs_s[:sl])

            topk_indices = attn._index_branch(xq, xk)
            n_blocks = (sl + attn.block_size - 1) // attn.block_size
            actual_topk = topk_indices.shape[-1]

            unique_blocks = torch.unique(topk_indices).numel()
            total_possible = n_blocks
            sparsity = 1.0 - (actual_topk / total_possible)

            sparse_analysis[str(sl)] = {
                "n_blocks": n_blocks,
                "topk": actual_topk,
                "unique_blocks_selected": unique_blocks,
                "sparsity": sparsity,
                "kv_ratio": f"{actual_topk * attn.block_size}/{sl}",
            }
            print_result(f"seq={sl}", sparsity * 100, "% sparse",
                         f"blocks={n_blocks}, topk={actual_topk}, "
                         f"unique={unique_blocks}, KV选比={actual_topk * attn.block_size}/{sl}")
    except Exception as e:
        sparse_analysis[str(sl)] = {"error": str(e)[:60]}
        print_result(f"seq={sl}", 0, "%", f"[ERROR] {str(e)[:50]}")

    del x
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

RESULTS["msa_sparsity_analysis"] = sparse_analysis

del msa_sparse_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================
print_section("[18] 长上下文内存压力 — MSA vs Standard 峰值内存")

if torch.cuda.is_available():
    mem_msa_config = MiniMindConfig(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=4, head_dim=32, vocab_size=320,
        flash_attn=False, dropout=0.0,
        msa_enabled=True, msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64,
    )
    mem_std_config = MiniMindConfig(
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=4, head_dim=32, vocab_size=320,
        flash_attn=False, dropout=0.0,
    )

    mem_compare = OrderedDict()
    for sl in [128, 256, 512]:
        for label, cfg in [("MSA", mem_msa_config), ("Standard", mem_std_config)]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            model = MiniMindForCausalLM(cfg).to(DEVICE).half().eval()
            x = torch.randint(0, cfg.vocab_size, (1, sl), device=DEVICE)

            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    _ = model(x)
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                key = f"{label}_seq={sl}"
                mem_compare[key] = {"peak_mb": peak_mb}
                print_result(key, peak_mb, "MB", "")
            except Exception as e:
                key = f"{label}_seq={sl}"
                mem_compare[key] = {"peak_mb": None, "error": str(e)[:50]}
                print_result(key, 0, "MB", f"[ERROR] {str(e)[:40]}")

            del model, x
            gc.collect()
            torch.cuda.empty_cache()

    if f"MSA_seq=256" in mem_compare and f"Standard_seq=256" in mem_compare:
        msa_256 = mem_compare["MSA_seq=256"]["peak_mb"]
        std_256 = mem_compare["Standard_seq=256"]["peak_mb"]
        if msa_256 and std_256:
            print_result("MSA/Standard 内存比 (seq=256)", msa_256 / std_256, "x", "")

    RESULTS["long_context_memory"] = mem_compare
else:
    print("  跳过 (无 CUDA 设备)")
    RESULTS["long_context_memory"] = "N/A - No CUDA"

# ============================================================
print_section("[19] 超长序列梯度稳定性 — MSA vs Standard")

grad_test_lengths = [64, 128, 256, 512]
grad_results = OrderedDict()

for sl in grad_test_lengths:
    for label, msa_on in [("Standard", False), ("MSA", True)]:
        cfg = dict(
            hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
            num_key_value_heads=4, head_dim=32, vocab_size=320,
            flash_attn=False, dropout=0.0,
            msa_enabled=msa_on, msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64,
        )
        config = MiniMindConfig(**cfg)
        model = MiniMindForCausalLM(config).to(DEVICE).to(DTYPE).train()
        x = torch.randint(0, config.vocab_size, (1, sl), device=DEVICE)
        labels = torch.randint(0, config.vocab_size, (1, sl), device=DEVICE)

        try:
            out = model(x, labels=labels)
            loss = out.loss
            loss.backward()

            total_norm = 0.0
            nan_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
                    if torch.isnan(p.grad).any():
                        nan_count += 1
            total_norm = total_norm ** 0.5

            key = f"{label}_seq={sl}"
            grad_results[key] = {
                "loss": loss.item(),
                "grad_norm": total_norm,
                "nan_params": nan_count,
                "stable": nan_count == 0 and math.isfinite(total_norm),
            }
            print_result(key, total_norm, "grad_norm",
                         f"loss={loss.item():.4f}, nan={nan_count}, "
                         f"{'STABLE' if nan_count == 0 else 'UNSTABLE'}")
        except Exception as e:
            key = f"{label}_seq={sl}"
            grad_results[key] = {"error": str(e)[:60], "stable": False}
            print_result(key, 0, "", f"[ERROR] {str(e)[:50]}")

        del model, x, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

RESULTS["long_context_gradient_stability"] = grad_results

# ============================================================
print_section("性能测试总结")

summary = OrderedDict()
summary["device"] = get_device_name()
summary["pytorch_version"] = torch.__version__
summary["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"

if "forward_throughput" in RESULTS:
    best = max(RESULTS["forward_throughput"].items(), key=lambda x: x[1]["tokens_per_sec"])
    summary["fastest_forward"] = f"{best[0]}: {best[1]['tokens_per_sec']:.0f} tok/s"

if "generation_throughput" in RESULTS:
    gt = RESULTS["generation_throughput"]
    summary["gen_with_cache"] = f"{gt['with_cache_tok_s']:.1f} tok/s"
    summary["gen_without_cache"] = f"{gt['without_cache_tok_s']:.1f} tok/s"
    summary["kv_cache_speedup"] = f"{gt['cache_speedup']:.2f}x"

if "ttt_overhead" in RESULTS:
    summary["ttt_overhead"] = f"{RESULTS['ttt_overhead']['overhead_ratio']:.2f}x"

if "advanced_modules" in RESULTS:
    summary["csa_vs_std_attn"] = f"{RESULTS['advanced_modules']['csa_std_ratio']:.2f}x"

if "training_throughput" in RESULTS:
    summary["train_throughput"] = f"{RESULTS['training_throughput']['throughput_tok_s']:.0f} tok/s"

if "long_context_variants" in RESULTS:
    std_scaling = RESULTS["long_context_variants"].get("Standard", {}).get("scaling", {})
    msa_scaling = RESULTS["long_context_variants"].get("MSA (topk=0.25)", {}).get("scaling", {})
    if std_scaling and msa_scaling:
        max_sl = max(k for k in std_scaling if std_scaling[k].get("ok") and msa_scaling.get(k, {}).get("ok"))
        if max_sl:
            ratio = msa_scaling[max_sl]["time_ms"] / std_scaling[max_sl]["time_ms"]
            summary[f"MSA/Standard_ratio_seq={max_sl}"] = f"{ratio:.2f}x"

if "msa_sparsity_analysis" in RESULTS:
    for sl_key, info in RESULTS["msa_sparsity_analysis"].items():
        if "sparsity" in info:
            summary[f"MSA_sparsity_seq={sl_key}"] = f"{info['sparsity']*100:.1f}%"

if "long_context_gradient_stability" in RESULTS:
    stable_count = sum(1 for v in RESULTS["long_context_gradient_stability"].values() if v.get("stable", False))
    total_count = len(RESULTS["long_context_gradient_stability"])
    summary["gradient_stability"] = f"{stable_count}/{total_count} stable"

for k, v in summary.items():
    print(f"  {k:<30s} {v}")

RESULTS["summary"] = summary

# 保存结果
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  结果已保存至: {output_path}")

print(f"\n{'=' * 70}")
print("  性能基准测试完成!")
print(f"{'=' * 70}")
