"""
MiniMind 新架构模块测试
覆盖: Linear Attention, Parallel Attn+FFN, ALiBi, LoRA-FFN, Mamba Hybrid
"""
import sys, os, torch, math
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
        PASS += 1
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


print("=" * 70)
print("  MiniMind 新架构模块测试")
print("=" * 70)

from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM, MiniMindModel, MiniMindBlock,
    Attention, LinearAttention, ALiBiAttention, FeedForward, LoRAFeedForward,
    MambaLayer, TTTFeedForward, MTPHead, precompute_freqs_cis, apply_rotary_pos_emb
)

# ============================================================
print("\n--- [1] Linear Attention ---")
config_la = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    attention_type="linear", flash_attn=False
)
model_la = MiniMindForCausalLM(config_la)
check("Linear Attention 模型实例化", model_la is not None)

attn_layer = model_la.model.layers[0].self_attn
check("self_attn 类型为 LinearAttention", isinstance(attn_layer, LinearAttention))

x = torch.randint(0, 100, (2, 32))
out = model_la(x)
check("Linear Attention 前向: logits shape", out.logits.shape == (2, 32, 100))
check("Linear Attention 前向: 无 NaN", not torch.isnan(out.logits).any())

labels = torch.randint(0, 100, (2, 32))
labels[:, :2] = -100
out = model_la(x, labels=labels)
check("Linear Attention 带 labels: loss 有限", math.isfinite(out.loss.item()))
check("Linear Attention loss > 0", out.loss.item() > 0)

out.loss.backward()
gc = sum(1 for p in model_la.parameters() if p.requires_grad and p.grad is not None)
check("Linear Attention 梯度回流", gc > 0, f"{gc} params have grad")

# ============================================================
print("\n--- [2] Parallel Attention + FFN ---")
config_par = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    parallel_attn_ffn=True, flash_attn=False
)
model_par = MiniMindForCausalLM(config_par)
check("Parallel Attn+FFN 模型实例化", model_par is not None)

block = model_par.model.layers[0]
check("parallel_attn_ffn 标志", block.parallel_attn_ffn is True)

x = torch.randint(0, 100, (2, 32))
out = model_par(x)
check("Parallel 前向: logits shape", out.logits.shape == (2, 32, 100))
check("Parallel 前向: 无 NaN", not torch.isnan(out.logits).any())

labels = torch.randint(0, 100, (2, 32))
out = model_par(x, labels=labels)
check("Parallel 带 labels: loss 有限", math.isfinite(out.loss.item()))

out.loss.backward()
gc = sum(1 for p in model_par.parameters() if p.requires_grad and p.grad is not None)
check("Parallel 梯度回流", gc > 0, f"{gc} params have grad")

# ============================================================
print("\n--- [3] ALiBi Attention ---")
config_alibi = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    attention_type="alibi", flash_attn=False
)
model_alibi = MiniMindForCausalLM(config_alibi)
check("ALiBi 模型实例化", model_alibi is not None)

attn_layer = model_alibi.model.layers[0].self_attn
check("self_attn 类型为 ALiBiAttention", isinstance(attn_layer, ALiBiAttention))

x = torch.randint(0, 100, (2, 32))
out = model_alibi(x)
check("ALiBi 前向: logits shape", out.logits.shape == (2, 32, 100))
check("ALiBi 前向: 无 NaN", not torch.isnan(out.logits).any())

labels = torch.randint(0, 100, (2, 32))
out = model_alibi(x, labels=labels)
check("ALiBi 带 labels: loss 有限", math.isfinite(out.loss.item()))

out.loss.backward()
gc = sum(1 for p in model_alibi.parameters() if p.requires_grad and p.grad is not None)
check("ALiBi 梯度回流", gc > 0, f"{gc} params have grad")

# ALiBi 长度外推测试：训练 32 token，推理 64 token
model_alibi.eval()
with torch.no_grad():
    x_long = torch.randint(0, 100, (1, 64))
    out_long = model_alibi(x_long)
check("ALiBi 长度外推: 64 token 无 NaN", not torch.isnan(out_long.logits).any())

# ============================================================
print("\n--- [4] LoRA-FFN ---")
config_lora = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    lora_ffn=True, lora_ffn_r=4, flash_attn=False
)
model_lora = MiniMindForCausalLM(config_lora)
check("LoRA-FFN 模型实例化", model_lora is not None)

mlp_layer = model_lora.model.layers[0].mlp
check("mlp 类型为 LoRAFeedForward", isinstance(mlp_layer, LoRAFeedForward))
check("LoRA rank", mlp_layer.lora_r == 4)

x = torch.randint(0, 100, (2, 32))
out = model_lora(x)
check("LoRA-FFN 前向: logits shape", out.logits.shape == (2, 32, 100))
check("LoRA-FFN 前向: 无 NaN", not torch.isnan(out.logits).any())

labels = torch.randint(0, 100, (2, 32))
out = model_lora(x, labels=labels)
check("LoRA-FFN 带 labels: loss 有限", math.isfinite(out.loss.item()))

out.loss.backward()
lora_B_grad = mlp_layer.gate_lora_B.weight.grad is not None
check("LoRA 旁路梯度回流", lora_B_grad)

# 参数量对比
p_base = sum(p.numel() for p in MiniMindForCausalLM(MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    flash_attn=False
)).parameters())
p_lora = sum(p.numel() for p in model_lora.parameters())
check("LoRA-FFN 参数量增加", p_lora > p_base, f"base={p_base}, lora={p_lora}, +{p_lora-p_base}")

# ============================================================
print("\n--- [5] Mamba Hybrid ---")
config_mamba = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    mamba_hybrid=True, mamba_ratio=0.5, flash_attn=False
)
model_mamba = MiniMindForCausalLM(config_mamba)
check("Mamba Hybrid 模型实例化", model_mamba is not None)

mamba_count = sum(1 for l in model_mamba.model.layers if isinstance(l.self_attn, MambaLayer))
attn_count = sum(1 for l in model_mamba.model.layers if isinstance(l.self_attn, Attention))
check("Mamba 层数 (底层)", mamba_count == 2, f"{mamba_count} Mamba layers")
check("Attention 层数 (顶层)", attn_count == 2, f"{attn_count} Attention layers")

x = torch.randint(0, 100, (2, 32))
out = model_mamba(x)
check("Mamba Hybrid 前向: logits shape", out.logits.shape == (2, 32, 100))
check("Mamba Hybrid 前向: 无 NaN", not torch.isnan(out.logits).any())

labels = torch.randint(0, 100, (2, 32))
out = model_mamba(x, labels=labels)
check("Mamba Hybrid 带 labels: loss 有限", math.isfinite(out.loss.item()))

out.loss.backward()
gc = sum(1 for p in model_mamba.parameters() if p.requires_grad and p.grad is not None)
check("Mamba Hybrid 梯度回流", gc > 0, f"{gc} params have grad")

# ============================================================
print("\n--- [6] 组合测试: Linear Attention + LoRA-FFN ---")
config_combo1 = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    attention_type="linear", lora_ffn=True, lora_ffn_r=4, flash_attn=False
)
model_combo1 = MiniMindForCausalLM(config_combo1)
x = torch.randint(0, 100, (2, 32))
out = model_combo1(x, labels=torch.randint(0, 100, (2, 32)))
check("Linear+LoRA 组合: loss 有限", math.isfinite(out.loss.item()))

# ============================================================
print("\n--- [7] 组合测试: ALiBi + Parallel Attn+FFN ---")
config_combo2 = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    attention_type="alibi", parallel_attn_ffn=True, flash_attn=False
)
model_combo2 = MiniMindForCausalLM(config_combo2)
x = torch.randint(0, 100, (2, 32))
out = model_combo2(x, labels=torch.randint(0, 100, (2, 32)))
check("ALiBi+Parallel 组合: loss 有限", math.isfinite(out.loss.item()))

# ============================================================
print("\n--- [8] 组合测试: Mamba Hybrid + LoRA-FFN + TTT ---")
config_combo3 = MiniMindConfig(
    hidden_size=64, num_hidden_layers=4, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    mamba_hybrid=True, mamba_ratio=0.5,
    lora_ffn=True, lora_ffn_r=4,
    ttt_enabled=True, ttt_chunk_size=32,
    flash_attn=False
)
model_combo3 = MiniMindForCausalLM(config_combo3)
x = torch.randint(0, 100, (2, 32))
out = model_combo3(x, labels=torch.randint(0, 100, (2, 32)))
check("Mamba+LoRA+TTT 组合: loss 有限", math.isfinite(out.loss.item()))

# ============================================================
print("\n--- [9] Generate 生成兼容性 ---")
for name, model in [
    ("Linear Attention", model_la),
    ("Parallel Attn+FFN", model_par),
    ("ALiBi", model_alibi),
    ("LoRA-FFN", model_lora),
    ("Mamba Hybrid", model_mamba),
]:
    model.eval()
    try:
        with torch.no_grad():
            gen = model.generate(torch.randint(0, 100, (1, 4)), max_new_tokens=8, do_sample=False)
        check(f"{name} Generate", gen.shape[1] <= 12, f"shape={gen.shape}")
    except Exception as e:
        check(f"{name} Generate", False, str(e)[:80])

# ============================================================
print("\n--- [10] MSA (MiniMax Sparse Attention) ---")
from model.model_advanced import MiniMaxSparseAttention

cfg_msa = MiniMindConfig(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=4, head_dim=64, msa_enabled=True,
    msa_block_size=32, msa_topk_ratio=0.25, msa_fallback_len=64,
)
model_msa = MiniMindForCausalLM(cfg_msa)
model_msa.eval()

msa_params = sum(p.numel() for p in model_msa.parameters())
check("MSA 模型构建", msa_params > 0, f"params={msa_params / 1e6:.2f}M")

try:
    with torch.no_grad():
        out = model_msa(torch.randint(0, cfg_msa.vocab_size, (1, 32)))
    check("MSA 短序列前向", out.logits.shape == (1, 32, cfg_msa.vocab_size), f"shape={out.logits.shape}")
    check("MSA 短序列无NaN", not torch.isnan(out.logits).any().item())
except Exception as e:
    check("MSA 短序列前向", False, str(e)[:80])

try:
    with torch.no_grad():
        out = model_msa(torch.randint(0, cfg_msa.vocab_size, (1, 128)))
    check("MSA 长序列前向(稀疏路径)", out.logits.shape == (1, 128, cfg_msa.vocab_size), f"shape={out.logits.shape}")
    check("MSA 长序列无NaN", not torch.isnan(out.logits).any().item())
except Exception as e:
    check("MSA 长序列前向", False, str(e)[:80])

try:
    with torch.no_grad():
        gen = model_msa.generate(torch.randint(0, cfg_msa.vocab_size, (1, 4)), max_new_tokens=4, do_sample=False, use_cache=False)
    check("MSA Generate (no cache)", gen.shape[1] <= 8, f"shape={gen.shape}")
except Exception as e:
    check("MSA Generate (no cache)", False, str(e)[:80])

try:
    with torch.no_grad():
        gen = model_msa.generate(torch.randint(0, cfg_msa.vocab_size, (1, 4)), max_new_tokens=4, do_sample=False, use_cache=True)
    check("MSA Generate (KVCache)", gen.shape[1] <= 8, f"shape={gen.shape}")
except Exception as e:
    check("MSA Generate (KVCache)", False, str(e)[:80])

try:
    x = torch.randint(0, cfg_msa.vocab_size, (1, 32))
    labels = torch.randint(0, cfg_msa.vocab_size, (1, 32))
    out = model_msa(x, labels=labels)
    out.loss.backward()
    n_grad = sum(1 for p in model_msa.parameters() if p.grad is not None)
    check("MSA 梯度回传", n_grad > 0, f"{n_grad} params have grad")
    check("MSA Loss有限", torch.isfinite(out.loss).item())
except Exception as e:
    check("MSA 梯度回传", False, str(e)[:80])

try:
    msa_layer = MiniMaxSparseAttention(cfg_msa)
    xq = torch.randn(1, 128, 8, 64)
    xk = torch.randn(1, 128, 4, 64)
    topk_indices = msa_layer._index_branch(xq, xk)
    n_blocks = (128 + 32 - 1) // 32
    check("MSA Index Branch", topk_indices.shape == (1, 128, 4, max(1, int(n_blocks * 0.25))),
          f"topk_indices={topk_indices.shape}")
    check("MSA TopK范围", (topk_indices >= 0).all().item() and (topk_indices < n_blocks).all().item(),
          f"max={topk_indices.max().item()}, n_blocks={n_blocks}")
except Exception as e:
    check("MSA Index Branch", False, str(e)[:80])

# ============================================================
print()
print("=" * 70)
print(f"  结果: {PASS}/{PASS + FAIL} 通过")
print(f"  {'全部通过!' if FAIL == 0 else f'{FAIL} 个失败项'}")
print("=" * 70)
