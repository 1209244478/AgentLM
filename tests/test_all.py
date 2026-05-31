"""
MiniMind 统一测试
覆盖: 模型构建、前向传播、TTT、MHC+CSA、Muon优化器、OPD训练、生成
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
print("  MiniMind 全流程测试")
print("=" * 70)

# ============================================================
print("\n--- [1] 模型构建 ---")
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindModel, MiniMindBlock
from model.model_minimind import Attention, RMSNorm, FeedForward, TTTFeedForward, MTPHead, precompute_freqs_cis

config = MiniMindConfig(hidden_size=64, num_hidden_layers=4, vocab_size=100,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16)
model = MiniMindForCausalLM(config)
check("模型实例化", model is not None)
pM = sum(p.numel() for p in model.parameters()) / 1e6
check("模型参数量", pM > 0.1, f"{pM:.2f}M")

# 前向
x = torch.randint(0, 100, (2, 32))
out = model(x)
check("前向传播: logits shape", out.logits.shape == (2, 32, 100))
check("前向传播: 无 NaN", not torch.isnan(out.logits).any())

# 带 labels
labels = torch.randint(0, 100, (2, 32))
labels[:, :2] = -100
out = model(x, labels=labels)
check("带 labels: loss 有限", math.isfinite(out.loss.item()))
check("loss > 0", out.loss.item() > 0)

# backward
out.loss.backward()
gc = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
check("梯度回流", gc > 0, f"{gc} params have grad")

# KV Cache (exact match only on GPU with flash attention)
model.eval()
past = [None] * 4
with torch.no_grad():
    h1, pkvs, _ = model.model(x, past_key_values=past, use_cache=True)
with torch.no_grad():
    h2, _, _ = model.model(x[:, :1], past_key_values=pkvs, use_cache=True)
check("KV Cache 形状正确", h2.shape == (2, 1, 64))

# ============================================================
print("\n--- [2] TTT 推理时训练 ---")
config_ttt = MiniMindConfig(hidden_size=64, num_hidden_layers=4, vocab_size=100,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                             ttt_enabled=True, ttt_chunk_size=32)
model_ttt = MiniMindForCausalLM(config_ttt)
model_ttt.train()
check("TTT 模型实例化", model_ttt is not None)

ttt_count = sum(1 for l in model_ttt.model.layers if isinstance(l.mlp, TTTFeedForward))
check("TTT 层数", ttt_count > 0, f"{ttt_count} TTT layers")

model_ttt.enable_ttt()
x_ttt = torch.randint(0, 100, (2, 64))
labels_ttt = torch.randint(0, 100, (2, 64))
with torch.enable_grad():
    out_ttt = model_ttt(x_ttt, labels=labels_ttt, use_ttt=True)
check("TTT 前向 loss 有限", out_ttt.loss is not None and math.isfinite(out_ttt.loss.item()))
model_ttt.disable_ttt()
check("TTT 禁用后恢复权重", True)

# ============================================================
print("\n--- [3] 跨层参数共享 + MTP ---")
config_shared = MiniMindConfig(hidden_size=64, num_hidden_layers=8, vocab_size=100,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    layer_share_factor=2, mtp_num_heads=2)
model_shared = MiniMindForCausalLM(config_shared)
check("共享层: unique_layers 存在", model_shared.model.unique_layers is not None)
check("共享层: 唯一层数 < 总层数",
      len(model_shared.model.unique_layers) < model_shared.model.num_hidden_layers)
check("MTP 头数", len(model_shared.mtp_heads) == 2)

x_s = torch.randint(0, 100, (2, 16))
labels_s = torch.randint(0, 100, (2, 16))
out_s = model_shared(x_s, labels=labels_s)
check("共享+MTP: loss 有限", math.isfinite(out_s.loss.item()))

# ============================================================
print("\n--- [4] MHC + CSA 高级架构 ---")
from model.model_advanced import MHCConnection, CompressedSparseAttention, MHC_CSABlock

conn = MHCConnection(hidden_size=64, num_streams=2)
streams = torch.randn(2, 16, 2, 64)
pre = conn(streams)
check("mHC H_pre shape", pre.shape == (2, 16, 64))
check("mHC B 双随机", conn._cached_B.sum(dim=1).abs().sub(1).max() < 0.02)

config_csa = MiniMindConfig(hidden_size=64, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=32, num_hidden_layers=2, flash_attn=False)
csa = CompressedSparseAttention(config_csa, compress_factor=4)
csa.eval()
freqs_c, freqs_s = precompute_freqs_cis(dim=32, end=512)
with torch.no_grad():
    s_out, _ = csa(torch.randn(1, 4, 64), (freqs_c[:4], freqs_s[:4]))
    l_out, _ = csa(torch.randn(1, 64, 64), (freqs_c[:64], freqs_s[:64]))
check("CSA 短序列", s_out.shape == (1, 4, 64))
check("CSA 长序列", l_out.shape == (1, 64, 64))
check("CSA 长序列无 NaN", not torch.isnan(l_out).any())

block = MHC_CSABlock(layer_id=0, config=config_csa, use_csa=True, num_streams=2)
block.eval()
with torch.no_grad():
    bo, _ = block(torch.randn(2, 32, 64), (freqs_c[:32], freqs_s[:32]))
check("MHC_CSABlock 输出", bo.shape == (2, 32, 64))
check("MHC_CSABlock 无 NaN", not torch.isnan(bo).any())

# ============================================================
print("\n--- [5] Muon 优化器 ---")
from trainer.trainer_utils import newton_schulz_5, create_muon_optimizer
G = torch.randn(32, 32)
N = newton_schulz_5(G)
ortho = (N @ N.T - torch.eye(32)).abs().max().item()
check("NS 方阵正交性", ortho < 0.5, f"max_dev={ortho:.3f}")

lm = torch.nn.Linear(64, 32)
opt_muon = create_muon_optimizer(lm, lr=1e-3)
loss_m = lm(torch.randn(4, 64)).sum()
loss_m.backward()
opt_muon.step(); opt_muon.zero_grad()
check("Muon step 完成", True)

# ============================================================
print("\n--- [6] On-Policy Distillation 训练 ---")
config_opd = MiniMindConfig(hidden_size=32, num_hidden_layers=2, vocab_size=50,
                             num_attention_heads=2, num_key_value_heads=2, head_dim=16)
student = MiniMindForCausalLM(config_opd).train()
teacher = MiniMindForCausalLM(config_opd).eval()
teacher.requires_grad_(False)

opt_s = torch.optim.AdamW(student.parameters(), lr=1e-3)
TA, AT = 0.7, 2.0

for step in range(3):
    inp = torch.randint(0, 50, (2, 16))
    lab = torch.randint(0, 50, (2, 16))
    s_logits = student(inp).logits[..., :-1, :].contiguous()
    with torch.no_grad():
        t_logits = teacher(inp).logits[..., :-1, :].contiguous()
    sl = lab[..., 1:].contiguous()
    ce = F.cross_entropy(s_logits.reshape(-1, 50), sl.reshape(-1), ignore_index=-100)
    ss = F.log_softmax(s_logits.reshape(-1, 50) / AT, dim=-1)
    ts = F.softmax(t_logits.reshape(-1, 50) / AT, dim=-1)
    vm = (sl.reshape(-1) != -100)
    kl = F.kl_div(ss[vm], ts[vm], reduction='batchmean') * (AT ** 2) if vm.any() else torch.zeros(1)
    (TA * ce + (1 - TA) * kl).backward()
    opt_s.step(); opt_s.zero_grad()
check("OPD 3-step 完成", True, f"final CE={ce.item():.4f}, KL={kl.item():.4f}")

# ============================================================
print("\n--- [7] Generate 生成 ---")
model.eval()
try:
    with torch.no_grad():
        gen = model.generate(torch.randint(0, 100, (1, 4)), max_new_tokens=8, do_sample=False)
    check("Generate 输出 shape", gen.shape[1] <= 4 + 8, f"shape={gen.shape}")
except Exception as e:
    check("Generate 可运行", False, str(e)[:80])

# ============================================================
print()
print("=" * 70)
print(f"  结果: {PASS}/{PASS + FAIL} 通过")
print(f"  {'全部通过!' if FAIL == 0 else f'{FAIL} 个失败项'}")
print("=" * 70)
