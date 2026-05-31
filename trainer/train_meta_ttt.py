"""
Meta-TTT 元学习训练器

核心思想：训练模型"学会在推理时更新自己"。
不是训练模型直接给出好答案，而是训练模型在 TTT 更新后能给出好答案。

训练流程（MAML 风格）：
1. 内循环（Adaptation）：对每个 batch，执行 TTT 更新（模拟推理时适应）
2. 外循环（Meta-Optimization）：用 TTT 更新后的模型在新数据上计算损失，反向传播到原始参数

这样模型的初始参数被优化为"容易通过少量梯度步骤适应新上下文"的参数。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import argparse
import warnings
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from contextlib import nullcontext

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, TTTFeedForward
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import Logger, init_model, setup_seed

warnings.filterwarnings('ignore')


def meta_ttt_step(model, input_ids, labels, attention_mask, ttt_lr, ttt_chunk_size, meta_inner_steps=1):
    """执行一步 Meta-TTT 训练（一阶 MAML 近似）

    一阶 MAML (FOMAML) 的关键：内循环更新权重后，外循环直接用更新后的权重
    计算损失并反向传播，梯度只经过外循环路径，不回传内循环的梯度计算图。
    这避免了二阶梯度的计算开销，同时仍能优化初始参数使其"容易适应"。

    正确做法：
    1. 内循环：用 adapt 数据做 TTT 更新（修改 weight.data，不建计算图）
    2. 外循环：用 eval 数据在更新后的模型上前向，loss.backward() 计算梯度
    3. 梯度自动回传到所有参数（包括 TTT 层的初始权重），因为外循环前向时
       使用的是更新后的权重值，但梯度仍通过计算图回传到原始参数

    Args:
        model: MiniMindForCausalLM
        input_ids: [batch, seq_len]
        labels: [batch, seq_len]
        attention_mask: [batch, seq_len]
        ttt_lr: TTT 内循环学习率
        ttt_chunk_size: TTT chunk 大小
        meta_inner_steps: 内循环 TTT 更新步数

    Returns:
        meta_loss: 元学习外循环损失（标量）
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    # 将序列分为两半：前半用于 TTT 适应，后半用于评估
    split_point = seq_len // 2
    adapt_ids = input_ids[:, :split_point]
    adapt_labels = labels[:, :split_point]
    adapt_mask = attention_mask[:, :split_point]
    eval_ids = input_ids[:, split_point:]
    eval_labels = labels[:, split_point:]
    eval_mask = attention_mask[:, split_point:]

    # ===== 内循环：TTT 适应 =====
    # 保存 TTT 层的初始权重
    ttt_layers = _get_ttt_layers(model)
    saved_weights = {}
    for name, layer in ttt_layers:
        saved_weights[name] = layer.down_proj.weight.data.clone()
        layer.ttt_enabled = True
        layer._W0 = saved_weights[name]

    try:
        # 在适应数据上执行 TTT 更新
        # TTT 更新直接修改 weight.data，不建立计算图
        # 临时切换到 eval 模式，因为 TTT 只在 eval 模式下生效
        model.eval()
        with torch.no_grad():
            for step in range(meta_inner_steps):
                adapt_output = model(
                    adapt_ids, attention_mask=adapt_mask,
                    use_ttt=True, labels=adapt_labels
                )
                # TTT 更新已在 TTTFeedForward.forward 中完成
        model.train()

        # ===== 外循环：在评估数据上计算损失 =====
        # 关键：这里的前向传播会建立计算图，梯度会回传到所有参数
        # 包括 TTT 层的 down_proj.weight（此时已被内循环更新）
        # FOMAML：不回传内循环的梯度，只回传外循环的梯度
        # 这等价于：θ_meta ← θ_meta - η * ∇_θ L_eval(θ + Δθ_TTT)
        # 其中 Δθ_TTT 是内循环的更新量（视为常数）
        eval_output = model(eval_ids, attention_mask=eval_mask, labels=eval_labels)
        meta_loss = eval_output.loss

        # ttt_predictor 正则：防止外循环优化将其过度适配到特定任务，破坏泛化性
        predictor_l2 = torch.tensor(0.0, device=meta_loss.device)
        for name, layer in ttt_layers:
            predictor_l2 += layer.ttt_predictor.weight.pow(2).mean()
        meta_loss = meta_loss + 1e-4 * predictor_l2

    finally:
        # 恢复 TTT 层的初始权重
        # 注意：必须在 loss.backward() 之前恢复，否则梯度会回传到已恢复的权重
        # 但我们需要梯度回传到原始参数，所以先 backward 再恢复
        pass

    # 先 backward（此时 TTT 权重仍是更新后的值，梯度回传正确）
    # 然后恢复权重
    # 注意：meta_loss.backward() 应该在调用方执行
    # 这里我们返回 meta_loss 和恢复函数，让调用方控制 backward 时机
    return meta_loss, ttt_layers, saved_weights


def restore_ttt_weights(ttt_layers, saved_weights):
    """恢复 TTT 层的初始权重（在 backward 之后调用）

    仅恢复 down_proj（被内循环 TTT 更新修改），不恢复 ttt_predictor。
    ttt_predictor 未被内循环修改，应保留优化器的梯度更新。
    """
    for name, layer in ttt_layers:
        layer.down_proj.weight.data.copy_(saved_weights[name])
        layer.ttt_enabled = False
        layer._W0 = None


def _get_ttt_layers(model):
    """获取模型中所有 TTT 层"""
    raw_model = model.module if hasattr(model, 'module') else model
    mm_model = raw_model.model
    layers = mm_model.unique_layers if mm_model.unique_layers is not None else mm_model.layers
    result = []
    for i, layer in enumerate(layers):
        if isinstance(layer.mlp, TTTFeedForward):
            result.append((f"layer_{i}", layer.mlp))
    return result


def meta_train_epoch(epoch, loader, model, optimizer, scheduler, ttt_lr, ttt_chunk_size,
                     meta_inner_steps, accumulation_steps, autocast_ctx, device, log_interval=10):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader, 1):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = (input_ids != 0).long()

        with autocast_ctx:
            meta_loss, ttt_layers, saved_weights = meta_ttt_step(
                model, input_ids, labels, attention_mask,
                ttt_lr=ttt_lr, ttt_chunk_size=ttt_chunk_size,
                meta_inner_steps=meta_inner_steps
            )
            scaled_loss = meta_loss / accumulation_steps

        # 先 backward（此时 TTT 权重仍是更新后的值，梯度正确回传）
        scaled_loss.backward()

        # backward 完成后恢复 TTT 权重
        restore_ttt_weights(ttt_layers, saved_weights)

        if step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += meta_loss.item()

        if step % log_interval == 0:
            avg_loss = total_loss / step
            lr = scheduler.get_last_lr()[0]
            Logger(f'Epoch [{epoch}] Step [{step}] Meta-Loss: {meta_loss.item():.4f} Avg: {avg_loss:.4f} LR: {lr:.2e}')

    return total_loss / max(step, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-TTT Training")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--from_weight", type=str, default="full_sft")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="meta_ttt")
    # Meta-TTT 特定参数
    parser.add_argument("--ttt_lr", type=float, default=1e-4, help="TTT 内循环学习率")
    parser.add_argument("--ttt_chunk_size", type=int, default=256, help="TTT chunk 大小")
    parser.add_argument("--meta_inner_steps", type=int, default=1, help="内循环 TTT 更新步数")
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe), ttt_enabled=True,
        ttt_lr=args.ttt_lr, ttt_chunk_size=args.ttt_chunk_size,
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'Meta-TTT Training: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params')

    # 检查 TTT 层数
    ttt_layers = _get_ttt_layers(model)
    Logger(f'TTT layers: {len(ttt_layers)}')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // args.accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate / 10)

    for epoch in range(1, args.epochs + 1):
        avg_loss = meta_train_epoch(
            epoch, loader, model, optimizer, scheduler,
            args.ttt_lr, args.ttt_chunk_size, args.meta_inner_steps,
            args.accumulation_steps, autocast_ctx, args.device, args.log_interval
        )
        Logger(f'Epoch [{epoch}/{args.epochs}] Avg Meta-Loss: {avg_loss:.4f}')

        # 保存 checkpoint
        from trainer.trainer_utils import lm_checkpoint
        lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                     optimizer=optimizer, epoch=epoch, save_dir=args.save_dir)

    Logger('Meta-TTT Training Complete!')
