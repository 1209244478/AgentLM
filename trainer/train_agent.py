import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
from trainer.honest_training import calculate_honest_rewards
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps
from trainer.agent_tools import TOOLS, MOCK_RESULTS, CHECK_ARGS, parse_tool_calls, execute_tool, SYSTEM_PROMPTS

warnings.filterwarnings('ignore')

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义已移至 trainer/agent_tools.py ========
# TOOLS, MOCK_RESULTS, CHECK_ARGS, parse_tool_calls, execute_tool 均从 agent_tools 导入

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=5, max_new_tokens=256, thinking_ratio=0.5, device="cuda", use_ttt=False, ttt_lr=1e-4, ttt_interval=64):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    # Agent RL + TTT: 在工具调用循环中启用推理时权重更新
    # TTT 让模型在每次工具调用后根据返回结果动态调整权重
    if use_ttt and hasattr(rollout_engine, 'policy_model'):
        raw_model = rollout_engine.policy_model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module
        if hasattr(raw_model, 'enable_ttt'):
            raw_model.enable_ttt(lr=ttt_lr)

    try:
        for turn in range(max_turns):
            context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
            inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
            context_ids = inputs["input_ids"][0].tolist()
            if prompt_ids is None:
                prompt_ids = context_ids

            # TTT: 在工具调用后的生成中启用推理时训练
            do_ttt = use_ttt and turn > 0  # 第一个 turn 不做 TTT，后续 turn 根据工具返回结果更新

            # TTT: 在工具调用后，用模型对工具返回结果做一次前向传播来更新权重
            if do_ttt and hasattr(rollout_engine, 'policy_model'):
                raw_model = rollout_engine.policy_model
                if hasattr(raw_model, 'module'):
                    raw_model = raw_model.module
                if hasattr(raw_model, 'enable_ttt') and raw_model.model.unique_layers is not None:
                    ttt_layers = raw_model.model.unique_layers
                elif hasattr(raw_model, 'enable_ttt') and raw_model.model.layers is not None:
                    ttt_layers = raw_model.model.layers
                else:
                    ttt_layers = None
                if ttt_layers is not None:
                    with torch.enable_grad():
                        _ = raw_model(
                            inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            use_ttt=True
                        )

            rollout_result = rollout_engine.rollout(
                prompt_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_generations=1,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
            )
            new_ids = rollout_result.completion_ids[0].tolist()
            new_logps = rollout_result.per_token_logps[0].tolist()
            if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
            pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
            new_ids = [t for t, _ in pairs]
            new_logps = [lp for _, lp in pairs]
            new_text = rollout_result.completions[0]
            all_outputs.append(new_text)
            response_ids.extend(new_ids)
            response_mask.extend([1] * len(new_ids))
            response_old_logps.extend(new_logps)
            final_context = context + new_text
            calls = parse_tool_calls(new_text)
            if not calls:
                break
            unfinished = turn == max_turns - 1
            messages.append({"role": "assistant", "content": new_text})
            for call in calls:
                name, raw = call.get("name", ""), call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                result = execute_tool(name, raw)
                result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
                messages.append({"role": "tool", "content": result_str})

            observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
            observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            current_len = len(prompt_ids) + len(response_ids)
            obs_delta = observe_ids[current_len:]
            response_ids.extend(obs_delta)
            response_mask.extend([0] * len(obs_delta))
            response_old_logps.extend([0.0] * len(obs_delta))
            final_context = observe_context

    finally:
        # 禁用 TTT，恢复初始权重
        if use_ttt and hasattr(rollout_engine, 'policy_model'):
            raw_model = rollout_engine.policy_model
            if hasattr(raw_model, 'module'):
                raw_model = raw_model.module
            if hasattr(raw_model, 'disable_ttt'):
                raw_model.disable_ttt()

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=5, max_new_tokens=256, thinking_ratio=0.5, device="cuda", use_ttt=False, ttt_lr=1e-4, ttt_interval=64):
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device, use_ttt=use_ttt, ttt_lr=ttt_lr, ttt_interval=ttt_interval)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ================================ 工具与 Reward = End ================================
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False, metrics_log=None):
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        for msgs in messages_batch:
            if msgs and msgs[0].get('role') != 'system':
                msgs.insert(0, {'role': 'system', 'content': system_prompt})
            elif msgs and msgs[0].get('role') == 'system' and args.system_prompt_type != 'default':
                msgs[0]['content'] = system_prompt
        last_step = step

        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=args.max_turns, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device, use_ttt=bool(args.use_ttt), ttt_lr=args.ttt_lr, ttt_interval=args.ttt_interval)

        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]

        # 推断任务类型：有工具 → "tool"
        task_types = ["tool" if (t and len(t) > 0) else "general" for t in tools_batch]
        # 从 rollout per_token_logps 计算近似熵值（-mean(log_prob) 越低越确定）
        entropies_list = []
        for old_lp_batch in response_old_logps_batch:
            valid_lps = [lp for lp in old_lp_batch if lp != 0.0]
            entropies_list.append(-sum(valid_lps) / max(len(valid_lps), 1))

        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            mask = [0] * len(p) + m
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        rewards = calculate_honest_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch, task_types=task_types, entropies=entropies_list)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})
            if metrics_log is not None:
                metrics_log.append({"epoch": epoch+1, "step": step, "reward": ar, "kl": kl, "group_reward_std": gs, "advantages_std": ast, "policy_loss": pl, "avg_response_len": al, "lr": lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=4096, help="训练侧最终总长度上界")
    parser.add_argument("--max_turns", type=int, default=5, help="Agent 最大工具调用轮数")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument('--config', default=None, type=str, help="从JSON配置文件加载模型参数（优先级高于hidden_size/num_hidden_layers/use_moe）")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--system_prompt_type", type=str, default="default", choices=["default", "react", "plan_execute"], help="Agent系统提示模板类型")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    parser.add_argument("--use_ttt", default=0, type=int, choices=[0, 1], help="是否启用 Agent RL + TTT 混合方案")
    parser.add_argument("--ttt_lr", type=float, default=1e-4, help="TTT 学习率")
    parser.add_argument("--ttt_interval", type=int, default=64, help="TTT 更新间隔步数")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        lm_config = MiniMindConfig(**{k: v for k, v in config_dict.items() if k in MiniMindConfig().__dict__})
        Logger(f'Loaded config from {args.config}: hidden_size={lm_config.hidden_size}, num_hidden_layers={lm_config.num_hidden_layers}')
    else:
        lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                   max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    reward_model = None
    if args.reward_model_path and os.path.isdir(args.reward_model_path):
        try:
            reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
            Logger(f'Loaded reward model from {args.reward_model_path}')
        except Exception as e:
            Logger(f'Warning: Failed to load reward model: {e}, using rule-based rewards only')
    else:
        Logger(f'No reward model at {args.reward_model_path}, using rule-based rewards only')
    # Rollout引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt_type, SYSTEM_PROMPTS["default"])
    Logger(f"System prompt type: {args.system_prompt_type}")
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    metrics_log = []
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"), metrics_log=metrics_log)
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"), metrics_log=metrics_log)

    # 保存训练指标
    if metrics_log and is_main_process():
        moe_suffix = '_moe' if lm_config.use_moe else ''
        metrics_path = f"../metrics/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_agent.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_log, f, indent=2, ensure_ascii=False)
        Logger(f'Metrics saved to {metrics_path}')

    if dist.is_initialized(): dist.destroy_process_group()
