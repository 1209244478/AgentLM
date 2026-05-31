"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM
from model.model_omni import MiniMindOmni

def log_model_params(model, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = ['audio_encoder', 'vision_encoder']
    def should_count(n): return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    Logger(f'Model Params: {total:.2f}M')


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


def init_omni_model(omni_config, from_weight='full_sft', tokenizer_path='../model',
                    audio_encoder_path='../model/SenseVoiceSmall',
                    vision_model_path='../model/siglip2-base-p32-256-ve',
                    save_dir='../out', device='cuda', freeze_backbone='none', from_resume=0):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindOmni(omni_config, audio_encoder_path=audio_encoder_path, vision_model_path=vision_model_path)

    if from_weight != 'none':
        moe_suffix = '_moe' if omni_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{omni_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            param_shapes = {k: v.shape for k, v in model.named_parameters()}
            incompatible = {k for k, v in weights.items() if k in param_shapes and v.shape != param_shapes[k]}
            if incompatible:
                Logger(f'跳过shape不匹配的权重: {incompatible}')
                weights = {k: v for k, v in weights.items() if k not in incompatible}
            model.load_state_dict(weights, strict=False)
            Logger(f'已加载权重: {weight_path}')
            if from_resume == 0 and omni_config.talker_hidden_size == omni_config.hidden_size:
                n_talker = omni_config.num_talker_hidden_layers
                n_thinker = len(model.thinker.layers)
                has_talker = any(k.startswith('talker.layers.') for k in weights)
                if not has_talker and n_talker > 0:
                    for i in range(n_talker):
                        src = n_thinker - n_talker + i
                        model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())
                    Logger(f'Talker层初始化: 复制thinker layers[{n_thinker-n_talker}:{n_thinker}] → talker layers[0:{n_talker}]')

    if freeze_backbone == 'all':
        for param in model.model.parameters():
            param.requires_grad = False

    return model.to(device), tokenizer


def omni_checkpoint(omni_config, weight='pretrain_omni', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if omni_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{omni_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        clean_state_dict = {k: v for k, v in raw_model.state_dict().items()
                           if not k.startswith('audio_encoder.') and not k.startswith('vision_encoder.')}
        state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            return torch.load(resume_path, map_location='cpu')
        return None


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)


# ============================================================
# Muon Optimizer (DeepSeek V4)
# ============================================================

def newton_schulz_5(G, eps=1e-5):
    if G.dim() > 2:
        G_flat, is_reshaped = G.reshape(G.shape[0], -1), True
    elif G.dim() < 2:
        return G / (G.norm() + eps)
    else:
        G_flat, is_reshaped = G, False
    X = G_flat / (G_flat.norm() + eps)
    m, n = X.shape
    a, b, c = 3.4445, -4.7750, 2.0315
    if m > n:
        X = a * X + b * (X @ X.T) @ X + c * (X @ X.T) @ (X @ X.T) @ X
    elif m < n:
        X = a * X + b * X @ (X.T @ X) + c * X @ (X.T @ X) @ (X.T @ X)
    else:
        X = a * X + b * (X @ X.T) @ X + c * (X @ X.T) @ (X @ X.T) @ X
    for _ in range(5):
        if m > n:
            X = 1.5 * X - 0.5 * (X @ X.T) @ X
        elif m < n:
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
        else:
            X = 1.5 * X - 0.5 * (X @ X.T) @ X
    return X.reshape(G.shape) if is_reshaped else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, nesterov, wd = group['lr'], group['momentum'], group['nesterov'], group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad + wd * p.data if wd != 0 else p.grad
                state = self.state[p]
                if 'buf' not in state: state['buf'] = torch.zeros_like(grad)
                buf = state['buf'].mul_(momentum).add_(grad)
                update = newton_schulz_5(buf)
                if nesterov:
                    update = momentum * update + grad
                    update = newton_schulz_5(update) if grad.dim() >= 2 else update
                if grad.dim() >= 2:
                    m_dim = max(grad.shape[0], grad.reshape(grad.shape[0], -1).shape[1])
                    update = update * (m_dim ** 0.5)
                p.data.add_(update, alpha=-lr)
        return loss


class MixedMuonAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom, (b1, b2), wd, eps = group['lr'], group['momentum'], group['betas'], group['weight_decay'], group['eps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad + wd * p.data if wd != 0 else p.grad
                state = self.state[p]
                if grad.dim() >= 2 and min(grad.shape[0], grad.shape[-1]) >= 16:
                    if 'muon_buf' not in state: state['muon_buf'] = torch.zeros_like(grad)
                    buf = state['muon_buf'].mul_(mom).add_(grad)
                    update = newton_schulz_5(buf)
                    scale = max(grad.shape[0], grad.reshape(grad.shape[0], -1).shape[1]) ** 0.5
                    p.data.add_(update * scale, alpha=-lr)
                else:
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = 0
                    state['step'] += 1
                    state['exp_avg'].mul_(b1).add_(grad, alpha=1 - b1)
                    state['exp_avg_sq'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(1 - b2 ** state['step'])).add_(eps)
                    p.data.addcdiv_(state['exp_avg'], denom, value=-lr / (1 - b1 ** state['step']))
        return loss


def create_muon_optimizer(model, lr=1e-3, momentum=0.95, weight_decay=0.1):
    return MixedMuonAdamW(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)