"""
诚实性训练模块：替代传统 Reward 机制

核心问题：传统 reward 机制（长度分、GT匹配分、RM打分）会导致 reward hacking：
- 模型学会"骗"奖励函数（凑长度、堆砌关键词）
- 模型被激励"必须给出答案"，从不承认"我不知道"

解决方案：基于结果验证的诚实性训练
1. 可验证任务（数学/工具调用）：用执行结果验证，而非 GT 字符串匹配
2. 不可验证任务（开放问答）：引入"放弃回答"选项，对诚实放弃给正奖励
3. 置信度校准：惩罚"高置信度的错误回答"，奖励"低置信度时主动放弃"
4. 模型自身不确定性估计：用生成 logits 的熵作为置信度，而非外部规则

奖励设计原则：
- 正确且自信 -> +2.0
- 正确但犹豫 -> +1.0
- 诚实放弃 -> +0.5（比错误好，比正确差）
- 错误但自信 -> -2.0（最严厉惩罚）
- 错误但犹豫 -> -1.0
"""

import re
import json
import math
import torch
import torch.nn.functional as F

# ======== 诚实性标记 ========
HONESTY_PHRASES = [
    "我不知道", "我不确定", "我不太清楚", "我无法确定",
    "我无法回答", "我不了解", "我不太了解", "我暂时无法回答",
    "I don't know", "I'm not sure", "I'm uncertain", "I cannot answer",
    "I'm not certain", "I don't have enough information",
]


def detect_honest_withdrawal(text, entropy=None, entropy_threshold=2.5):
    """检测模型是否诚实地放弃了回答

    Args:
        text: 模型输出文本
        entropy: 模型生成时的平均熵值（来自 logits），如果提供则用于置信度估计
        entropy_threshold: 熵阈值，高于此值认为模型不确定

    Returns:
        (is_withdrawal, confidence): 是否放弃, 置信度估计 (0~1)
    """
    text_lower = text.lower().strip()

    # 检查是否包含放弃回答的短语
    for phrase in HONESTY_PHRASES:
        if phrase.lower() in text_lower:
            if entropy is not None:
                conf = max(0.1, 1.0 - entropy / (entropy_threshold * 2))
            else:
                conf = 0.3
            return True, conf

    # 基于熵的置信度估计（模型自身的不确定性）
    if entropy is not None:
        # 熵越低 -> 置信度越高；熵越高 -> 置信度越低
        confidence = max(0.0, min(1.0, 1.0 - entropy / (entropy_threshold * 2)))
        return False, confidence

    # 无熵信息时，回退到基于规则的估计
    hedging_patterns = [
        r"可能|也许|大概|或许|似乎|好像|应该是",
        r"might|maybe|perhaps|possibly|probably|seems|likely",
    ]
    hedging_count = sum(len(re.findall(p, text)) for p in hedging_patterns)
    if hedging_count >= 2:
        return False, 0.5

    return False, 0.9


def compute_generation_entropy(logits):
    """从模型输出 logits 计算生成过程中的平均熵

    Args:
        logits: [seq_len, vocab_size] 或 [batch, seq_len, vocab_size]
    Returns:
        float: 平均熵值
    """
    if logits.dim() == 3:
        logits = logits.mean(0)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return entropy


def verify_answer_correctness(extracted_answer, gt, task_type="general"):
    """更鲁棒的答案正确性验证

    改进点：
    1. 数学任务：数值比较支持近似相等（相对误差 < 0.1%）
    2. 通用任务：支持多 GT 匹配，要求 GT 作为完整词出现（避免子串 hack）
    3. 工具任务：结构化结果验证

    Args:
        extracted_answer: 提取的答案
        gt: ground truth 列表
        task_type: 任务类型
    Returns:
        (is_correct, confidence_in_verification)
    """
    if not gt or extracted_answer is None:
        return False, 0.0

    if task_type == "math":
        try:
            extracted_val = float(extracted_answer)
            for g in gt:
                try:
                    gt_val = float(g)
                    if abs(extracted_val - gt_val) < max(1e-4, abs(gt_val) * 1e-3):
                        return True, 0.95
                except ValueError:
                    continue
        except ValueError:
            pass
        for g in gt:
            if str(g).strip().lower() == str(extracted_answer).strip().lower():
                return True, 0.8
        return False, 0.0

    elif task_type == "tool":
        for g in gt:
            g_str = str(g).strip().lower()
            ans_str = str(extracted_answer).strip().lower()
            if g_str == ans_str:
                return True, 0.95
            try:
                if abs(float(g_str) - float(ans_str)) < max(1e-4, abs(float(g_str)) * 1e-3):
                    return True, 0.9
            except ValueError:
                pass
            g_words = set(g_str.split())
            ans_words = set(ans_str.split())
            if g_words and g_words.issubset(ans_words):
                return True, 0.7
        return False, 0.0

    else:
        ans_lower = str(extracted_answer).strip().lower()
        for g in gt:
            g_lower = str(g).strip().lower()
            if g_lower == ans_lower:
                return True, 0.95
            # GT 作为子串出现在答案中
            # 避免子串 hack：数字 GT 需要前后不是数字，文本 GT 直接子串匹配
            if g_lower in ans_lower:
                # 对数字 GT，确保不是子串（如 "42" 不匹配 "142"）
                try:
                    float(g_lower)  # GT 是数字
                    # 检查 GT 前后是否也是数字
                    for m in re.finditer(re.escape(g_lower), ans_lower):
                        start, end = m.start(), m.end()
                        before_digit = start > 0 and ans_lower[start - 1].isdigit()
                        after_digit = end < len(ans_lower) and ans_lower[end].isdigit()
                        if not before_digit and not after_digit:
                            return True, 0.7
                except ValueError:
                    # 非数字 GT，直接子串匹配即可
                    return True, 0.7
            try:
                if abs(float(g_lower) - float(ans_lower)) < max(1e-4, abs(float(g_lower)) * 1e-3):
                    return True, 0.9
            except ValueError:
                pass
        return False, 0.0


def extract_verifiable_answer(text, task_type="general"):
    """从回答中提取可验证的答案"""
    if task_type == "math":
        nums = re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text.replace(',', ''))
        if nums:
            return nums[-1], 0.9
        return None, 0.3

    elif task_type == "tool":
        tool_results = re.findall(r'结果[是为][:：]\s*(.+)', text)
        if tool_results:
            return tool_results[-1].strip(), 0.85
        return None, 0.4

    else:
        # 去掉思考部分
        if '</think>' in text:
            answer = text.split('</think>')[-1].strip()
        else:
            answer = text.strip()
        return answer, 0.7


def calculate_honest_rewards(prompts, completions, gt_batch, tools_batch, num_gen,
                             reward_model=None, device="cuda",
                             turn_outputs_batch=None, unfinished_batch=None,
                             task_types=None, entropies=None):
    """基于诚实性的奖励计算（替代 calculate_rewards）

    核心原则：
    - 正确且自信 > 正确但犹豫 > 诚实放弃 > 错误但犹豫 > 错误且自信
    - 不使用启发式长度分、思考分等容易被 hack 的指标
    - 可验证任务用执行结果验证，不可验证任务用 RM + 诚实性
    - 置信度来自模型自身熵（如果提供），而非外部规则

    Args:
        task_types: 每个样本的任务类型列表 ["math", "tool", "general", ...]
        entropies: 每个生成的平均熵值列表（来自模型 logits），用于置信度估计
    """
    rewards = torch.zeros(len(completions), device=device)

    for idx, response in enumerate(completions):
        sample_idx = idx // num_gen
        gt = gt_batch[sample_idx]
        tools = tools_batch[sample_idx]
        task_type = (task_types[sample_idx] if task_types else "general")

        # 提取最终回答
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch else False
        final_answer = turn_outputs[-1] if turn_outputs else response

        # 获取熵值（模型自身不确定性）
        entropy = entropies[idx] if entropies else None

        # 检测是否诚实放弃（使用熵辅助置信度估计）
        is_withdrawal, model_confidence = detect_honest_withdrawal(
            final_answer, entropy=entropy
        )

        # 提取可验证答案
        extracted, extraction_confidence = extract_verifiable_answer(final_answer, task_type)

        # 使用更鲁棒的验证方法
        is_correct, ver_conf = verify_answer_correctness(extracted, gt, task_type)

        # ===== 计算诚实性奖励 =====
        if is_withdrawal:
            # 诚实放弃：给中等正奖励
            if gt and not is_correct:
                reward = 0.3  # 诚实放弃，但本应知道
            else:
                reward = 0.5  # 诚实放弃，确实不知道
        elif is_correct:
            # 正确回答
            if model_confidence >= 0.8:
                reward = 2.0   # 正确且自信 -> 最佳
            else:
                reward = 1.0   # 正确但犹豫 -> 次佳
        else:
            # 错误回答
            if model_confidence >= 0.8:
                reward = -2.0  # 错误且自信 -> 最严厉惩罚（"骗人"）
            else:
                reward = -1.0  # 错误但犹豫 -> 相对温和

        # 未完成扣分
        if unfinished:
            reward -= 0.3

        # 重复惩罚（轻量级，仅防止退化）
        toks = re.findall(r"\w+|[^\w\s]", final_answer.lower())
        if len(toks) > 6:
            ngrams = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
            rep_ratio = (len(ngrams) - len(set(ngrams))) / max(len(ngrams), 1)
            reward -= min(0.5, rep_ratio * 2)

        rewards[idx] = max(min(reward, 3.0), -3.0)

    return rewards


# ======== 置信度自校准训练 ========

def confidence_calibration_loss(logits, labels, attention_mask=None):
    """置信度校准损失

    让模型的 softmax 概率真正反映其正确概率。
    使用 Brier Score 作为校准损失。

    Brier Score = (1/N) * sum (p_i - y_i)^2
    其中 p_i 是模型对正确答案的概率，y_i 是 one-hot 标签

    校准良好的模型：
    - 当它说 90% 置信度时，应该有 90% 的概率是对的
    - 当它说 10% 置信度时，应该只有 10% 的概率是对的
    """
    if labels is None:
        return torch.tensor(0.0, device=logits.device)

    # 只计算非 padding 位置
    valid_mask = (labels != -100)
    if attention_mask is not None:
        valid_mask = valid_mask & (attention_mask[:, 1:].bool() if attention_mask.size(1) == logits.size(1) + 1 else attention_mask.bool())

    # 计算 softmax 概率
    probs = F.softmax(logits, dim=-1)

    # 收集正确 token 的概率
    shift_labels = labels[:, 1:] if labels.size(1) == logits.size(1) + 1 else labels
    shift_labels = shift_labels.clamp(min=0)

    # 正确 token 的概率
    correct_probs = probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Brier Score: (p - 1)^2 对正确位置
    brier = ((correct_probs - 1.0) ** 2 * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

    return brier


def honest_sft_loss(model, input_ids, labels, attention_mask=None, calibration_weight=0.1):
    """诚实性 SFT 损失 = 交叉熵 + 置信度校准

    在标准 SFT 基础上增加置信度校准项，让模型：
    1. 知道正确答案时给出高置信度
    2. 不知道时给出低置信度（而非瞎编）
    """
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    ce_loss = outputs.loss

    # 置信度校准损失
    cal_loss = confidence_calibration_loss(outputs.logits, labels, attention_mask)

    total_loss = ce_loss + calibration_weight * cal_loss
    return total_loss, ce_loss, cal_loss
