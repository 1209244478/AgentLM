"""
评估模块：TTT 效果、置信度校准、诚实性评估

提供三类评估：
1. TTT 效果评估：对比 TTT 前后 perplexity 变化
2. 置信度校准评估：ECE (Expected Calibration Error)
3. 诚实性评估：在不可回答问题上的放弃率与正确率
"""
import torch
import torch.nn.functional as F
import math


def evaluate_ttt_effect(model, tokenizer, eval_texts, device="cuda", ttt_lr=1e-4, max_length=512):
    """评估 TTT 效果：对比启用/禁用 TTT 时的 perplexity

    Args:
        model: MiniMindForCausalLM
        tokenizer: 分词器
        eval_texts: 评估文本列表
        device: 设备
        ttt_lr: TTT 学习率
        max_length: 最大长度
    Returns:
        dict: {"ppl_without_ttt": float, "ppl_with_ttt": float, "ppl_improvement": float}
    """
    model.eval()

    # 不启用 TTT 的 perplexity
    total_loss_no_ttt = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss_no_ttt += outputs.loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    ppl_no_ttt = math.exp(total_loss_no_ttt / total_tokens) if total_tokens > 0 else float('inf')

    # 启用 TTT 的 perplexity（TTT 需要梯度，不能用 torch.no_grad()）
    # 每个文本独立评估：先重置权重，再前向+TTT更新，计算 loss
    model.enable_ttt(lr=ttt_lr)
    total_loss_ttt = 0.0
    total_tokens = 0
    for text in eval_texts:
        # 每个文本前重置 TTT 权重，确保公平对比
        model.reset_ttt_weights()
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, use_ttt=True)
        total_loss_ttt += outputs.loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
    ppl_with_ttt = math.exp(total_loss_ttt / total_tokens) if total_tokens > 0 else float('inf')
    model.disable_ttt()

    return {
        "ppl_without_ttt": ppl_no_ttt,
        "ppl_with_ttt": ppl_with_ttt,
        "ppl_improvement": ppl_no_ttt - ppl_with_ttt,  # 正值表示 TTT 有改善
    }


@torch.no_grad()
def evaluate_calibration(model, tokenizer, eval_texts, device="cuda", n_bins=10, max_length=512):
    """评估置信度校准：计算 ECE (Expected Calibration Error)

    ECE 越低越好。完美校准的模型 ECE = 0。

    Args:
        model: MiniMindForCausalLM
        tokenizer: 分词器
        eval_texts: 评估文本列表
        n_bins: 分箱数量
        max_length: 最大长度
    Returns:
        dict: {"ece": float, "bin_accuracies": list, "bin_confidences": list}
    """
    model.eval()
    bin_correct = [0.0] * n_bins
    bin_confidence = [0.0] * n_bins
    bin_count = [0] * n_bins

    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab]
        targets = input_ids[:, 1:]  # [1, seq_len-1]

        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = probs.max(dim=-1)  # [1, seq_len-1]

        correct = (predictions == targets).float()

        for t in range(max_probs.shape[1]):
            conf = max_probs[0, t].item()
            acc = correct[0, t].item()
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bin_correct[bin_idx] += acc
            bin_confidence[bin_idx] += conf
            bin_count[bin_idx] += 1

    # 计算 ECE
    total = sum(bin_count)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        if bin_count[i] > 0:
            avg_acc = bin_correct[i] / bin_count[i]
            avg_conf = bin_confidence[i] / bin_count[i]
            ece += (bin_count[i] / total) * abs(avg_acc - avg_conf)
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "ece": ece,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
    }


def evaluate_honesty(model, tokenizer, unanswerable_questions, answerable_questions_with_gt,
                     device="cuda", max_new_tokens=256, **generate_kwargs):
    """评估诚实性：在不可回答问题上的放弃率

    Args:
        model: MiniMindForCausalLM
        tokenizer: 分词器
        unanswerable_questions: 不可回答的问题列表
        answerable_questions_with_gt: [(question, gt_answer), ...] 可回答的问题和 GT
        device: 设备
        max_new_tokens: 最大生成 token 数
    Returns:
        dict: {"withdrawal_rate": float, "correct_rate": float, "honesty_score": float}
    """
    from trainer.honest_training import detect_honest_withdrawal

    model.eval()

    # 在不可回答问题上的放弃率（越高越好）
    withdrawal_count = 0
    for question in unanswerable_questions:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        is_withdrawal, _ = detect_honest_withdrawal(response)
        if is_withdrawal:
            withdrawal_count += 1

    withdrawal_rate = withdrawal_count / max(len(unanswerable_questions), 1)

    # 在可回答问题上的正确率
    correct_count = 0
    for question, gt in answerable_questions_with_gt:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if str(gt).lower() in response.lower():
            correct_count += 1

    correct_rate = correct_count / max(len(answerable_questions_with_gt), 1)

    # 诚实性分数：放弃率和正确率的加权平均
    # 理想模型：不可回答时放弃，可回答时正确
    honesty_score = 0.5 * withdrawal_rate + 0.5 * correct_rate

    return {
        "withdrawal_rate": withdrawal_rate,
        "correct_rate": correct_rate,
        "honesty_score": honesty_score,
    }
