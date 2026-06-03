#!/bin/bash
# ============================================================
#  AgentLM B+C 方案训练脚本
#
#  B: T4 — minimind-3/5/6 全流程 + minimind-5 消融实验
#  C: A10 — minimind-7 全流程 + ReAct vs Plan-Execute 对比
#
#  用法:
#    bash scripts/train_bc.sh t4          # T4 全部任务
#    bash scripts/train_bc.sh a10         # A10 全部任务
#    bash scripts/train_bc.sh t4_data     # 仅数据准备
#    bash scripts/train_bc.sh t4_m3       # 仅 minimind-3
#    bash scripts/train_bc.sh t4_m5       # 仅 minimind-5
#    bash scripts/train_bc.sh t4_m6       # 仅 minimind-6
#    bash scripts/train_bc.sh t4_ablation # 仅消融实验
#    bash scripts/train_bc.sh a10_m7      # 仅 minimind-7 主流程
#    bash scripts/train_bc.sh a10_compare # 仅 ReAct vs Plan-Execute
# ============================================================

set -e

TASK=${1:-t4}
cd trainer

# ======================== 公共函数 ========================

prepare_data() {
  echo "========== [数据准备] =========="

  # 检查 SFT 数据
  if [ ! -f "../dataset/sft_combined.jsonl" ]; then
    echo "  未找到 sft_combined.jsonl，尝试下载..."
    pip install -q addict datasets 2>/dev/null || true
    python -c "
import os, json
sft_path = '../dataset/sft_combined.jsonl'
if not os.path.exists(sft_path):
    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load('gongjy/minimind_dataset', subset_name='default', split='train')
        with open(sft_path, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'  下载完成: {sft_path}')
    except Exception as e:
        print(f'  modelscope下载失败: {e}')
        print('  尝试 HuggingFace...')
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='jingyaogong/minimind_dataset',
                repo_type='dataset',
                local_dir='../dataset',
                allow_patterns=['*.jsonl']
            )
            # 如果文件名不同，做软链接
            import glob
            sft_files = glob.glob('../dataset/sft*.jsonl')
            if sft_files and not os.path.exists(sft_path):
                os.symlink(os.path.basename(sft_files[0]), sft_path)
            print(f'  HuggingFace下载完成')
        except Exception as e2:
            print(f'  HuggingFace下载失败: {e2}')
            print('  请手动下载: https://www.modelscope.cn/datasets/gongjy/minimind_dataset')
"
  else
    echo "  sft_combined.jsonl 已存在"
  fi

  # 生成 Agent RL 数据
  if [ ! -f "../dataset/agent_rl.jsonl" ] || [ $(wc -l < ../dataset/agent_rl.jsonl 2>/dev/null || echo 0) -lt 100 ]; then
    echo "  生成 agent_rl.jsonl (10000 条)..."
    python ../dataset/generate_agent_data.py --output ../dataset/agent_rl.jsonl --num_samples 10000
  else
    echo "  agent_rl.jsonl 已存在 ($(wc -l < ../dataset/agent_rl.jsonl) 行)"
  fi

  # 生成 Agent RL Math 数据
  if [ ! -f "../dataset/agent_rl_math.jsonl" ] || [ $(wc -l < ../dataset/agent_rl_math.jsonl 2>/dev/null || echo 0) -lt 100 ]; then
    echo "  生成 agent_rl_math.jsonl (5000 条)..."
    python ../dataset/generate_agent_data.py --output ../dataset/agent_rl_math.jsonl --num_samples 5000 --multi_turn_ratio 0.6
  else
    echo "  agent_rl_math.jsonl 已存在"
  fi

  echo "  数据准备完成"
}

# ======================== T4 阶段 ========================

# --- minimind-3 (64M, hidden=768): Pretrain → SFT + Agent RL ---
train_m3() {
  echo ""
  echo "========== [T4] minimind-3 (64M) =========="
  local CONFIG="../minimind-3/config.json"

  # Pretrain
  echo "  [1/3] Pretrain..."
  python train_pretrain.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --accumulation_steps 4 \
    --save_weight "pretrain" \
    --num_workers 4

  # SFT (20% 工具混合)
  echo "  [2/3] SFT (20% tool_call_ratio)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --accumulation_steps 4 \
    --tool_call_ratio 0.2 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "full_sft" \
    --num_workers 4

  # Agent RL
  echo "  [3/3] Agent RL (ReAct)..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 3e-7 \
    --accumulation_steps 8 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "agent" \
    --num_workers 2

  echo "  minimind-3 完成"
}

# --- minimind-5 (186M, hidden=1024): Pretrain → SFT + Agent RL ---
train_m5() {
  echo ""
  echo "========== [T4] minimind-5 (186M) =========="
  local CONFIG="../minimind-3/config_5.json"

  # Pretrain
  echo "  [1/3] Pretrain..."
  python train_pretrain.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --accumulation_steps 4 \
    --save_weight "pretrain" \
    --num_workers 4

  # SFT (20% 工具混合)
  echo "  [2/3] SFT (20% tool_call_ratio)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --accumulation_steps 8 \
    --tool_call_ratio 0.2 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "full_sft" \
    --num_workers 4

  # Agent RL
  echo "  [3/3] Agent RL (ReAct)..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "agent" \
    --num_workers 2

  echo "  minimind-5 完成"
}

# --- minimind-6 (500M, hidden=1280): Pretrain → SFT + Agent RL ---
train_m6() {
  echo ""
  echo "========== [T4] minimind-6 (500M) =========="
  local CONFIG="../minimind-3/config_6.json"

  # Pretrain
  echo "  [1/3] Pretrain..."
  python train_pretrain.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --accumulation_steps 8 \
    --save_weight "pretrain" \
    --num_workers 4

  # SFT (20% 工具混合)
  echo "  [2/3] SFT (20% tool_call_ratio)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --accumulation_steps 16 \
    --tool_call_ratio 0.2 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "full_sft" \
    --num_workers 4

  # Agent RL
  echo "  [3/3] Agent RL (ReAct)..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "agent" \
    --num_workers 2

  echo "  minimind-6 完成"
}

# --- minimind-5 消融实验 (3 组) ---
train_ablation() {
  echo ""
  echo "========== [T4] minimind-5 消融实验 =========="
  local CONFIG="../minimind-3/config_5.json"

  # ---- 消融 A: tool_call_ratio 对比 ----
  echo ""
  echo "  --- 消融 A: tool_call_ratio = 0.0 (无工具混合) ---"
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --accumulation_steps 8 \
    --tool_call_ratio 0.0 \
    --from_weight "pretrain" \
    --save_weight "ablation_sft_ratio0" \
    --num_workers 4

  echo ""
  echo "  --- 消融 A: tool_call_ratio = 0.1 ---"
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --accumulation_steps 8 \
    --tool_call_ratio 0.1 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "ablation_sft_ratio01" \
    --num_workers 4

  echo ""
  echo "  --- 消融 A: tool_call_ratio = 0.3 ---"
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --accumulation_steps 8 \
    --tool_call_ratio 0.3 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "ablation_sft_ratio03" \
    --num_workers 4

  # ---- 消融 B: max_turns 对比 (基于 full_sft_1024) ----
  echo ""
  echo "  --- 消融 B: Agent RL max_turns=3 ---"
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 3 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "ablation_agent_turns3" \
    --num_workers 2

  # ---- 消融 C: loss_type 对比 ----
  echo ""
  echo "  --- 消融 C: Agent RL loss_type=grpo ---"
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type grpo \
    --from_weight "full_sft" \
    --save_weight "ablation_agent_grpo" \
    --num_workers 2

  echo "  消融实验完成"
}

# ======================== A10 阶段 ========================

# --- minimind-7 (1.5B, hidden=2048): Pretrain → SFT + Agent RL ---
train_m7() {
  echo ""
  echo "========== [A10] minimind-7 (1.5B) =========="
  local CONFIG="../minimind-3/config_7.json"

  # Pretrain
  echo "  [1/3] Pretrain..."
  python train_pretrain.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --accumulation_steps 16 \
    --save_weight "pretrain" \
    --num_workers 4

  # SFT (20% 工具混合)
  echo "  [2/3] SFT (20% tool_call_ratio)..."
  python train_full_sft.py \
    --config "$CONFIG" \
    --epochs 2 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --accumulation_steps 16 \
    --tool_call_ratio 0.2 \
    --tool_call_data_path ../dataset/agent_rl.jsonl \
    --from_weight "pretrain" \
    --save_weight "full_sft" \
    --num_workers 4

  # Agent RL (ReAct)
  echo "  [3/3] Agent RL (ReAct)..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 5 \
    --system_prompt_type react \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "agent" \
    --num_workers 2

  echo "  minimind-7 主流程完成"
}

# --- minimind-7 ReAct vs Plan-Execute 对比 ---
train_compare() {
  echo ""
  echo "========== [A10] minimind-7 ReAct vs Plan-Execute 对比 =========="
  local CONFIG="../minimind-3/config_7.json"

  # ReAct Agent
  if [ ! -f "../out/agent_2048.pth" ]; then
    echo "  [1/2] Agent RL (ReAct)..."
    python train_agent.py \
      --config "$CONFIG" \
      --epochs 1 \
      --batch_size 1 \
      --learning_rate 3e-7 \
      --accumulation_steps 16 \
      --max_turns 5 \
      --system_prompt_type react \
      --loss_type cispo \
      --from_weight "full_sft" \
      --save_weight "agent_react" \
      --num_workers 2
  else
    echo "  [1/2] ReAct 已训练 (agent_2048.pth)，跳过"
  fi

  # Plan-Execute Agent
  echo "  [2/2] Agent RL (Plan-Execute)..."
  python train_agent.py \
    --config "$CONFIG" \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 3e-7 \
    --accumulation_steps 16 \
    --max_turns 5 \
    --system_prompt_type plan_execute \
    --loss_type cispo \
    --from_weight "full_sft" \
    --save_weight "agent_plan_execute" \
    --num_workers 2

  echo "  ReAct vs Plan-Execute 对比完成"
}

# ======================== 评估 ========================

evaluate() {
  echo ""
  echo "========== [评估] =========="
  cd ..

  for WEIGHT in pretrain full_sft agent \
               ablation_sft_ratio0 ablation_sft_ratio01 ablation_sft_ratio03 \
               ablation_agent_turns3 ablation_agent_grpo \
               agent_react agent_plan_execute; do
    for SIZE in 768 1024 1280 2048; do
      if [ -f "out/${WEIGHT}_${SIZE}.pth" ]; then
        echo "  评估: ${WEIGHT}_${SIZE}"
        python scripts/eval_toolcall.py --weight "$WEIGHT" --hidden_size "$SIZE" --benchmark 1 2>/dev/null || true
      fi
    done
  done

  # 汇总结果
  echo "  汇总评估结果..."
  python scripts/collect_results.py --output results_summary.md 2>/dev/null || true

  cd trainer
  echo "  评估完成"
}

# ======================== 任务调度 ========================

case $TASK in
  t4_data)
    prepare_data
    ;;
  t4_m3)
    prepare_data; train_m3
    ;;
  t4_m5)
    prepare_data; train_m5
    ;;
  t4_m6)
    prepare_data; train_m6
    ;;
  t4_ablation)
    train_ablation
    ;;
  t4)
    echo "=========================================="
    echo "  T4 阶段 — B 方案"
    echo "  预计机时: ~44 机时"
    echo "=========================================="
    prepare_data
    train_m3
    train_m5
    train_m6
    train_ablation
    ;;
  a10_m7)
    prepare_data; train_m7
    ;;
  a10_compare)
    train_compare
    ;;
  a10)
    echo "=========================================="
    echo "  A10 阶段 — C 方案"
    echo "  预计机时: ~73 机时"
    echo "=========================================="
    prepare_data
    train_m7
    train_compare
    ;;
  eval)
    evaluate
    ;;
  all)
    echo "=========================================="
    echo "  B+C 完整方案"
    echo "  T4: ~44 机时 | A10: ~73 机时"
    echo "=========================================="
    prepare_data
    train_m3
    train_m5
    train_m6
    train_ablation
    train_m7
    train_compare
    evaluate
    ;;
  *)
    echo "用法: bash scripts/train_bc.sh [任务]"
    echo ""
    echo "T4 任务 (B方案):"
    echo "  t4_data      — 仅数据准备"
    echo "  t4_m3        — minimind-3 Pretrain+SFT+Agent RL"
    echo "  t4_m5        — minimind-5 Pretrain+SFT+Agent RL"
    echo "  t4_m6        — minimind-6 Pretrain+SFT+Agent RL"
    echo "  t4_ablation  — minimind-5 消融实验 (3组)"
    echo "  t4           — T4 全部任务"
    echo ""
    echo "A10 任务 (C方案):"
    echo "  a10_m7       — minimind-7 Pretrain+SFT+Agent RL"
    echo "  a10_compare  — minimind-7 ReAct vs Plan-Execute"
    echo "  a10          — A10 全部任务"
    echo ""
    echo "其他:"
    echo "  eval         — 评估所有已训练模型"
    echo "  all          — B+C 完整方案"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "  当前任务完成!"
echo "  模型保存在: ../out/"
echo "=========================================="
