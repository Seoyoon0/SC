#!/bin/bash
# Online PPO 학습 (cls_head jointly 학습 포함)
#
# 사용법:
#   bash scripts/run_ppo_online.sh
#
# GPU 배치 (CUDA_VISIBLE_DEVICES=2,3,6,7 기준 논리 인덱스):
#   논리 GPU 0 (물리 2) → RolloutWorker 0
#   논리 GPU 1 (물리 3) → RolloutWorker 1
#   논리 GPU 2 (물리 6) → RolloutWorker 2
#   논리 GPU 3 (물리 7) → Trainer (policy + ref + critic + cls_head)

set -e

PYTHON=/home/seoyoon/miniconda3/envs/NRL/bin/python3

# ---- 모델 / 데이터 ----
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET="datasets/DeepMath-103K_converted.parquet"
OUTPUT_DIR="models/ppo_online_v2"
CLASSIFIER_HEAD="checkpoints/action_cls/best_model/classifier_head.pt"

# ---- Rollout 설정 ----
N_ROLLOUT_WORKERS=3
PROBLEMS_PER_ROLLOUT=10
N_ROLLOUTS=8
MAX_STEPS=10
MAX_NEW_TOKENS=512
TEMPERATURE=0.8

# ---- PPO 학습 설정 ----
NUM_ITERATIONS=2
PPO_EPOCHS=1
BATCH_SIZE=8
GRAD_ACCUM=64
LR=1e-6
CRITIC_LR=1e-5
CLS_LR=1e-4
SAVE_EVERY=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

# 기존 rollout 데이터를 새 output_dir로 복사
mkdir -p "$OUTPUT_DIR/rollouts"
cp models/ppo_online/rollouts/online_ppo_DeepMath-103K_converted_worker*.jsonl "$OUTPUT_DIR/rollouts/" 2>/dev/null || true

echo "===== Online PPO 학습 시작 (cls_head jointly 학습) ====="
CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON scripts/ppo_online_trainer.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --n_rollout_workers $N_ROLLOUT_WORKERS \
    --problems_per_rollout $PROBLEMS_PER_ROLLOUT \
    --n_rollouts $N_ROLLOUTS \
    --max_steps $MAX_STEPS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --num_iterations $NUM_ITERATIONS \
    --ppo_epochs $PPO_EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --critic_lr $CRITIC_LR \
    --cls_lr $CLS_LR \
    --clip_eps 0.2 \
    --kl_coef 0.01 \
    --vf_coef 0.1 \
    --entropy_coef 0.01 \
    --gamma 0.99 \
    --lam 0.95 \
    --save_every $SAVE_EVERY \
    --classifier_head_path "$CLASSIFIER_HEAD" \
    --use_cached_rollout \
    --use_wandb \
    --wandb_project "ppo_sc_math" \
    --log_file "$OUTPUT_DIR/train.log"

echo ""
echo "Done."
