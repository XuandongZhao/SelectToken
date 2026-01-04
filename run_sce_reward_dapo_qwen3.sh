#!/usr/bin/env bash

# Script for training with self-certainty as reward instead of correctness
# Key differences from run_entropy_tokens_dapo_qwen3.sh:
# 1. Uses algorithm.use_sce_as_reward=True
# 2. Shuffles training data with seed
# 3. Maximum 50 training steps
# 4. 16 samples per question (rollout.n=16)

set -xeuo pipefail

# Load environment variables
if [ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/.env"
fi
export WANDB_API_KEY

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

source /opt/conda/etc/profile.d/conda.sh
conda activate verl

ray stop --force
pkill -9 ray || true

# Wait for processes to fully terminate and GPU memory to clear
sleep 5

ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# TOGGLE: Set to "full", "entropy", "probability", or "entropy-probability" to choose training mode
# "full" = train on all tokens
# "entropy" = train on only top 20% high-entropy tokens
# "probability" = train on tokens selected by probability distribution criteria
# "entropy-probability" = train on tokens selected by both entropy AND probability criteria
TRAINING_MODE=${TRAINING_MODE:-"probability"}  # default to full tokens

project_name='verl_select_token'
if [ "$TRAINING_MODE" = "full" ]; then
    exp_name='RLVR-SCE-reward-full-tokens-Qwen2.5-3B'
    mask_mode='entropy'
    entropy_top_ratio=1.0  # not used in full mode
elif [ "$TRAINING_MODE" = "entropy" ]; then
    exp_name='RLVR-SCE-reward-entropy-tokens-Qwen2.5-3B'
    mask_mode='entropy'
    entropy_top_ratio=0.2
elif [ "$TRAINING_MODE" = "probability" ]; then
    exp_name='RLVR-SCE-reward-probability-tokens-Qwen2.5-3B'
    mask_mode='probability'
    max_prob_threshold=0.5
elif [ "$TRAINING_MODE" = "entropy-probability" ]; then
    exp_name='RLVR-SCE-reward-entropy-prob-tokens-Qwen2.5-3B'
    mask_mode='entropy-probability'
    entropy_top_ratio=0.2
else
    echo "Invalid training mode: $TRAINING_MODE"
    exit 1
fi

# GRPO is required for use_sce_as_reward
adv_estimator=grpo
norm_adv_by_std_in_grpo=True

# Self-certainty as reward instead of correctness
use_sce_as_reward=True

# Data shuffling seed
data_seed=42

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 6))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean" # select from 'token-mean', 'seq-mean-token-sum', 'seq-mean-token-mean'

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=32
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=16  # 16 samples per question
train_prompt_mini_bsz=32

# Use total_epochs=1 (must be integer for range() loop) with explicit total_training_steps
# Setting total_training_steps=50 will override the epochs calculation
total_epochs=1

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/home/ubuntu/mnt/Beyond-the-80-20-Rule-RLVR"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B"}
CKPTS_DIR=${CKPTS_DIR:-"${PROJECT_ROOT}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math__combined_54.4k.parquet"}
VAL_AIME_FILE=${VAL_AIME_FILE:-"${RAY_DATA_HOME}/data/math__aime_repeated_32x_960.parquet"}
VAL_MATH500_FILE=${VAL_MATH500_FILE:-"${RAY_DATA_HOME}/data/math__math_500.parquet"}
VAL_FILES=${VAL_FILES:-"['${VAL_AIME_FILE}', '${VAL_MATH500_FILE}']"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=False
gen_tp=1

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.shuffle=True \
    data.seed=${data_seed} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.use_sce_as_reward=${use_sce_as_reward} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=${total_epochs} \
    +trainer.total_training_steps=50 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    $(if [ "$TRAINING_MODE" = "entropy" ]; then 
        echo "actor_rollout_ref.actor.mask_mode=entropy actor_rollout_ref.actor.entropy_top_ratio=${entropy_top_ratio}"; 
    elif [ "$TRAINING_MODE" = "probability" ]; then 
        echo "actor_rollout_ref.actor.mask_mode=probability actor_rollout_ref.actor.max_prob_threshold=${max_prob_threshold}"; 
    elif [ "$TRAINING_MODE" = "entropy-probability" ]; then 
        echo "actor_rollout_ref.actor.mask_mode=entropy-probability actor_rollout_ref.actor.entropy_top_ratio=${entropy_top_ratio}"; 
    fi)


# cd /home/ubuntu/mnt/SelectToken && conda activate verl && python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir ckpts/verl_select_token/RLVR-SCE-reward-full-tokens-Qwen2.5-3B/global_step_50/actor \
#     --target_dir ckpts/hf_models/RLVR-SCE-reward-full-tokens-Qwen2.5-3B-step50