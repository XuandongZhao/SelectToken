#!/usr/bin/env bash

# Nohup wrapper for run_sce_reward_dapo_qwen3.sh
# This script runs training with self-certainty as reward in the background
# Usage: 
#   ./run_sce_reward_dapo_qwen3_nohup.sh [training_mode]
#   training_mode: full, entropy, probability, or entropy-probability (default: full)

TRAINING_MODE=${1:-"probability"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/training_sce_reward_${TRAINING_MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with self-certainty as reward"
echo "Mask mode: ${TRAINING_MODE}"
echo "Key settings:"
echo "  - algorithm.use_sce_as_reward=True"
echo "  - rollout.n=16 (16 samples per question)"
echo "  - Max steps: 50"
echo "  - Data shuffle seed: 42"
echo "Log file: ${LOG_FILE}"

nohup env TRAINING_MODE=${TRAINING_MODE} bash "${SCRIPT_DIR}/run_sce_reward_dapo_qwen3.sh" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo "Stop with: kill ${PID}"
echo ${PID} > "${SCRIPT_DIR}/training_sce_reward_${TRAINING_MODE}.pid"
