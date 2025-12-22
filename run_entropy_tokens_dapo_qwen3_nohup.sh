#!/usr/bin/env bash

# Nohup wrapper for run_entropy_tokens_dapo_qwen3.sh
# Usage: 
#   ./run_entropy_tokens_dapo_qwen3_nohup.sh [training_mode]
#   training_mode: full, entropy, or probability (default: probability)

TRAINING_MODE=${1:-"probability"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/training_${TRAINING_MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with mode: ${TRAINING_MODE}"
echo "Log file: ${LOG_FILE}"

nohup env TRAINING_MODE=${TRAINING_MODE} bash "${SCRIPT_DIR}/run_entropy_tokens_dapo_qwen3.sh" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
echo "Stop with: kill ${PID}"
echo ${PID} > "${SCRIPT_DIR}/training_${TRAINING_MODE}.pid"
