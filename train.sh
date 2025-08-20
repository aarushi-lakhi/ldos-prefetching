#!/usr/bin/env bash
set -euo pipefail

# ========================
# Defaults (override via env)
# ========================
GPU="${GPU:-0}"
IP_WINDOW="${IP_WINDOW:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CTX="${CTX:-4096}"                  # context length used in dataset names
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

timestamp() { date +"%Y%m%d_%H%M%S"; }

# ======================================================
# run_all:
#   Runs the four training steps you showed, but parameterized.
#
#   Args:
#     1) DATASET_ID   (e.g., 605.mcf_s-782B_sm)
#     2) FAMILY       (e.g., mcf_sm)  <- appears in your --dataset names
#     3) MODEL_MLP            (e.g., transformer_mlp_mcf_sm_with_weight)
#     4) MODEL_JOINT          (e.g., transformer_joint_mcf_sm)
#     5) MODEL_ENCODER        (e.g., mcf_sm_encoder)
#     6) MODEL_EMBEDDER_MLP   (e.g., embedder_mlp_mcf_sm)
#
# Notes:
#  - We assume your file layout pattern stays the same:
#       data/labeled_data/<DATASET_ID>_cs_<CTX>_labeled.csv
#       data/collector_output/prefetches_<DATASET_ID>.csv
#  - We also assume your --dataset names include <FAMILY> and <CTX>, as in your examples:
#       transformer_mlp_<FAMILY>_<CTX>
#       transformer_joint_<FAMILY>_<CTX>
#       embedder_<FAMILY>_<CTX>
#       simple_mlp_<FAMILY>_<CTX>
#  - Customize below if your naming differs.
# ======================================================
run_all() {
  local DATASET_ID="$1"
  local FAMILY="$2"
  local MODEL_MLP="$3"
  local MODEL_JOINT="$4"
  local MODEL_ENCODER="$5"
  local MODEL_EMBEDDER_MLP="$6"

  local LABELED="data/labeled_data/${DATASET_ID}_cs_${CTX}_labeled.csv"
  local PREFETCH="data/collector_output/prefetches_${DATASET_ID}.csv"

  # --------- 1) train_mlp ---------
  local DS_MLP="transformer_mlp_${FAMILY}_${CTX}"
  local LOG1="${LOG_DIR}/$(timestamp)_${MODEL_MLP}_${DATASET_ID}.log"
  echo "=== [$(timestamp)] Training MLP: model='${MODEL_MLP}' dataset_id='${DATASET_ID}' --dataset='${DS_MLP}' ==="
  CUDA_VISIBLE_DEVICES="${GPU}" nohup \
    python -m jl.train.train_mlp \
      --ip_history_window "${IP_WINDOW}" \
      --batch_size "${BATCH_SIZE}" \
      --model_name "${MODEL_MLP}" \
      --cache_data_path "${LABELED}" \
      -p "${PREFETCH}" \
      --dataset "${DS_MLP}" \
      > "${LOG1}" 2>&1 &
  wait

  # --------- 2) train_joint_mlp ---------
  local DS_JOINT="transformer_joint_${FAMILY}_${CTX}"
  local LOG2="${LOG_DIR}/$(timestamp)_${MODEL_JOINT}_${DATASET_ID}.log"
  echo "=== [$(timestamp)] Training JOINT_MLP: model='${MODEL_JOINT}' dataset_id='${DATASET_ID}' --dataset='${DS_JOINT}' ==="
  CUDA_VISIBLE_DEVICES="${GPU}" nohup \
    python -m jl.train.train_joint_mlp \
      --ip_history_window "${IP_WINDOW}" \
      --batch_size "${BATCH_SIZE}" \
      --model_name "${MODEL_JOINT}" \
      --cache_data_path "${LABELED}" \
      -p "${PREFETCH}" \
      --dataset "${DS_JOINT}" \
      > "${LOG2}" 2>&1 &
  wait

  # --------- 3) train_embedders ---------
  local DS_EMB="embedder_${FAMILY}_${CTX}"
  local LOG3="${LOG_DIR}/$(timestamp)_${MODEL_ENCODER}_${DATASET_ID}.log"
  echo "=== [$(timestamp)] Training ENCODER: model='${MODEL_ENCODER}' dataset_id='${DATASET_ID}' --dataset='${DS_EMB}' ==="
  CUDA_VISIBLE_DEVICES="${GPU}" nohup \
    python -m jl.train.train_embedders \
      --ip_history_window "${IP_WINDOW}" \
      --batch_size "${BATCH_SIZE}" \
      --model_name "${MODEL_ENCODER}" \
      --cache_data_path "${LABELED}" \
      -p "${PREFETCH}" \
      --dataset "${DS_EMB}" \
      > "${LOG3}" 2>&1 &
  wait

  # --------- 4) train_mlp with encoder cache ---------
  local DS_SIMPLE="simple_mlp_${FAMILY}_${CTX}"
  local LOG4="${LOG_DIR}/$(timestamp)_${MODEL_EMBEDDER_MLP}_${DATASET_ID}.log"
  # We assume encoder_name is "${MODEL_ENCODER}_cache" based on your example.
  local ENCODER_NAME="${MODEL_ENCODER}_cache"
  echo "=== [$(timestamp)] Training EMBEDDER-MLP: model='${MODEL_EMBEDDER_MLP}' dataset_id='${DATASET_ID}' --dataset='${DS_SIMPLE}' --encoder_name='${ENCODER_NAME}' ==="
  CUDA_VISIBLE_DEVICES="${GPU}" nohup \
    python -m jl.train.train_mlp \
      --ip_history_window "${IP_WINDOW}" \
      --batch_size "${BATCH_SIZE}" \
      --model_name "${MODEL_EMBEDDER_MLP}" \
      --cache_data_path "${LABELED}" \
      -p "${PREFETCH}" \
      --dataset "${DS_SIMPLE}" \
      --encoder_name "${ENCODER_NAME}" \
      > "${LOG4}" 2>&1 &
  wait
}

# ========================
# Examples — add as many as you like
# ========================

# Example replicating your original 605.mcf_s-782B_sm run (FAMILY=mcf_sm, CTX=4096):
# You can also override CTX/GPU/BATCH_SIZE/IP_WINDOW above or inline like: CTX=8192 GPU=0 bash train_all.sh
run_all \
  "605.mcf_s-782B_sm" \
  "mcf_sm" \
  "transformer_mlp_mcf_sm_with_weight" \
  "transformer_joint_mcf_sm" \
  "mcf_sm_encoder" \
  "embedder_mlp_mcf_sm"

# Example of a different dataset + different model names:
run_all \
  "470.lbm-1274B_sm" \
  "lbm_sm" \
  "transformer_mlp_lbm_sm" \
  "transformer_joint_lbm_sm" \
  "lbm_sm_encoder" \
  "embedder_mlp_lbm_sm"

run_all \
  "401.bzip2-226B_sm" \
  "bzip2_sm" \
  "transformer_mlp_bzip2_sm" \
  "transformer_joint_bzip2_sm" \
  "bzip2_sm_encoder" \
  "embedder_mlp_bzip2_sm"

run_all \
  "437.leslie3d-232B_sm" \
  "leslie3d_sm" \
  "transformer_mlp_leslie3d_sm" \
  "transformer_joint_leslie3d_sm" \
  "leslie3d_sm_encoder" \
  "embedder_mlp_leslie3d_sm"

run_all \
  "623.xalancbmk_s-10B_sm" \
  "xalancbmk_sm" \
  "transformer_mlp_xalancbmk_sm" \
  "transformer_joint_xalancbmk_sm" \
  "xalancbmk_sm_encoder" \
  "embedder_mlp_xalancbmk_sm"

run_all \
  "620.omnetpp_s-874B_sm" \
  "omnetpp_sm" \
  "transformer_mlp_omnetpp_sm" \
  "transformer_joint_omnetpp_sm" \
  "omnetpp_sm_encoder" \
  "embedder_mlp_omnetpp_sm"

# (Add more run_all lines for more datasets/configs)
