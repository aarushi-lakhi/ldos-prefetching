#!/usr/bin/env bash
set -euo pipefail

# ========================
# Defaults (override via env)
# ========================
GPU="${GPU:-0}"
IP_WINDOW="${IP_WINDOW:-15}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CTX="${CTX:-4096}"                       # context length used in dataset names
LOG_DIR="${LOG_DIR:-eval_logs}"          # where per-dataset eval logs will go

mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
stamp() { date +"%Y%m%d_%H%M%S"; }

# ======================================================
# eval_all:
#   Runs the three eval steps matching your training flow.
#
#   Args:
#     1) DATASET_ID           (e.g., 605.mcf_s-782B_sm)
#     2) FAMILY               (e.g., mcf_sm)
#     3) MODEL_MLP            (e.g., transformer_mlp_mcf_sm_with_weight)
#     4) MODEL_JOINT          (e.g., transformer_joint_mcf_sm)
#     5) MODEL_ENCODER        (e.g., mcf_sm_encoder)  -> used as ${MODEL_ENCODER}_cache
#     6) MODEL_EMBEDDER_MLP   (e.g., embedder_mlp_mcf_sm)  <-- used in step 3
# ======================================================
eval_all() {
  local DATASET_ID="$1"
  local FAMILY="$2"
  local MODEL_MLP="$3"
  local MODEL_JOINT="$4"
  local MODEL_ENCODER="$5"
  local MODEL_EMBEDDER_MLP="$6"

  local LABELED="data/labeled_data/${DATASET_ID}_cs_${CTX}_labeled.csv"
  local PREFETCH="data/collector_output/prefetches_${DATASET_ID}.csv"

  local DS_MLP="transformer_mlp_${FAMILY}_${CTX}"
  local DS_JOINT="transformer_joint_${FAMILY}_${CTX}"
  local DS_SIMPLE="simple_mlp_${FAMILY}_${CTX}"
  local ENCODER_NAME="${MODEL_ENCODER}_cache"

  local DATASET_LOG="${LOG_DIR}/$(stamp)_${DATASET_ID}.eval.log"

  {
    echo "==============================================================================="
    echo "EVAL START  : $(ts)"
    echo "DATASET_ID  : ${DATASET_ID}"
    echo "FAMILY      : ${FAMILY}"
    echo "GPU         : ${GPU}"
    echo "IP_WINDOW   : ${IP_WINDOW}"
    echo "BATCH_SIZE  : ${BATCH_SIZE}"
    echo "CTX         : ${CTX}"
    echo "LOG FILE    : ${DATASET_LOG}"
    echo "==============================================================================="
    echo

    # --------- 1) eval_mlp ---------
    echo "[${DATASET_ID}] ($(ts)) STEP 1/3: eval_mlp  --model_name='${MODEL_MLP}' --dataset='${DS_MLP}'"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup \
      python -m jl.eval.eval_mlp \
        --ip_history_window "${IP_WINDOW}" \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_MLP}" \
        --cache_data_path "${LABELED}" \
        -p "${PREFETCH}" \
        --dataset "${DS_MLP}" \
        >> "${DATASET_LOG}" 2>&1 &
    wait
    rc1=$?; echo "[${DATASET_ID}] ($(ts)) STEP 1/3 DONE (rc=${rc1})"; echo

    # --------- 2) eval_joint_mlp ---------
    echo "[${DATASET_ID}] ($(ts)) STEP 2/3: eval_joint_mlp  --model_name='${MODEL_JOINT}' --dataset='${DS_JOINT}'"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup \
      python -m jl.eval.eval_joint_mlp \
        --ip_history_window "${IP_WINDOW}" \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_JOINT}" \
        --cache_data_path "${LABELED}" \
        -p "${PREFETCH}" \
        --dataset "${DS_JOINT}" \
        >> "${DATASET_LOG}" 2>&1 &
    wait
    rc2=$?; echo "[${DATASET_ID}] ($(ts)) STEP 2/3 DONE (rc=${rc2})"; echo

    # --------- 3) eval_mlp with encoder cache (uses EMBEDDER-MLP model) ---------
    echo "[${DATASET_ID}] ($(ts)) STEP 3/3: eval_mlp (encoder cache)  --model_name='${MODEL_EMBEDDER_MLP}' --dataset='${DS_SIMPLE}' --encoder_name='${ENCODER_NAME}'"
    CUDA_VISIBLE_DEVICES="${GPU}" nohup \
      python -m jl.eval.eval_mlp \
        --ip_history_window "${IP_WINDOW}" \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_EMBEDDER_MLP}" \
        --cache_data_path "${LABELED}" \
        -p "${PREFETCH}" \
        --dataset "${DS_SIMPLE}" \
        --encoder_name "${ENCODER_NAME}" \
        >> "${DATASET_LOG}" 2>&1 &
    wait
    rc3=$?; echo "[${DATASET_ID}] ($(ts)) STEP 3/3 DONE (rc=${rc3})"; echo

    # --------- Summary footer ---------
    echo "==============================================================================="
    echo "EVAL END    : $(ts)"
    echo "RET CODES   : step1=${rc1} step2=${rc2} step3=${rc3}"
    if [[ ${rc1} -eq 0 && ${rc2} -eq 0 && ${rc3} -eq 0 ]]; then
      echo "STATUS      : SUCCESS"
    else
      echo "STATUS      : FAILURE"
    fi
    echo "LOG SAVED   : ${DATASET_LOG}"
    echo "==============================================================================="
    echo
  } | tee -a "${DATASET_LOG}"
}

# ========================
# Examples — mirror your train.sh datasets
# ========================

# 605.mcf_s-782B_sm (FAMILY=mcf_sm, CTX=4096)
eval_all \
  "605.mcf_s-782B_sm" \
  "mcf_sm" \
  "transformer_mlp_mcf_sm_with_weight" \
  "transformer_joint_mcf_sm" \
  "mcf_sm_encoder" \
  "embedder_mlp_mcf_sm"

# 470.lbm-1274B_sm
eval_all \
  "470.lbm-1274B_sm" \
  "lbm_sm" \
  "transformer_mlp_lbm_sm" \
  "transformer_joint_lbm_sm" \
  "lbm_sm_encoder" \
  "embedder_mlp_lbm_sm"

# 401.bzip2-226B_sm
eval_all \
  "401.bzip2-226B_sm" \
  "bzip2_sm" \
  "transformer_mlp_bzip2_sm" \
  "transformer_joint_bzip2_sm" \
  "bzip2_sm_encoder" \
  "embedder_mlp_bzip2_sm"

# 437.leslie3d-232B_sm
eval_all \
  "437.leslie3d-232B_sm" \
  "leslie3d_sm" \
  "transformer_mlp_leslie3d_sm" \
  "transformer_joint_leslie3d_sm" \
  "leslie3d_sm_encoder" \
  "embedder_mlp_leslie3d_sm"

# 623.xalancbmk_s-10B_sm
eval_all \
  "623.xalancbmk_s-10B_sm" \
  "xalancbmk_sm" \
  "transformer_mlp_xalancbmk_sm" \
  "transformer_joint_xalancbmk_sm" \
  "xalancbmk_sm_encoder" \
  "embedder_mlp_xalancbmk_sm"

# 620.omnetpp_s-874B_sm
eval_all \
  "620.omnetpp_s-874B_sm" \
  "omnetpp_sm" \
  "transformer_mlp_omnetpp_sm" \
  "transformer_joint_omnetpp_sm" \
  "omnetpp_sm_encoder" \
  "embedder_mlp_omnetpp_sm"
