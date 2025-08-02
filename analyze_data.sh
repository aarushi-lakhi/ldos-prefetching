#!/usr/bin/env bash
# run_experiments.sh — continue even if a step fails
set -uo pipefail          # ← no “-e”  ➜ don’t abort on errors

log_dir="logs"
mkdir -p "$log_dir"

########################################
# Helper: run one command, report code #
########################################
run_step () {
  local label="$1"        # human-readable tag, doubles as log-file stem
  shift                   # the rest of the args are the command itself

  echo "==> $label"
  nohup "$@" >"$log_dir/${label}.out" 2>&1
  local code=$?

  if (( code != 0 )); then
    echo "‼️  $label failed with exit code $code — continuing…" \
         | tee -a "$log_dir/${label}.out"
  fi
}

########################################
# Helper: run label + analyze for one  #
########################################
run_pair () {
  local dataset="$1"                     # e.g. 605.mcf_s-782B_sm
  local analysis="${dataset}_cs_4096"

  run_step "${dataset}_label" \
           python experiments/label_data.py --dataset_name "$dataset"

  run_step "${dataset}_analyze" \
           python experiments/analyze_data.py "$analysis"
}

########################################
# Main execution sequence              #
########################################
datasets=(
  470.lbm-1274B_sm
  401.bzip2-226B_sm
  437.leslie3d-232B_sm
  623.xalancbmk_s-10B_sm
  623.xalancbmk_s-165B_sm
)

for ds in "${datasets[@]}"; do
  run_pair "$ds"
done

echo "Pipeline finished (check $log_dir/ for details). ✅"
