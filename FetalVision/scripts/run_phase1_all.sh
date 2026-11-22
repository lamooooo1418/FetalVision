#!/usr/bin/env bash
# Phase 1 (revised): train every run with its own config, then select on validation, calibrate and analyse errors.
# Usage: bash scripts/run_phase1_all.sh            (export FETALVISION_DATA first)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export FETALVISION_DATA=${FETALVISION_DATA:-./data}
mkdir -p results/phase1/logs
log(){ echo "[$(date '+%F %T')] $*" | tee -a results/phase1/logs/run_phase1_all.log; }
RUNS="cnn_scratch mnv3s_frozen mnv3s_partial_ft mnv3s_full_ft mnv3s_full_ft_lowlr mnv3s_full_ft_cw mnv3s_partial_ft_cw mnv3s_partial_ft_e25 mnv3s_partial_ft_cw_e25 mnv3s_partial_ft_e40 mnv3s_full_ft_cw_sgkf mnv3s_partial_ft_cw_sgkf"
for r in $RUNS; do
  name=$(grep '^run_name:' configs/phase1_${r}.yaml | awk '{print $2}')
  log "START $name"
  if $PY -m fetalvision.classification.train --config configs/phase1_${r}.yaml > results/phase1/logs/${name}.log 2>&1; then log "OK $name"; else log "FAIL $name"; fi
done
$PY -m fetalvision.classification.select_model --results-root results > results/phase1/logs/select_model.log 2>&1
SEL=$($PY -c "import json;print(json.load(open('results/phase1/model_selection.json'))['selected_run'])")
log "SELECTED $SEL"
for run in "$SEL" p1v2-cnn-scratch; do
  $PY -m fetalvision.classification.calibrate --run-name "$run" --results-root results > "results/phase1/logs/calibrate_${run}.log" 2>&1 || log "FAIL calibrate $run"
  $PY -m fetalvision.classification.error_analysis --run-name "$run" --results-root results > "results/phase1/logs/error_analysis_${run}.log" 2>&1 || log "FAIL error_analysis $run"
done
log "DONE"
