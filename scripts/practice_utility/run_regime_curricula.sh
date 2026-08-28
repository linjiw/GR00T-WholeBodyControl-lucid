#!/usr/bin/env bash
# Stage 9: rerun the curricula under a chosen fine-tuning regime.
# Usage: run_regime_curricula.sh <tag> <num_envs> <iterations> <overrides...>
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
TAG=$1; NE=$2; IT=$3; shift 3
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "stage 8 done" "$LOG"; do sleep 60; done
say "stage 9 [$TAG]: off fixed lucid ta_lucid_50 x 3 seeds x $IT it, num_envs=$NE, overrides: $*"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs "$NE" --iterations "$IT" --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes off fixed lucid ta_lucid_50 --tag "$TAG" \
  --min-free-mib "$MIN_FREE" --execute --extra-overrides "$@" > "$OUT/regime_${TAG}_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/regime_${TAG}_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 9 [$TAG] training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 9 [$TAG] FAILED"; exit 1; }
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/regime_${TAG}_eval_stdout.log" 2>&1
say "stage 9 [$TAG] eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/regime_${TAG}_eval_stdout.log" | tail -1)"
say "=== stage 9 [$TAG] done ==="
