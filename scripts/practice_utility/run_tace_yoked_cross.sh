#!/usr/bin/env bash
# Stage 4: cross-seed yoked control, after the pilot driver finishes.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "driver done" "$LOG"; do sleep 60; done
SRC=$(cat "$OUT/tace_pilot.receipt")
say "stage 4: cross-seed yoked (ta_yoked_25x) from $SRC"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 32 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes ta_yoked_25x --yoked-source-receipt "$SRC" \
  --min-free-mib "$MIN_FREE" --execute > "$OUT/tace_yokedx_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/tace_yokedx_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 4 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 4 FAILED"; exit 1; }
echo "$receipt" > "$OUT/tace_yokedx.receipt"
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/tace_yokedx_eval_stdout.log" 2>&1
say "stage 4 eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/tace_yokedx_eval_stdout.log" | tail -1)"
say "=== stage 4 done ==="
