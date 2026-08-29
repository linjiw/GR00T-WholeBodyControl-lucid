#!/usr/bin/env bash
# Stage 11: combined regime (512 envs + small updates), off x 3 seeds x 128 it.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "stage 11 [combined_128]: off x 3 seeds x 128 it, 512 envs + small updates"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 512 --iterations 128 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes off --tag combined_128 --min-free-mib "$MIN_FREE" --execute \
  --extra-overrides ++algo.config.adaptive_lr_min=1e-6 ++algo.config.adaptive_lr_max=2e-5 ++algo.config.num_learning_epochs=1 \
  > "$OUT/combined_128_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/combined_128_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 11 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 11 FAILED"; exit 1; }
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_full --min-free-mib "$MIN_FREE" --execute > "$OUT/combined_128_eval_stdout.log" 2>&1
say "stage 11 eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/combined_128_eval_stdout.log" | tail -1)"
say "=== stage 11 [combined_128] done ==="
