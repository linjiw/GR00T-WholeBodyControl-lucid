#!/usr/bin/env bash
# Stage 8: can fine-tuning be made non-destructive? Three no-DR cells x 32 it x 3 seeds.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "stage 7 done" "$LOG"; do sleep 60; done
run_cell() {  # name num_envs overrides...
  local name=$1 ne=$2; shift 2
  say "stage 8 cell $name: off x 3 seeds x 32 it, num_envs=$ne, overrides: $*"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --num-envs "$ne" --iterations 32 --warmup-iterations 10 \
    --seeds 8600 8601 8602 --modes off --tag "$name" --min-free-mib "$MIN_FREE" --execute \
    --extra-overrides "$@" > "$OUT/sustain_${name}_stdout.log" 2>&1
  local receipt; receipt=$(grep -o "receipt .*json" "$OUT/sustain_${name}_stdout.log" | tail -1 | cut -d' ' -f2)
  say "cell $name training exit=$? receipt=$receipt"
  [ -z "$receipt" ] && { say "cell $name FAILED"; return 1; }
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
    --presets id_clean dr_full --min-free-mib "$MIN_FREE" --execute > "$OUT/sustain_${name}_eval_stdout.log" 2>&1
  say "cell $name eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/sustain_${name}_eval_stdout.log" | tail -1)"
}
run_cell C2_update 128 ++algo.config.adaptive_lr_min=1e-6 ++algo.config.adaptive_lr_max=2e-5 ++algo.config.num_learning_epochs=1
run_cell C3_entropy 128 ++algo.config.entropy_coef=0.0
run_cell C1_batch 512
say "=== stage 8 done ==="
