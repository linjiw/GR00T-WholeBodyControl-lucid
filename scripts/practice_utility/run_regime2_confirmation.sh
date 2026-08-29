#!/usr/bin/env bash
# Stage 13: paper-grade seeds for off/fixed @128 and fixed @256 under the lossless regime.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
OV="++algo.config.adaptive_lr_min=1e-6 ++algo.config.adaptive_lr_max=2e-5 ++algo.config.num_learning_epochs=1"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cell() { # tag iterations seeds eval_seed_base modes...
  local tag=$1 it=$2 seeds=$3 evbase=$4; shift 4
  say "stage 13 [$tag]: $* x seeds $seeds x $it it, lossless regime"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --num-envs 512 --iterations "$it" --warmup-iterations 10 \
    --seeds $seeds --modes "$@" --tag "$tag" --min-free-mib "$MIN_FREE" --execute \
    --extra-overrides $OV > "$OUT/${tag}_stdout.log" 2>&1
  local receipt; receipt=$(grep -o "receipt .*json" "$OUT/${tag}_stdout.log" | tail -1 | cut -d' ' -f2)
  say "stage 13 [$tag] training exit=$? receipt=$receipt"
  [ -z "$receipt" ] && { say "STAGE 13 [$tag] FAILED"; return 1; }
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$receipt" --num-envs 128 --seeds $seeds --eval-seed-base $evbase \
    --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute > "$OUT/${tag}_eval_stdout.log" 2>&1
  say "stage 13 [$tag] eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/${tag}_eval_stdout.log" | tail -1)"
}
cell seeds_128 128 "8603 8604" 8703 off fixed
cell fixed_256 256 "8600 8601 8602" 8700 fixed
say "=== stage 13 done ==="
