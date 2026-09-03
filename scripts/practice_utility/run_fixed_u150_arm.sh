#!/usr/bin/env bash
# The expand-don't-replace support arm: a per-env lambda mixture with 75% of
# the focus cohort pinned at lambda = 1.5 (clamped) and a 25% uniform tail over
# the lower doses. Screening seed 8600 first, compared against the existing
# seed-8600 campaign cells on the clamped 14-preset instrument.
# Preregistered: manifests/lucid_fixed_u150_expand_support_preregistration_20260831.json
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export WANDB_MODE="${WANDB_MODE:-online}"
export LUCID_GPU_WAIT_SECONDS=7200
OUT="$LUCID_ROOT/outputs"; LOG="$OUT/lucid_campaign.log"
P="$LUCID_ROOT/manifests/replicate_panel_panel_hob002_k512.json"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

SEEDS="${FIXED_U150_SEEDS:-8600}"

say "fixed_u150 arm: 75%-frontier mixture at 1.5x, seeds $SEEDS x 8000 it (max-delay 12)"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch --num-envs 1024 --iterations 8000 --warmup-iterations 10 \
  --seeds $SEEDS --modes fixed_u150 --max-delay 12 \
  --termination-thresholds default --wandb-project lucid-campaign \
  --horizons 500 1000 2000 4000 6000 \
  --motion-file "$LUCID_ROOT/pools/subsets/m1_hob002/robot_filtered" --smpl-motion-file dummy \
  --min-free-mib 8000 --execute > "$OUT/fixed_u150_stdout.log" 2>&1
say "fixed_u150 training exit=$? receipt=$(receipt_of "$OUT/fixed_u150_stdout.log")"
R=$(receipt_of "$OUT/fixed_u150_stdout.log")
CFG=$(grep -aoE "logs_rl/[^ ]*sonic_release_test-[0-9_]+" "$OUT"/curriculum_comparison_ne1024_*_s8600_fixed_u150.log 2>/dev/null | head -1)
[ -n "$R" ] && python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$R" --panel-receipt "$P" ${CFG:+--training-config "$CFG/config.yaml"} \
  --num-envs 512 --seeds $SEEDS --max-delay 12 \
  --presets phys_000 phys_025 phys_050 phys_075 phys_100 phys_125 phys_150 phys_175 phys_200 lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms \
  --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/fixed_u150_eval_stdout.log" 2>&1
say "fixed_u150 eval exit=$? receipt=$(receipt_of "$OUT/fixed_u150_eval_stdout.log")"
say "=== fixed_u150 arm done ==="
