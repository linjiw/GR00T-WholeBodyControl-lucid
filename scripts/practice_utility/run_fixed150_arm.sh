#!/usr/bin/env bash
# The support-extension arm (fixed DR at 1.5x the envelope), queued FOURTH:
# campaign -> LUCID+PLR study -> margin arm -> this. Waits for the margin
# driver's completion marker; every stage upstream already waits for its own.
# Preregistered: manifests/lucid_fixed150_support_preregistration_20260830.json
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export WANDB_MODE=online
export LUCID_GPU_WAIT_SECONDS=7200
OUT="$LUCID_ROOT/outputs"; LOG="$OUT/lucid_campaign.log"
P="$LUCID_ROOT/manifests/replicate_panel_panel_hob002_k512.json"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

say "fixed_150 arm armed; waiting for the margin arm to finish"
while ! grep -q "=== margin arm done ===" "$LOG" 2>/dev/null; do sleep 300; done

say "fixed_150 arm: support extension x 3 seeds x 8000 it (max-delay 12)"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch --num-envs 1024 --iterations 8000 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes fixed_150 --max-delay 12 \
  --termination-thresholds default --wandb-project lucid-campaign \
  --horizons 500 1000 2000 4000 6000 \
  --motion-file "$LUCID_ROOT/pools/subsets/m1_hob002/robot_filtered" --smpl-motion-file dummy \
  --min-free-mib 8000 --execute > "$OUT/fixed150_arm_stdout.log" 2>&1
say "fixed_150 training exit=$? receipt=$(receipt_of "$OUT/fixed150_arm_stdout.log")"
R=$(receipt_of "$OUT/fixed150_arm_stdout.log")
CFG=$(grep -aoE "logs_rl/[^ ]*sonic_release_test-[0-9_]+" "$OUT"/curriculum_comparison_ne1024_*_s8600_fixed_150.log 2>/dev/null | head -1)
[ -n "$R" ] && python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$R" --panel-receipt "$P" ${CFG:+--training-config "$CFG/config.yaml"} \
  --num-envs 512 --seeds 8600 8601 8602 --max-delay 12 \
  --presets phys_000 phys_025 phys_050 phys_075 phys_100 phys_125 phys_150 phys_175 phys_200 lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms lat_60ms \
  --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/fixed150_arm_eval_stdout.log" 2>&1
say "fixed_150 eval exit=$? receipt=$(receipt_of "$OUT/fixed150_arm_eval_stdout.log")"

# Comparator top-up: H_X3 compares at 60 ms, which the campaign arms were never
# scored at. Eval-only, from the campaign's own training receipt.
CAMP=$(ls -t "$LUCID_ROOT"/manifests/curriculum_comparison_ne1024_20260829_000249*.json 2>/dev/null | head -1)
if [ -n "$CAMP" ]; then
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$CAMP" --panel-receipt "$P" \
    --num-envs 512 --seeds 8600 8601 8602 --max-delay 12 \
    --presets lat_60ms \
    --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/fixed150_comparator_lat60_stdout.log" 2>&1
  say "comparator lat_60ms exit=$? receipt=$(receipt_of "$OUT/fixed150_comparator_lat60_stdout.log")"
else
  say "comparator top-up skipped: campaign receipt not found"
fi
say "=== fixed_150 arm done ==="
