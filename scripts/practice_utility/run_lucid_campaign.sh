#!/usr/bin/env bash
# LUCID vs fixed DR vs no DR, from scratch, on the single-motion testbed.
# Preregistered: manifests/lucid_single_motion_campaign_preregistration_20260829.json
#
# B = 8000, chosen from the baseline's own curve rather than from its configured
# horizon. Over 13,245 iterations the no-DR control's capability metrics plateaued
# early -- episode length 183.2 at window 4001-4500 against 187.4 at 12501-13000
# (+2.3% for 9,000 further iterations) and time_out 0.9703 -> 0.9921 -- while
# reward kept creeping (+21%), which is tracking precision, not capability. B=8000
# is double the control's plateau, giving the DR arms slack the control did not
# need, and costs 66 GPU-hours instead of 165.
#
# Four arms x three seeds, equal iterations, equal environment interactions
# (8000 x 1024 x 24 = 1.97e8 transitions per cell). One card, strictly serial.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export WANDB_MODE=online
export LUCID_GPU_WAIT_SECONDS=3600

OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_campaign.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "campaign: off / fixed / lucid_rg / lucid_s4_rg x 3 seeds x 8000 it, single motion, from scratch"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch \
  --num-envs 1024 \
  --iterations 8000 \
  --warmup-iterations 10 \
  --seeds 8600 8601 8602 \
  --modes off fixed lucid_rg lucid_s4_rg \
  --termination-thresholds default \
  --wandb-project lucid-campaign \
  --horizons 500 1000 2000 4000 6000 \
  --motion-file "$LUCID_ROOT/pools/subsets/m1_hob002/robot_filtered" \
  --smpl-motion-file dummy \
  --min-free-mib 8000 \
  --execute > "$OUT/lucid_campaign_stdout.log" 2>&1
say "campaign exit=$? receipt=$(grep -o 'receipt [^ ]*json' "$OUT/lucid_campaign_stdout.log" | tail -1 | cut -d' ' -f2)"
say "=== lucid campaign done ==="
