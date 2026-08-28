#!/usr/bin/env bash
# Plain single-motion tracking, trained from scratch. No curriculum, no DR, no arms.
# The only question: does a fresh policy learn to track ONE motion at our scale?
#
# Motion: walk_hands_on_back_loop_002__A066_M (4.03 s, walk family, adaptation
# partition, zero overlap with the dev panel). Arms held still, so the wrist half
# of ee_body_pos is free -- measured at 0.000 on this clip against 0.903 on the
# 16-motion pool -- leaving foot placement as the single axis of difficulty.
#
# Thresholds are the values each terms/*.yaml declares (0.5/0.5/0.5/1.0), not the
# strict exp-preset overrides. Under strict, 93% of from-scratch episodes die on
# tracking error in 0.25 s and 0.07% reach time-out, so it would measure the
# threshold rather than the learning. Every number is upstream's own.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export WANDB_MODE=online
export LUCID_GPU_WAIT_SECONDS=1800

OUT="$LUCID_ROOT/outputs"
LOG="$OUT/single_motion_baseline.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "single-motion baseline: walk_hands_on_back_loop_002, 1024 envs, from scratch, no DR"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch \
  --num-envs 1024 \
  --iterations 20000 \
  --warmup-iterations 10 \
  --seeds 8600 \
  --modes off \
  --termination-thresholds default \
  --wandb-project lucid-single-motion \
  --horizons 250 500 1000 2000 4000 8000 12000 16000 \
  --motion-file "$LUCID_ROOT/pools/subsets/m1_hob002/robot_filtered" \
  --smpl-motion-file dummy \
  --min-free-mib 8000 \
  --execute > "$OUT/single_motion_baseline_stdout.log" 2>&1
say "exit=$? receipt=$(grep -o 'receipt [^ ]*json' "$OUT/single_motion_baseline_stdout.log" | tail -1 | cut -d' ' -f2)"
say "=== single-motion baseline done ==="
