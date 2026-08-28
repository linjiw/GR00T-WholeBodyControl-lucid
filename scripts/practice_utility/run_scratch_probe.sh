#!/usr/bin/env bash
# The long from-scratch probe: does episode length ever take off?
#
# One arm, no DR, 16 motions, fresh initialisation. Everything about it is a
# measured choice, not a guess:
#
#   num_envs=1280   largest that fits the user's 8 GB budget. Measured on the
#                   real configuration (delayed actuator + research callbacks)
#                   from a clean card: 7,887 MiB peak against a 751 MiB desktop
#                   baseline, 2.71 s/iter, 11,348 env-steps/s -- 3x the sample
#                   throughput of 256 envs. Receipt vram_ladder_20260828_131226.
#
#   terminations    tracking/base (0.25 m position, 1.0 rad orientation, no foot
#                   term). Under the stock TRAINING preset, which is stricter
#                   than the eval preset, 93% of from-scratch episodes die on
#                   tracking error in 0.25 s and 0.07% reach time-out. Relaxing
#                   nearly doubles episode length (12.8 -> 22.2) and lifts reward
#                   (0.98 -> 1.33). This is the probe's best shot at learning;
#                   handicapping it with thresholds calibrated for a competent
#                   policy would measure the threshold, not the learning.
#
#   8000 iterations 6.0 h, 245.8M transitions -- 6% of the 4.08e9 transitions
#                   SONIC's released policy was trained on. Capsules on a
#                   geometric ladder so the whole curve is measurable afterwards
#                   from ONE trajectory rather than across separate runs.
#
# The question this answers is narrow and worth answering before any campaign:
# episode length is currently pinned near the termination threshold. If it rises,
# an equal-budget multi-arm from-scratch comparison is an experiment. If it stays
# flat for 8000 iterations, it is a bet, and the finding is that this task is not
# learnable from scratch at this scale.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

export WANDB_MODE=online
export LUCID_GPU_WAIT_SECONDS=1800   # queue behind anything else, never die at the gate

OUT="$LUCID_ROOT/outputs"
LOG="$OUT/scratch_probe.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "long from-scratch probe: 1280 envs, train016, no DR, relaxed terminations, 8000 iterations"
say "wandb: project lucid-scratch (WANDB_MODE=online)"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch \
  --num-envs 1280 \
  --iterations 8000 \
  --warmup-iterations 10 \
  --seeds 8600 \
  --modes off \
  --terminations tracking/base \
  --wandb-project lucid-scratch \
  --horizons 100 250 500 1000 2000 4000 6000 \
  --motion-file "$LUCID_ROOT/pools/subsets/train016/robot_filtered" \
  --smpl-motion-file dummy \
  --min-free-mib 9000 \
  --execute > "$OUT/scratch_probe_stdout.log" 2>&1
say "probe exit=$? receipt=$(grep -o 'receipt [^ ]*json' "$OUT/scratch_probe_stdout.log" | tail -1 | cut -d' ' -f2)"
say "=== scratch probe done ==="
