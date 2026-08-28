#!/usr/bin/env bash
# From-scratch feasibility pilot.
#
# The programme is pivoting away from fine-tuning the released checkpoint: held
# out, plain no-DR continuation costs 23.0 profile-AUC points against the
# untrained origin, so every arm comparison started there measured damage rather
# than learning. Curricula are a claim about learning and belong in a setting
# where learning is what happens.
#
# Before committing tens of GPU-hours to a from-scratch campaign, this asks the
# only question that decides whether such a campaign is possible at all: does a
# fresh policy learn anything measurable here, and at what pool size and batch?
# Three bounded runs, no claims, no arms -- just the slope of the reward curve
# and the wall-clock and memory it costs.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

SUBSETS="$LUCID_ROOT/pools/subsets"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/scratch_pilot.log"
ITERS=300
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

say "scratch pilot armed; waiting for the running evaluations to release the card"
while [ "$(free_mib)" -lt 11000 ]; do sleep 120; done
say "card free: $(free_mib) MiB"

pilot() {  # $1 subset, $2 num_envs
  local subset="$1" envs="$2" tag="scratch_${1}_ne${2}"
  say "pilot $tag: from scratch, $ITERS iterations, no DR"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --from-scratch --num-envs "$envs" --iterations "$ITERS" --warmup-iterations 10 \
    --seeds 8600 --modes off \
    --horizons 50 100 200 \
    --motion-file "$SUBSETS/$subset/robot_filtered" --smpl-motion-file dummy \
    --min-free-mib 9000 --execute > "$OUT/${tag}_stdout.log" 2>&1
  local code=$?
  say "pilot $tag exit=$code receipt=$(receipt_of "$OUT/${tag}_stdout.log")"
}

# Smallest pool first: if a fresh policy cannot learn 16 motions, it will not
# learn 64, and we would rather find that out in twenty minutes.
pilot train016 512
pilot train064 512
pilot train064 1024

say "=== scratch pilot done ==="
