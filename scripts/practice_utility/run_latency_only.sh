#!/usr/bin/env bash
# Stage 13: a curriculum on the one axis that has headroom, and only that axis.
# Preregistered in manifests/lucid_latency_only_preregistration_20260828.json.
#
# The untrained origin is already robust to the five non-latency channels
# (60.5% at the full envelope, 56.2% at 1.25x it) and scores 0.00% with latency
# pinned at 60 ms. Channel attribution says latency carries 89% of the damage
# full-envelope training does. So the only axis where a curriculum could add
# capability is exactly the axis that destroys the policy at full strength --
# which is an argument for scheduling that axis gently and leaving the others
# alone, not for scheduling all six.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

MIN_FREE=9000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260828_054436/model_step_000024.pt
SEEDS="8600 8601 8602"
PRESETS="id_clean dr_050 dr_full dr_125"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

say "stage 13 (latency-only curriculum) armed; waiting for the budget curve"
while ! grep -q "budget curve done" "$LOG" 2>/dev/null; do sleep 120; done

say "stage 13: lucid_latonly_s4_rg ta_lucid_50_latonly_s4_rg x 3 seeds x 128 it"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 128 --warmup-iterations 10 \
  --seeds $SEEDS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
  --modes lucid_latonly_s4_rg ta_lucid_50_latonly_s4_rg \
  --execute > "$OUT/stage13_latonly_stdout.log" 2>&1
say "stage 13 training exit=$? receipt=$(receipt_of "$OUT/stage13_latonly_stdout.log")"
S13=$(receipt_of "$OUT/stage13_latonly_stdout.log")

if [ -n "$S13" ]; then
  say "eval H/stage13 on the severity grid"
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$S13" --num-envs 128 --seeds $SEEDS \
    --presets $PRESETS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
    --execute > "$OUT/eval_stage13_stdout.log" 2>&1
  say "eval H/stage13 exit=$? receipt=$(receipt_of "$OUT/eval_stage13_stdout.log")"
else
  say "SKIP eval stage 13: no training receipt"
fi

say "=== stage 13 done ==="
