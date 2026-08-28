#!/usr/bin/env bash
# LUCID-S campaign, second host (dedicated RTX 5080). Preregistered in
# manifests/lucid_support_expansion_preregistration_20260828.json.
#
#   stage 7  channel attribution + same-host references   (already in flight)
#   stage 8  LUCID-S arms                                  (this driver launches)
#   eval A   stage 7 branches, five presets
#   eval B   the untrained origin -- the retention reference
#   eval C   stage 8 branches, five presets
#
# The evaluation presets span the non-latency DR scale s in {0, 0.5, 1.0, 1.25}
# plus a 60 ms latency cell. s = 1.25 is outside the training envelope on
# purpose: a deployment claim is about conditions the randomization did not
# anticipate, and every robustness number before this campaign was measured
# inside the envelope the policy trained on.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

MIN_FREE=9000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260828_054436/model_step_000024.pt
SEEDS="8600 8601 8602"
PRESETS="id_clean dr_050 dr_full dr_125 latency_60ms"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

# --- wait for stage 7 -------------------------------------------------------
STAGE7_STDOUT="$OUT/stage7_chattr_stdout.log"
say "waiting for stage 7 (channel attribution) to write its receipt"
while ! grep -q "^receipt " "$STAGE7_STDOUT" 2>/dev/null; do sleep 60; done
S7=$(receipt_of "$STAGE7_STDOUT")
say "stage 7 receipt = $S7"
[ -z "$S7" ] && { say "STAGE 7 PRODUCED NO RECEIPT -- stopping"; exit 1; }

# --- stage 8: LUCID-S training ---------------------------------------------
say "stage 8: lucid_s4 lucid_rg lucid_s4_rg ta_lucid_50 ta_lucid_50_s4_rg x 3 seeds x 128 it"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 128 --warmup-iterations 10 \
  --seeds $SEEDS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
  --modes lucid_s4 lucid_rg lucid_s4_rg ta_lucid_50 ta_lucid_50_s4_rg \
  --execute > "$OUT/stage8_lucids_stdout.log" 2>&1
say "stage 8 training exit=$?"
S8=$(receipt_of "$OUT/stage8_lucids_stdout.log")
say "stage 8 receipt = $S8"

# --- evaluations ------------------------------------------------------------
evaluate() {  # $1 label, $2 training receipt, $3 stdout log
  [ -z "$2" ] && { say "SKIP eval $1: no training receipt"; return 1; }
  say "eval $1 on presets: $PRESETS"
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$2" --num-envs 128 --seeds $SEEDS \
    --presets $PRESETS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
    --execute > "$3" 2>&1
  say "eval $1 exit=$? receipt=$(receipt_of "$3")"
}

evaluate "B/origin" "$LUCID_ROOT/manifests/origin_step24_local_pseudo_receipt.json" \
  "$OUT/eval_origin_stdout.log"
evaluate "A/stage7" "$S7" "$OUT/eval_stage7_stdout.log"
evaluate "C/stage8" "$S8" "$OUT/eval_stage8_stdout.log"

say "=== lucid-s campaign done ==="
