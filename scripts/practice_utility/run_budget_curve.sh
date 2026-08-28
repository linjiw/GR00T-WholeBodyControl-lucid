#!/usr/bin/env bash
# Stage 12: the training-budget dose-response, on one host and one origin.
#
# The untrained settled origin scores 89.54% clean / 60.46% dr_full. Every
# 128-iteration arm ever measured is below it on both. The open question is
# whether that is a property of the 128-iteration budget or of DR fine-tuning at
# this scale at all -- i.e. whether there is *any* budget where training helps.
#
# The 32-iteration parity cell (lucid, fixed) already exists on this origin and
# has never been evaluated. This adds `off` and `ta_lucid_50_s4_rg` at the same
# budget, then evaluates all four, giving 0 -> 32 -> 128 iterations for four arms
# with the origin, pool, panel and eval seeds all held fixed.
#
# latency_60ms is deliberately omitted: it reads 0.00% for every policy ever
# measured, the untrained origin included, so it costs cells and discriminates
# nothing. Stage 11's ladder is the deployment-latency endpoint.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

MIN_FREE=9000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260828_054436/model_step_000024.pt
SEEDS="8600 8601 8602"
PRESETS="id_clean dr_050 dr_full dr_125"
PARITY="$LUCID_ROOT/manifests/curriculum_comparison_ne128_20260828_054615.json"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

say "stage 12 (budget curve) armed; waiting for stages 9 and 10"
while ! grep -q "stages 9 and 10 done" "$LOG" 2>/dev/null; do sleep 120; done

say "stage 12: off and ta_lucid_50_s4_rg at 32 iterations, matching the parity cell"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 128 --iterations 32 --warmup-iterations 10 \
  --seeds $SEEDS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
  --modes off ta_lucid_50_s4_rg --execute > "$OUT/stage12_train_stdout.log" 2>&1
say "stage 12 training exit=$? receipt=$(receipt_of "$OUT/stage12_train_stdout.log")"
S12=$(receipt_of "$OUT/stage12_train_stdout.log")

evaluate() {  # $1 label, $2 training receipt, $3 stdout log
  [ -z "$2" ] && { say "SKIP eval $1: no training receipt"; return 1; }
  say "eval $1 (32-iteration budget point)"
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$2" --num-envs 128 --seeds $SEEDS \
    --presets $PRESETS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
    --execute > "$3" 2>&1
  say "eval $1 exit=$? receipt=$(receipt_of "$3")"
}

evaluate "F/parity32 (lucid, fixed)" "$PARITY" "$OUT/eval_parity32_stdout.log"
evaluate "G/stage12 (off, ta_lucid_50_s4_rg)" "$S12" "$OUT/eval_stage12_stdout.log"

say "=== budget curve done ==="
