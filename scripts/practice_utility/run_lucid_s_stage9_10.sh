#!/usr/bin/env bash
# Stages 9 and 10, queued behind run_lucid_s_campaign.sh.
#
#   stage 9   per-channel latency cap, 2x2 against stage 8's uncapped pair
#             (manifests/lucid_latency_cap_preregistration_20260828.json)
#   stage 10  batch-size validity control at num_envs=256
#             (manifests/lucid_batch_size_control_preregistration_20260828.json)
#
# Stage 10 is not optional and not conditional on stage 9: every horizon result
# in this programme was measured at num_envs=128, against a released policy
# trained at 4096. If the collapse turns out to be a small-batch artifact, the
# mechanism reading has to be retracted, and it is better to find that out here
# than in review.
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

say "stage 9/10 driver armed; waiting for the stage 7/8 campaign to finish"
while ! grep -q "lucid-s campaign done" "$LOG" 2>/dev/null; do sleep 120; done

train() {  # $1 label, $2 stdout log, rest: extra args
  local label="$1" log="$2"; shift 2
  say "$label training"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --iterations 128 --warmup-iterations 10 \
    --seeds $SEEDS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
    "$@" --execute > "$log" 2>&1
  say "$label training exit=$? receipt=$(receipt_of "$log")"
}

evaluate() {  # $1 label, $2 training receipt, $3 stdout log, $4 eval num_envs
  [ -z "$2" ] && { say "SKIP eval $1: no training receipt"; return 1; }
  say "eval $1"
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$2" --num-envs "${4:-128}" --seeds $SEEDS \
    --presets $PRESETS --smpl-motion-file dummy --min-free-mib "$MIN_FREE" \
    --execute > "$3" 2>&1
  say "eval $1 exit=$? receipt=$(receipt_of "$3")"
}

# Stage 10 runs FIRST. It decides how every other result in the programme is
# read: the untrained origin scores 90.2% clean and 60.8% dr_full, above every
# 128-iteration trained arm ever measured, so at this batch size all DR
# fine-tuning has been net-destructive even on the robustness it targets. If
# that reverses at 256 environments, the mechanism reading is a small-batch
# artifact and stage 9 is answering a question that does not arise.
# --- stage 10: batch-size control -------------------------------------------
# Trained at 256 envs; evaluated at 128 like every other arm, so the evaluation
# is identical across the batch-size contrast and only training differs.
train "stage 10 (batch-size control, 256 envs)" "$OUT/stage10_batch_stdout.log" \
  --num-envs 256 --modes fixed off ta_lucid_50_s4_rg
S10=$(receipt_of "$OUT/stage10_batch_stdout.log")
evaluate "E/stage10" "$S10" "$OUT/eval_stage10_stdout.log" 128

# --- stage 9: latency cap ---------------------------------------------------
train "stage 9 (latency cap 0.5)" "$OUT/stage9_latcap_stdout.log" \
  --num-envs 128 --latency-cap 0.5 \
  --modes lucid_latcap_s4_rg ta_lucid_50_latcap_s4_rg
S9=$(receipt_of "$OUT/stage9_latcap_stdout.log")
evaluate "D/stage9" "$S9" "$OUT/eval_stage9_stdout.log" 128

say "=== stages 9 and 10 done ==="
