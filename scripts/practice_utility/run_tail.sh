#!/usr/bin/env bash
# Everything after the stage 7/8 evaluations, reordered around what stage 7's
# evaluation actually showed.
#
# WHY THE ORDER CHANGED. The held-out decomposition says plain PPO fine-tuning
# at num_envs=128 costs 23.0 profile-AUC points against the untrained origin
# (95% CI [-28.3, -17.9], paired over 102 motions x 3 seeds) -- about 90% of the
# total damage -- and that adding the full DR envelope on top costs a further
# 2.7 points whose interval covers zero. The collapse is not DR-induced. It is
# fine-tuning-induced, and the six-channel envelope is close to irrelevant to it.
#
# So the only question worth GPU time now is whether there is a fine-tuning
# configuration that is not destructive. Two probes, cheapest first:
#
#   stage 12  fewer updates  -- 32 iterations instead of 128
#   stage 10  larger batch   -- 256 environments instead of 128
#
# then, only because its premise survives independently of channel attribution
# (the origin scores 0.00% at a pinned 60 ms, so latency is the one axis with
# headroom left):
#
#   stage 13  a curriculum on latency alone
#   stage 11  the deployment-latency ladder
#
# Stage 9 (the latency cap) is DROPPED. It existed because training reward
# attributed 89% of the harm to the latency channel; held out, fixed_nolat and
# fixed_latonly are indistinguishable (-0.03 AUC, CI [-7.8, +9.1]). There is no
# channel to cap.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

MIN_FREE=9000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260828_054436/model_step_000024.pt
SEEDS="8600 8601 8602"
GRID="id_clean dr_050 dr_full dr_125"
RUNGS="lat_10ms lat_20ms lat_30ms lat_40ms lat_60ms"
PARITY="$LUCID_ROOT/manifests/curriculum_comparison_ne128_20260828_054615.json"
ORIGIN_RECEIPT="$LUCID_ROOT/manifests/origin_step24_local_pseudo_receipt.json"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

train() {  # label, stdout log, extra args...
  local label="$1" log="$2"; shift 2
  say "$label training"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --warmup-iterations 10 --seeds $SEEDS \
    --smpl-motion-file dummy --min-free-mib "$MIN_FREE" "$@" \
    --execute > "$log" 2>&1
  say "$label training exit=$? receipt=$(receipt_of "$log")"
}

evaluate() {  # label, training receipt, stdout log, presets, modes...
  local label="$1" receipt="$2" log="$3" presets="$4"; shift 4
  [ -z "$receipt" ] && { say "SKIP eval $label: no training receipt"; return 1; }
  local modes=()
  [ "$#" -gt 0 ] && modes=(--modes "$@")
  say "eval $label on: $presets"
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$receipt" --num-envs 128 --seeds $SEEDS \
    --presets $presets "${modes[@]}" --smpl-motion-file dummy \
    --min-free-mib "$MIN_FREE" --execute > "$log" 2>&1
  say "eval $label exit=$? receipt=$(receipt_of "$log")"
}

say "tail driver armed; waiting for the stage 7/8 campaign"
while ! grep -q "lucid-s campaign done" "$LOG" 2>/dev/null; do sleep 120; done

# --- stage 12: fewer updates ------------------------------------------------
# The 32-iteration parity cell (lucid, fixed) already exists on this origin and
# was never evaluated; `off` and ta_lucid_50_s4_rg join it at the same budget.
train "stage 12 (32 iterations)" "$OUT/stage12_train_stdout.log" \
  --num-envs 128 --iterations 32 --modes off ta_lucid_50_s4_rg
S12=$(receipt_of "$OUT/stage12_train_stdout.log")
evaluate "F/parity32" "$PARITY" "$OUT/eval_parity32_stdout.log" "$GRID"
evaluate "G/stage12" "$S12" "$OUT/eval_stage12_stdout.log" "$GRID"
say "=== budget curve done ==="

# --- stage 10: larger batch -------------------------------------------------
# Trained at 256 envs, evaluated at 128 like everything else, so only training
# differs across the contrast.
train "stage 10 (256 environments)" "$OUT/stage10_batch_stdout.log" \
  --num-envs 256 --iterations 128 --modes fixed off ta_lucid_50_s4_rg
S10=$(receipt_of "$OUT/stage10_batch_stdout.log")
evaluate "E/stage10" "$S10" "$OUT/eval_stage10_stdout.log" "$GRID"
say "=== batch control done ==="

# --- stage 13: a curriculum on the one axis with headroom -------------------
train "stage 13 (latency-only curriculum)" "$OUT/stage13_latonly_stdout.log" \
  --num-envs 128 --iterations 128 --modes lucid_latonly_s4_rg ta_lucid_50_latonly_s4_rg
S13=$(receipt_of "$OUT/stage13_latonly_stdout.log")
evaluate "H/stage13" "$S13" "$OUT/eval_stage13_stdout.log" "$GRID"
say "=== stage 13 done ==="

# --- stage 11: the deployment-latency ladder --------------------------------
S7=$(receipt_of "$OUT/stage7_chattr_stdout.log")
S8=$(receipt_of "$OUT/stage8_lucids_stdout.log")
evaluate "L/origin" "$ORIGIN_RECEIPT" "$OUT/ladder_origin_stdout.log" "$RUNGS"
evaluate "L/stage13" "$S13" "$OUT/ladder_stage13_stdout.log" "$RUNGS" \
  lucid_latonly_s4_rg ta_lucid_50_latonly_s4_rg
evaluate "L/stage7" "$S7" "$OUT/ladder_stage7_stdout.log" "$RUNGS" off fixed fixed_latonly
evaluate "L/stage8" "$S8" "$OUT/ladder_stage8_stdout.log" "$RUNGS" ta_lucid_50_s4_rg
say "=== latency ladder done ==="

say "=== tail complete ==="
