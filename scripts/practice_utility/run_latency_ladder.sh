#!/usr/bin/env bash
# Stage 11: the deployment-latency ladder, queued behind stages 9 and 10.
# Preregistered in manifests/lucid_latency_ladder_preregistration_20260828.json.
#
# Latency is pinned at 10/20/30/40/60 ms against nominal physics on every other
# channel, so it is the only axis moving. Five arms: the untrained origin, the
# no-DR control, the standard DR baseline, the LUCID-S proposal, and the capped
# proposal -- which trained on 0-20 ms and must be shown paying for that here if
# it pays anywhere.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"

MIN_FREE=9000
SEEDS="8600 8601 8602"
RUNGS="lat_10ms lat_20ms lat_30ms lat_40ms lat_60ms"
OUT="$LUCID_ROOT/outputs"
LOG="$OUT/lucid_s_driver.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }

# Chained behind the budget curve rather than behind stages 9/10, so the two
# never contend for the one card.
say "stage 11 (latency ladder) armed; waiting for the budget curve"
while ! grep -q "budget curve done" "$LOG" 2>/dev/null; do sleep 120; done

ladder() {  # $1 label, $2 training receipt, $3 stdout log, $4.. modes
  local label="$1" receipt="$2" log="$3"; shift 3
  [ -z "$receipt" ] && { say "SKIP ladder $label: no training receipt"; return 1; }
  say "ladder $label on rungs: $RUNGS"
  local modes=()
  [ "$#" -gt 0 ] && modes=(--modes "$@")
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$receipt" --num-envs 128 --seeds $SEEDS \
    --presets $RUNGS "${modes[@]}" --smpl-motion-file dummy \
    --min-free-mib "$MIN_FREE" --execute > "$log" 2>&1
  say "ladder $label exit=$? receipt=$(receipt_of "$log")"
}

S7=$(receipt_of "$OUT/stage7_chattr_stdout.log")
S8=$(receipt_of "$OUT/stage8_lucids_stdout.log")
S9=$(receipt_of "$OUT/stage9_latcap_stdout.log")

ladder "origin" "$LUCID_ROOT/manifests/origin_step24_local_pseudo_receipt.json" \
  "$OUT/ladder_origin_stdout.log"
ladder "stage7 (off, fixed)" "$S7" "$OUT/ladder_stage7_stdout.log" off fixed
ladder "stage8 (ta_lucid_50_s4_rg)" "$S8" "$OUT/ladder_stage8_stdout.log" ta_lucid_50_s4_rg
ladder "stage9 (ta_lucid_50_latcap_s4_rg)" "$S9" "$OUT/ladder_stage9_stdout.log" \
  ta_lucid_50_latcap_s4_rg

say "=== latency ladder done ==="
