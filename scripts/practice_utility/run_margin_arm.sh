#!/usr/bin/env bash
# The termination-margin arm, queued behind the live campaign AND the
# LUCID+PLR study (queue_lucid_plr_signal.py, another session's service).
# Preregistered: manifests/lucid_margin_signal_preregistration_20260830.json
# Chained third because the PLR queue's gpu_idle_gate is one-shot: if anything
# holds the GPU when it fires, that whole study dies unlaunched.
set -uo pipefail
LUCID_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${LUCID_ENV_SH:-$LUCID_REPO/../env/lucid_env.sh}"
cd "$LUCID_REPO"
export WANDB_MODE=online
export LUCID_GPU_WAIT_SECONDS=7200
OUT="$LUCID_ROOT/outputs"; LOG="$OUT/lucid_campaign.log"
P="$LUCID_ROOT/manifests/replicate_panel_panel_hob002_k512.json"
QSTATUS="$LUCID_ROOT/manifests/lucid_plr_signal_ne1024_20260830_queue_status.json"
QPID=1297554
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }
plr_terminal() {
  phase=$(grep -o '"phase"[: ]*"[^"]*"' "$QSTATUS" 2>/dev/null | tail -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$phase" in complete|failed) return 0;; esac
  kill -0 "$QPID" 2>/dev/null || return 0   # queue process gone without a terminal phase
  return 1
}

say "margin arm armed (rechained); waiting for the lucid campaign"
while ! grep -q "lucid campaign done" "$LOG" 2>/dev/null; do sleep 120; done
say "campaign done; waiting for the PLR study (queue pid $QPID) to reach a terminal state"
while ! plr_terminal; do sleep 300; done
say "PLR queue terminal (phase=$(grep -o '"phase"[: ]*"[^"]*"' "$QSTATUS" 2>/dev/null | tail -1)); margin arm starts"

say "margin arm: lucid_margin_s4_rg x 3 seeds x 8000 it"
python scripts/practice_utility/run_curriculum_comparison.py \
  --from-scratch --num-envs 1024 --iterations 8000 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes lucid_margin_s4_rg \
  --termination-thresholds default --wandb-project lucid-campaign \
  --horizons 500 1000 2000 4000 6000 \
  --motion-file "$LUCID_ROOT/pools/subsets/m1_hob002/robot_filtered" --smpl-motion-file dummy \
  --min-free-mib 8000 --execute > "$OUT/margin_arm_stdout.log" 2>&1
say "margin arm training exit=$? receipt=$(receipt_of "$OUT/margin_arm_stdout.log")"
R=$(receipt_of "$OUT/margin_arm_stdout.log")
CFG=$(grep -aoE "logs_rl/[^ ]*sonic_release_test-[0-9_]+" "$OUT"/curriculum_comparison_ne1024_*_s8600_lucid_margin_s4_rg.log 2>/dev/null | head -1)
[ -n "$R" ] && python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$R" --panel-receipt "$P" ${CFG:+--training-config "$CFG/config.yaml"} \
  --num-envs 512 --seeds 8600 8601 8602 --max-delay 12 \
  --presets phys_000 phys_025 phys_050 phys_075 phys_100 phys_125 phys_150 phys_175 phys_200 lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms \
  --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/margin_arm_eval_stdout.log" 2>&1
say "margin arm eval exit=$? receipt=$(receipt_of "$OUT/margin_arm_eval_stdout.log")"
say "=== margin arm done ==="
