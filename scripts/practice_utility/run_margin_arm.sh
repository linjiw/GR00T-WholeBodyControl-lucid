#!/usr/bin/env bash
# Stage 3 of the serial queue: campaign -> LUCID+PLR study -> THIS -> fixed_150.
# 3a: multi-seed ladder evaluation of the finished campaign (seeds 8601/8602,
#     all four arms, plus the h6000 mixture capsules) -- until this runs, only
#     seed 8600 has capability numbers and no cross-seed CI exists.
# 3b: the termination-margin arm (preregistered 775517eb) + its ladder.
# Waits for the PLR queue's terminal state because its gpu_idle_gate is one-shot.
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
LADDER="phys_000 phys_025 phys_050 phys_075 phys_100 phys_125 phys_150 phys_175 phys_200 lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
receipt_of() { grep -o "receipt [^ ]*json" "$1" | tail -1 | cut -d' ' -f2; }
plr_terminal() {
  phase=$(grep -o '"phase"[: ]*"[^"]*"' "$QSTATUS" 2>/dev/null | tail -1 | sed 's/.*"\([^"]*\)"$/\1/')
  case "$phase" in complete|failed) return 0;; esac
  kill -0 "$QPID" 2>/dev/null || return 0   # queue process gone without a terminal phase
  return 1
}

say "margin arm armed (v3, with multi-seed eval); waiting for the lucid campaign"
while ! grep -q "lucid campaign done" "$LOG" 2>/dev/null; do sleep 120; done
say "campaign done; waiting for the PLR study (queue pid $QPID) to reach a terminal state"
while ! plr_terminal; do sleep 300; done
say "PLR queue terminal; stage 3a: multi-seed campaign evaluation"

# --- 3a.1: seeds 8601/8602, all four arms, from the campaign's own receipt ---
CAMP=$(ls -t "$LUCID_ROOT"/manifests/curriculum_comparison_ne1024_20260829_000249*.json 2>/dev/null | head -1)
if [ -n "$CAMP" ]; then
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$CAMP" --panel-receipt "$P" \
    --num-envs 512 --seeds 8601 8602 --max-delay 12 \
    --presets $LADDER \
    --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/multiseed_eval_stdout.log" 2>&1
  say "multi-seed eval exit=$? receipt=$(receipt_of "$OUT/multiseed_eval_stdout.log")"
else
  say "multi-seed eval SKIPPED: campaign receipt not found"
fi

# --- 3a.2: h6000 mixture capsules for 8601/8602 (export + pseudo receipt + eval) ---
python - > "$OUT/h6000_export.log" 2>&1 <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from gear_sonic.research.practice_utility import branch_capsule as BC
from gear_sonic.research.practice_utility.paths import LUCID_ROOT

root = Path(LUCID_ROOT) / "artifacts/curriculum_comparison/curriculum_comparison_ne1024_20260829_000249"
template = json.load(open(Path(LUCID_ROOT) / "manifests/lucid_campaign_seed8600_all4.json"))
arms = {}
for seed in (8601, 8602):
    arm_dir = root / f"seed_{seed}/lucid_s4_rg"
    capsule = arm_dir / f"capsules/curriculum_comparison_ne1024_20260829_000249_s{seed}_lucid_s4_rg_h6000.capsule.pt"
    out = arm_dir / "checkpoint_h6000.pt"
    if not out.exists():
        BC.export_sonic_checkpoint(capsule, out)
    arms[f"lucid_s4_rg_h6000_s{seed}"] = {
        "seed": seed, "mode": "lucid_s4_rg_h6000", "checkpoint": str(out),
        "checkpoint_exported": True, "complete": True,
        "branch_id": f"curriculum_comparison_ne1024_20260829_000249_s{seed}_lucid_s4_rg_h6000",
    }
receipt = dict(template)
receipt["experiment_id"] = "lucid_campaign_h6000_s8601_s8602"
receipt["note"] = "h6000 mixture capsules for seeds 8601/8602, exported for the multi-seed ladder"
receipt["config"] = dict(template["config"], seeds=[8601, 8602], modes=["lucid_s4_rg_h6000"])
receipt["arms"] = arms
path = Path(LUCID_ROOT) / "manifests/lucid_campaign_h6000_s8601_s8602.json"
json.dump(receipt, open(path, "w"), indent=1)
print("pseudo receipt", path)
PYEOF
say "h6000 export exit=$? ($(tail -1 "$OUT/h6000_export.log"))"
H6R="$LUCID_ROOT/manifests/lucid_campaign_h6000_s8601_s8602.json"
if [ -f "$H6R" ]; then
  python scripts/practice_utility/run_curriculum_robustness_eval.py \
    --training-receipt "$H6R" --panel-receipt "$P" \
    --num-envs 512 --seeds 8601 8602 --modes lucid_s4_rg_h6000 --max-delay 12 \
    --presets $LADDER \
    --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/h6000_eval_stdout.log" 2>&1
  say "h6000 eval exit=$? receipt=$(receipt_of "$OUT/h6000_eval_stdout.log")"
else
  say "h6000 eval SKIPPED: pseudo receipt missing"
fi

# --- 3b: the margin arm, exactly as preregistered ---
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
  --presets $LADDER \
  --smpl-motion-file dummy --min-free-mib 6000 --execute > "$OUT/margin_arm_eval_stdout.log" 2>&1
say "margin arm eval exit=$? receipt=$(receipt_of "$OUT/margin_arm_eval_stdout.log")"
say "=== margin arm done ==="
