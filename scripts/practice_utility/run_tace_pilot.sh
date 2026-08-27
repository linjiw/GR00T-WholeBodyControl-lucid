#!/usr/bin/env bash
# Capacity-gated TACE pilot: smoke -> 32-iter four-arm comparison -> frozen eval.
# Waits for the shared GPU instead of failing; every stage writes its receipt.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=${LUCID_GPU_WAIT_SECONDS:-43200}   # 12 h per arm max
MIN_FREE=${MIN_FREE:-11000}
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
MAN=/data/robotixx/lucid-sonic/manifests
LOG=$OUT/tace_pilot_driver.log
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

wait_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$MIN_FREE" ]; then say "GPU free ${free} MiB >= ${MIN_FREE}: go"; return 0; fi
    say "waiting: GPU free ${free} MiB < ${MIN_FREE}"; sleep 120
  done
}

say "=== TACE pilot driver start (commit $(git rev-parse --short HEAD)) ==="

# ---- stage 1: smoke -------------------------------------------------------
if [ ! -f "$OUT/tace_smoke.ok" ]; then
  wait_gpu
  say "stage 1: smoke ta_lucid_25 seed 8600, 16 iters"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --num-envs 128 --iterations 16 --warmup-iterations 4 \
    --seeds 8600 --modes ta_lucid_25 --min-free-mib "$MIN_FREE" --execute \
    > "$OUT/tace_smoke_stdout.log" 2>&1
  code=$?
  receipt=$(grep -o "receipt .*json" "$OUT/tace_smoke_stdout.log" | tail -1 | cut -d' ' -f2)
  say "smoke exit=$code receipt=$receipt"
  if [ "$code" -ne 0 ] || [ -z "$receipt" ]; then say "SMOKE FAILED — stopping"; exit 1; fi
  python - "$receipt" <<'PY' | tee -a "$LOG"
import json,sys
r=json.load(open(sys.argv[1]))
for arm in r["arms"].values():
    t=arm.get("tace_final") or {}
    print("smoke tace:", {k:t.get(k) for k in ("num_anchor","num_focus","anchor_delay_mean_steps","focus_delay_mean_steps")},
          "final_lambda", arm.get("final_lambda"), "rows", arm.get("curriculum_rows"))
    ok = t.get("num_anchor")==32 and t.get("num_focus")==96
    print("SMOKE_OK" if ok else "SMOKE_BAD_COHORTS")
PY
  grep -q SMOKE_OK "$LOG" && touch "$OUT/tace_smoke.ok" || { say "cohort check failed — stopping"; exit 1; }
fi

# ---- stage 2: four-arm comparison ----------------------------------------
if [ ! -f "$OUT/tace_pilot.receipt" ]; then
  wait_gpu
  say "stage 2: 4 arms x 3 seeds x 32 iters"
  python scripts/practice_utility/run_curriculum_comparison.py \
    --checkpoint "$CKPT" --num-envs 128 --iterations 32 --warmup-iterations 10 \
    --seeds 8600 8601 8602 --modes lucid fixed ta_lucid_25 ta_yoked_25 \
    --min-free-mib "$MIN_FREE" --execute > "$OUT/tace_pilot_stdout.log" 2>&1
  code=$?
  receipt=$(grep -o "receipt .*json" "$OUT/tace_pilot_stdout.log" | tail -1 | cut -d' ' -f2)
  say "pilot training exit=$code receipt=$receipt"
  if [ -z "$receipt" ]; then say "TRAINING FAILED — stopping"; exit 1; fi
  echo "$receipt" > "$OUT/tace_pilot.receipt"
fi
TRAIN_RECEIPT=$(cat "$OUT/tace_pilot.receipt")

# ---- stage 3: frozen evaluation ------------------------------------------
wait_gpu
say "stage 3: frozen eval on $TRAIN_RECEIPT"
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$TRAIN_RECEIPT" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/tace_eval_stdout.log" 2>&1
code=$?
say "eval exit=$code; receipt: $(grep -o 'receipt .*json' "$OUT/tace_eval_stdout.log" | tail -1)"
say "=== TACE pilot driver done ==="
