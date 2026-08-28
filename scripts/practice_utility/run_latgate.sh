#!/usr/bin/env bash
# Stage 10: latency-gated arms at 512 envs, 32 it (runs concurrently with stage 9 @128).
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=9000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "stage 10 [latgate_32]: lucid_latgate ta_latgate_50 x 3 seeds x 32 it, num_envs=512"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 512 --iterations 32 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes lucid_latgate ta_latgate_50 --tag latgate_32 \
  --min-free-mib "$MIN_FREE" --execute > "$OUT/latgate_32_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/latgate_32_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 10 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 10 FAILED"; exit 1; }
python scripts/practice_utility/run_curriculum_robustness_eval.py \
  --training-receipt "$receipt" --num-envs 128 --seeds 8600 8601 8602 \
  --presets id_clean dr_050 dr_full latency_60ms --min-free-mib "$MIN_FREE" --execute \
  > "$OUT/latgate_32_eval_stdout.log" 2>&1
say "stage 10 eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/latgate_32_eval_stdout.log" | tail -1)"
say "=== stage 10 [latgate_32] done ==="
