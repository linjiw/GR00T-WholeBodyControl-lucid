#!/usr/bin/env bash
# Stage 18: dose frontier point fixed_lat50 @512 it; seen, unseen and U(0,60ms) panels.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
CKPT=logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260818_141446/model_step_000024.pt
OUT=/data/robotixx/lucid-sonic/outputs
MAN=/data/robotixx/lucid-sonic/manifests
LOG=$OUT/tace_pilot_driver.log
OV="++algo.config.adaptive_lr_min=1e-6 ++algo.config.adaptive_lr_max=2e-5 ++algo.config.num_learning_epochs=1"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "stage 18 [fixed_lat50_512]: fixed_lat50 x 3 seeds x 512 it"
python scripts/practice_utility/run_curriculum_comparison.py \
  --checkpoint "$CKPT" --num-envs 512 --iterations 512 --warmup-iterations 10 \
  --seeds 8600 8601 8602 --modes fixed_lat50 --tag fixed_lat50_512 --min-free-mib "$MIN_FREE" --execute \
  --extra-overrides $OV > "$OUT/fixed_lat50_stdout.log" 2>&1
code=$?
receipt=$(grep -o "receipt .*json" "$OUT/fixed_lat50_stdout.log" | tail -1 | cut -d' ' -f2)
say "stage 18 training exit=$code receipt=$receipt"
[ -z "$receipt" ] && { say "STAGE 18 FAILED"; exit 1; }
python scripts/practice_utility/run_curriculum_robustness_eval.py --training-receipt "$receipt" --num-envs 128 \
  --seeds 8600 8601 8602 --presets id_clean dr_full lat_u60_common --min-free-mib "$MIN_FREE" --execute > "$OUT/fixed_lat50_seen_eval.log" 2>&1
say "stage 18 seen eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/fixed_lat50_seen_eval.log" | tail -1)"
python scripts/practice_utility/run_curriculum_robustness_eval.py --training-receipt "$receipt" --num-envs 128 \
  --seeds 8600 8601 8602 --presets id_clean dr_full --min-free-mib "$MIN_FREE" --execute \
  --pool-manifest $MAN/pool_adapt4950.json --split-manifest $MAN/split_adapt4950_content.json --partition dev \
  --suite-root /data/robotixx/lucid-sonic/pools/adapt4950/content_dev_heldout200 --exclude-pool-manifest $MAN/pool_debug512.json \
  --max-motions 200 --subset-salt heldout_v1 --smpl-motion-file /data/robotixx/lucid-sonic/pools/adapt4950/smpl_filtered \
  > "$OUT/fixed_lat50_unseen_eval.log" 2>&1
say "stage 18 unseen eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/fixed_lat50_unseen_eval.log" | tail -1)"
say "=== stage 18 done ==="
