#!/usr/bin/env bash
# Stage 16: shared 0-60 ms deployment-latency evaluation of the operating point.
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
OUT=/data/robotixx/lucid-sonic/outputs
MAN=/data/robotixx/lucid-sonic/manifests
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "stage 15 done" "$LOG"; do sleep 60; done
run() {
  say "stage 16 [$1]: shared U(0,60ms) presets on $2"
  python scripts/practice_utility/run_curriculum_robustness_eval.py --training-receipt "$2" --num-envs 128 \
    --seeds 8600 8601 8602 --presets lat_u60_common dr_full_lat_u60_common --min-free-mib "$MIN_FREE" --execute > "$OUT/deplat_$1_stdout.log" 2>&1
  say "stage 16 [$1] eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/deplat_$1_stdout.log" | tail -1)"
}
run release "$MAN/release_step41550_pseudo_receipt.json"
run off128 "$MAN/curriculum_comparison_ne512_20260828_215844_combined_128.json"
run fixed512 "$MAN/curriculum_comparison_ne512_20260829_081600_fixed_512.json"
say "=== stage 16 done ==="
