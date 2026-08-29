#!/usr/bin/env bash
# Stage 15: unseen-motion panel (adapt4950 content-dev minus debug512, 200 motions).
set -uo pipefail
source /data/robotixx/lucid-sonic/lucid_env.sh
cd /home/robotixx/lucid/GR00T-WholeBodyControl
export LUCID_GPU_WAIT_SECONDS=43200
MIN_FREE=11000
OUT=/data/robotixx/lucid-sonic/outputs
MAN=/data/robotixx/lucid-sonic/manifests
LOG=$OUT/tace_pilot_driver.log
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "stage 14 done" "$LOG"; do sleep 60; done
PANEL="--pool-manifest $MAN/pool_adapt4950.json --split-manifest $MAN/split_adapt4950_content.json --partition dev \
  --suite-root /data/robotixx/lucid-sonic/pools/adapt4950/content_dev_heldout200 --exclude-pool-manifest $MAN/pool_debug512.json \
  --max-motions 200 --subset-salt heldout_v1 --smpl-motion-file /data/robotixx/lucid-sonic/pools/adapt4950/smpl_filtered"
run() { # tag receipt
  say "stage 15 [$1]: unseen 200-motion panel on $2"
  python scripts/practice_utility/run_curriculum_robustness_eval.py --training-receipt "$2" --num-envs 128 \
    --seeds 8600 8601 8602 --presets id_clean dr_full --min-free-mib "$MIN_FREE" --execute $PANEL > "$OUT/heldout_$1_stdout.log" 2>&1
  say "stage 15 [$1] eval exit=$?; receipt: $(grep -o 'receipt .*json' "$OUT/heldout_$1_stdout.log" | tail -1)"
}
run release "$MAN/release_step41550_pseudo_receipt.json"
run off128 "$MAN/curriculum_comparison_ne512_20260828_215844_combined_128.json"
run fixed512 "$MAN/curriculum_comparison_ne512_20260829_081600_fixed_512.json"
say "=== stage 15 done ==="
