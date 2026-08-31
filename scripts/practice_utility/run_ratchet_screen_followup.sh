#!/usr/bin/env bash
# Complete the preregistered seed-8601 ratchet screen after its live training run.
#
# This driver is intentionally resumable at receipt boundaries. It never reruns a
# completed evaluation, and it refuses to overwrite a partial or ambiguous receipt.
set -euo pipefail

readonly RAT_REPO="/home/linjiw/lucid/GR00T-WholeBodyControl"
readonly RAT_ENV="/home/linjiw/lucid/env/lucid_env.sh"
readonly RAT_PREREG="/home/linjiw/lucid-sonic/manifests/lucid_monotone_ratchet_preregistration_20260831.json"
readonly RAT_ENDPOINT_CLARIFICATION="/home/linjiw/lucid-sonic/manifests/lucid_monotone_ratchet_endpoint_clarification_20260831.json"
readonly RAT_TRAINING_RECEIPT="/home/linjiw/lucid-sonic/manifests/curriculum_comparison_ne1024_20260831_144022.json"
readonly RAT_TRAINING_CONFIG="${RAT_REPO}/logs_rl/lucid-campaign/manager/universal_token/all_modes/sonic_release_test-20260831_144024/config.yaml"
readonly RAT_BASELINE_RECEIPT="/home/linjiw/lucid-sonic/manifests/lucid_ratchet_fixed_s8601_baseline.json"
readonly RAT_BASELINE_CONFIG="${RAT_REPO}/logs_rl/lucid-campaign/manager/universal_token/all_modes/sonic_release_test-20260830_002425/config.yaml"
readonly RAT_PANEL="/home/linjiw/lucid-sonic/manifests/replicate_panel_panel_hob002_k512.json"
readonly RAT_TREATMENT_DIR="/home/linjiw/lucid-sonic/manifests/ratchet_screen_20260831/treatment"
readonly RAT_BASELINE_DIR="/home/linjiw/lucid-sonic/manifests/ratchet_screen_20260831/fixed"
readonly RAT_ANALYSIS="/home/linjiw/lucid-sonic/manifests/lucid_ratchet_screen_analysis_s8601_20260831.json"
readonly RAT_BASELINE_CHECKPOINT="/home/linjiw/lucid-sonic/artifacts/curriculum_comparison/curriculum_comparison_ne1024_20260829_000249/seed_8601/fixed/final_checkpoint.pt"
readonly RAT_TRAINING_DRIVER_PID=51631

assert_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "hash mismatch: ${path}: expected ${expected}, got ${actual}" >&2
        exit 1
    fi
}

wait_for_training_receipt() {
    local dead_checks=0
    while [[ ! -s "${RAT_TRAINING_RECEIPT}" ]]; do
        if kill -0 "${RAT_TRAINING_DRIVER_PID}" 2>/dev/null; then
            dead_checks=0
        else
            dead_checks=$((dead_checks + 1))
            if (( dead_checks >= 5 )); then
                echo "training driver exited without a receipt: ${RAT_TRAINING_RECEIPT}" >&2
                exit 1
            fi
        fi
        sleep 60
    done
    jq -e '
        (.verified | type == "array" and length > 0)
        and ([.arms[] | select(.seed == 8601 and .mode == "lucid_ratchet_rg")
              | .complete and .checkpoint_exported] == [true])
    ' "${RAT_TRAINING_RECEIPT}" >/dev/null
}

wait_for_idle_gpu() {
    local live
    while true; do
        live="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
        if [[ -z "${live}" ]]; then
            return
        fi
        echo "waiting for compute GPU to become idle; live PIDs: ${live//$'\n'/,}" >&2
        sleep 60
    done
}

validate_evaluation_receipt() {
    local receipt="$1"
    local mode="$2"
    local checkpoint_sha="$3"
    jq -e \
        --arg mode "${mode}" \
        --arg checkpoint_sha "${checkpoint_sha}" \
        --arg launcher_sha "308e24150e4d4f03d0abf0dc6a427063ac662904bb3a7765488a9bff63cd94ca" \
        --arg panel "${RAT_PANEL}" '
        def expected_presets: [
          "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
          "phys_125", "phys_150", "phys_175", "phys_200",
          "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms"
        ];
        (.verified | type == "array" and length > 0)
        and (.launcher_sha256 == $launcher_sha)
        and (.protocol.num_envs == 512)
        and (.protocol.checkpoint_seeds == [8601])
        and (.protocol.evaluation_seed_by_checkpoint_seed["8601"] == 8701)
        and (.protocol.modes == [$mode])
        and (.protocol.max_delay_capacity_steps == 12)
        and (.protocol.physics_step_ms == 5)
        and (.protocol.no_learning == true)
        and (.protocol.suite.motion_count == 512)
        and (.protocol.suite.replicate_panel.receipt == $panel)
        and (.protocol.suite.replicate_panel.replicates == 512)
        and (.protocol.suite.replicate_panel.alias_keys_sha256
             == "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430")
        and ((.runs | length) == 14)
        and ([.runs[] | select(.complete == true and .runtime.exit_code == 0)] | length == 14)
        and (([.runs[].preset] | sort) == (expected_presets | sort))
        and (([.runs[].mode] | unique) == [$mode])
        and (([.runs[].checkpoint_seed] | unique) == [8601])
        and (([.runs[].evaluation_seed] | unique) == [8701])
        and (.checkpoint_sha256_before == .checkpoint_sha256_after)
        and (([.checkpoint_sha256_before[]] | unique) == [$checkpoint_sha])
        and (([.runs[].checkpoint_sha256] | unique) == [$checkpoint_sha])
    ' "${receipt}" >/dev/null
}

run_or_reuse_evaluation() {
    local training_receipt="$1"
    local training_config="$2"
    local mode="$3"
    local receipt_dir="$4"
    local checkpoint_sha="$5"
    local existing=()

    mkdir -p "${receipt_dir}"
    shopt -s nullglob
    existing=("${receipt_dir}"/curriculum_robustness_ne512_*.json)
    shopt -u nullglob
    if (( ${#existing[@]} == 1 )); then
        validate_evaluation_receipt "${existing[0]}" "${mode}" "${checkpoint_sha}"
        echo "reusing verified evaluation receipt ${existing[0]}"
        return
    fi
    if (( ${#existing[@]} > 1 )); then
        echo "ambiguous evaluation receipts in ${receipt_dir}" >&2
        exit 1
    fi

    wait_for_idle_gpu
    python scripts/practice_utility/run_curriculum_robustness_eval.py \
        --training-receipt "${training_receipt}" \
        --training-config "${training_config}" \
        --panel-receipt "${RAT_PANEL}" \
        --num-envs 512 \
        --seeds 8601 \
        --modes "${mode}" \
        --eval-seed-base 8701 \
        --max-delay 12 \
        --presets \
            phys_000 phys_025 phys_050 phys_075 phys_100 \
            phys_125 phys_150 phys_175 phys_200 \
            lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms \
        --smpl-motion-file dummy \
        --receipt-dir "${receipt_dir}" \
        --min-free-mib 6000 \
        --execute

    shopt -s nullglob
    existing=("${receipt_dir}"/curriculum_robustness_ne512_*.json)
    shopt -u nullglob
    if (( ${#existing[@]} != 1 )); then
        echo "evaluation did not produce exactly one receipt in ${receipt_dir}" >&2
        exit 1
    fi
    validate_evaluation_receipt "${existing[0]}" "${mode}" "${checkpoint_sha}"
}

single_receipt() {
    local receipt_dir="$1"
    local receipts=()
    shopt -s nullglob
    receipts=("${receipt_dir}"/curriculum_robustness_ne512_*.json)
    shopt -u nullglob
    if (( ${#receipts[@]} != 1 )); then
        echo "expected exactly one receipt in ${receipt_dir}" >&2
        exit 1
    fi
    printf '%s\n' "${receipts[0]}"
}

main() {
    # Freeze every claim-bearing executable input before waiting or evaluating.
    assert_sha256 "${RAT_PREREG}" "11c2dca351e38db7b240728e5f57069a731410e5517c9f36b1d96c125962da32"
    assert_sha256 "${RAT_ENDPOINT_CLARIFICATION}" "d371f550c1e7c8f092eaa199c9fd369d6514ecf4f5aaa3d83c42d32b0dd5ef7c"
    assert_sha256 "${RAT_BASELINE_RECEIPT}" "089734629e188a48b5f1adc652c47f9af344531145e9a94b07995e3768345f73"
    assert_sha256 "${RAT_PANEL}" "e2e61933405e6701b0563eb4df793b6faf5c90d8ae5b7d8fc1e11f47142aefd7"
    assert_sha256 "${RAT_REPO}/scripts/practice_utility/run_curriculum_robustness_eval.py" "308e24150e4d4f03d0abf0dc6a427063ac662904bb3a7765488a9bff63cd94ca"
    assert_sha256 "${RAT_REPO}/scripts/practice_utility/analyze_ratchet.py" "ff33604e05e8ad0c5f0303997c1c60d34911fa66ec8e617f654bb3de4172be05"
    assert_sha256 "${RAT_TRAINING_CONFIG}" "3e0e4d3b0e2dd2f7a9abd57d61eb90daceb3b95bb2154d238005b15ff8776727"
    assert_sha256 "${RAT_BASELINE_CONFIG}" "d8e20c259517bba97a0c6779e7537e52f85327124466cd46dd12d7137671d98d"
    assert_sha256 "${RAT_BASELINE_CHECKPOINT}" "227a1c2f9822e8968fd17810ae2a2ea87193d0f5c00d052c37bb7f212df26bb1"

    wait_for_training_receipt
    local treatment_checkpoint
    local treatment_sha
    treatment_checkpoint="$(
        jq -er '.arms[] | select(.seed == 8601 and .mode == "lucid_ratchet_rg") | .checkpoint' \
            "${RAT_TRAINING_RECEIPT}"
    )"
    treatment_sha="$(sha256sum "${treatment_checkpoint}" | awk '{print $1}')"
    run_or_reuse_evaluation \
        "${RAT_TRAINING_RECEIPT}" "${RAT_TRAINING_CONFIG}" \
        lucid_ratchet_rg "${RAT_TREATMENT_DIR}" "${treatment_sha}"
    run_or_reuse_evaluation \
        "${RAT_BASELINE_RECEIPT}" "${RAT_BASELINE_CONFIG}" \
        fixed "${RAT_BASELINE_DIR}" "227a1c2f9822e8968fd17810ae2a2ea87193d0f5c00d052c37bb7f212df26bb1"

    if [[ -e "${RAT_ANALYSIS}" ]]; then
        echo "refusing to overwrite analysis receipt ${RAT_ANALYSIS}" >&2
        exit 1
    fi
    python scripts/practice_utility/analyze_ratchet.py \
        --robustness-receipt \
            "$(single_receipt "${RAT_TREATMENT_DIR}")" \
            "$(single_receipt "${RAT_BASELINE_DIR}")" \
        --training-receipt "${RAT_TRAINING_RECEIPT}" "${RAT_BASELINE_RECEIPT}" \
        --out "${RAT_ANALYSIS}"
    jq '{claim_scope, preregistered_decision, ratchet_vs_fixed, mechanism}' "${RAT_ANALYSIS}"
}

source "${RAT_ENV}"
export LUCID_GPU_WAIT_SECONDS=7200
cd "${RAT_REPO}"
main "$@"
