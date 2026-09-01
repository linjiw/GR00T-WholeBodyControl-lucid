#!/usr/bin/env bash
# Run the preregistered three-seed monotone-ratchet confirmation.
#
# Training is split into one-cell receipt boundaries.  A .started marker with
# no valid receipt is a deliberate fail-closed state: preserve the partial run
# and file a deviation instead of resuming or silently retraining it.
set -euo pipefail

readonly RAT_REPO="/home/linjiw/lucid-ratchet-confirm"
readonly RAT_SOURCE_REPO="/home/linjiw/lucid/GR00T-WholeBodyControl"
readonly RAT_ENV="/home/linjiw/lucid/env/lucid_env.sh"
readonly RAT_ROOT="/home/linjiw/lucid-sonic/manifests/ratchet_confirmation_20260831"
readonly RAT_PREREG="/home/linjiw/lucid-sonic/manifests/lucid_monotone_ratchet_confirmation_amendment_20260831.json"
readonly RAT_PANEL="/home/linjiw/lucid-sonic/manifests/replicate_panel_panel_hob002_k512.json"
readonly RAT_MOTION="/home/linjiw/lucid-sonic/pools/subsets/m1_hob002/robot_filtered"
readonly RAT_ENCODER="/home/linjiw/lucid-sonic/artifacts/lucid_encoder_debug512.pt"
readonly RAT_EVAL_ARTIFACT_ROOT="/home/linjiw/lucid-sonic/artifacts/ratchet_confirmation_eval_20260831"
readonly RAT_EVAL_LOG_ROOT="/home/linjiw/lucid-sonic/outputs/ratchet_confirmation_20260831"

readonly RAT_SCREEN_TRAINING="/home/linjiw/lucid-sonic/manifests/curriculum_comparison_ne1024_20260831_144022.json"
readonly RAT_SCREEN_RATCHET_CONFIG="${RAT_SOURCE_REPO}/logs_rl/lucid-campaign/manager/universal_token/all_modes/sonic_release_test-20260831_144024/config.yaml"
readonly RAT_SCREEN_RATCHET_EVAL="/home/linjiw/lucid-sonic/manifests/ratchet_screen_20260831/treatment/curriculum_robustness_ne512_20260831_201524.json"
readonly RAT_SCREEN_FIXED_EVAL="/home/linjiw/lucid-sonic/manifests/ratchet_screen_20260831/fixed/curriculum_robustness_ne512_20260831_202312.json"
readonly RAT_FIXED_8600_RECEIPT="${RAT_ROOT}/baseline/fixed_s8600_bridge.json"
readonly RAT_FIXED_8600_CONFIG="/home/linjiw/lucid-sonic/artifacts/ratchet_confirmation_20260831/baseline/fixed_s8600/config.yaml"
readonly RAT_FIXED_8601_RECEIPT="${RAT_ROOT}/baseline/fixed_s8601_bridge.json"
readonly RAT_FIXED_8601_CONFIG="${RAT_SOURCE_REPO}/logs_rl/lucid-campaign/manager/universal_token/all_modes/sonic_release_test-20260830_002425/config.yaml"

readonly RAT_TRAIN_R8600="${RAT_ROOT}/training/ratchet_s8600"
readonly RAT_TRAIN_F8602="${RAT_ROOT}/training/fixed_s8602"
readonly RAT_TRAIN_R8602="${RAT_ROOT}/training/ratchet_s8602"
readonly RAT_EVAL_R8600="${RAT_ROOT}/evaluation/ratchet_s8600"
readonly RAT_EVAL_F8600="${RAT_ROOT}/evaluation/fixed_s8600"
readonly RAT_EVAL_R8602="${RAT_ROOT}/evaluation/ratchet_s8602"
readonly RAT_EVAL_F8602="${RAT_ROOT}/evaluation/fixed_s8602"
readonly RAT_FREEZE_ROOT="${RAT_ROOT}/frozen_checkpoints"
readonly RAT_ANALYSIS="${RAT_ROOT}/lucid_ratchet_confirmation_analysis.json"

readonly -a RAT_PRESETS=(
    phys_000 phys_025 phys_050 phys_075 phys_100
    phys_125 phys_150 phys_175 phys_200
    lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms
)

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

assert_preregistered_state() {
    : "${LUCID_RATCHET_CONFIRM_PREREG_SHA256:?set the frozen preregistration SHA-256}"
    assert_sha256 "${RAT_PREREG}" "${LUCID_RATCHET_CONFIRM_PREREG_SHA256}"

    local expected_git
    expected_git="$(jq -er '.code_state.git_sha' "${RAT_PREREG}")"
    if [[ "$(git -C "${RAT_REPO}" rev-parse HEAD)" != "${expected_git}" ]]; then
        echo "SONIC HEAD differs from preregistered commit ${expected_git}" >&2
        exit 1
    fi
    git -C "${RAT_REPO}" diff --quiet
    git -C "${RAT_REPO}" diff --cached --quiet
    if [[ -n "$(git -C "${RAT_REPO}" status --porcelain --untracked-files=all)" ]]; then
        echo "confirmation worktree is not clean: ${RAT_REPO}" >&2
        exit 1
    fi

    local relative expected
    while IFS=$'\t' read -r relative expected; do
        assert_sha256 "${RAT_REPO}/${relative}" "${expected}"
    done < <(jq -r '.code_state.file_sha256 | to_entries[] | [.key, .value] | @tsv' "${RAT_PREREG}")

    while IFS=$'\t' read -r relative expected; do
        assert_sha256 "${relative}" "${expected}"
    done < <(jq -r '.frozen_inputs | to_entries[] | [.value.path, .value.sha256] | @tsv' "${RAT_PREREG}")
}

wait_for_idle_gpu() {
    local live
    while true; do
        live="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
        if [[ -z "${live}" ]]; then
            return
        fi
        echo "waiting for compute GPU; live PIDs: ${live//$'\n'/,}" >&2
        sleep 60
    done
}

single_receipt() {
    local directory="$1"
    local pattern="$2"
    local receipts=()
    shopt -s nullglob
    receipts=("${directory}"/${pattern})
    shopt -u nullglob
    if (( ${#receipts[@]} != 1 )); then
        echo "expected one ${pattern} receipt in ${directory}, found ${#receipts[@]}" >&2
        return 1
    fi
    printf '%s\n' "${receipts[0]}"
}

validate_training_receipt() {
    local receipt="$1"
    local seed="$2"
    local mode="$3"
    jq -e --argjson seed "${seed}" --arg mode "${mode}" --arg motion "${RAT_MOTION}" '
        (.verified | type == "array" and length > 0)
        and (.config.from_scratch == true)
        and (.config.num_envs == 1024)
        and (.config.iterations == 8000)
        and (.config.warmup_iterations == 10)
        and (.config.seeds == [$seed])
        and (.config.modes == [$mode])
        and (.config.termination_thresholds == "default")
        and (.config.max_delay_steps == 8)
        and (.config.consolidation_fraction == 0)
        and (.config.motion_file == $motion)
        and (.config.smpl_motion_file == "dummy")
        and (.config.controller.delta_target == 0.778)
        and (.config.controller.kp == 1.0)
        and (.config.controller.ki == 0.02)
        and (.config.controller.alpha == 0.05)
        and (.config.controller.integral_max == 1.0)
        and (.config.controller.return_floor == 8.0)
        and ([.arms[] | select(
            .seed == $seed and .mode == $mode and .complete == true
            and .checkpoint_exported == true and .iterations_parsed == 8000
            and .curriculum_rows == 8000
            and ($mode != "lucid_ratchet_rg" or (
                .arm_spec.monotonic == true
                and .arm_spec.return_guard == "relative"
                and .arm_spec.spread_strata == 1
                and .arm_spec.fixed_lambda == 1.0
                and .arm_spec.allow_extrapolation == false
            ))
            and ($mode != "fixed" or (
                .arm_spec.curriculum_mode == "fixed"
                and .arm_spec.spread_strata == 1
                and .arm_spec.fixed_lambda == 1.0
                and .arm_spec.allow_extrapolation == false
            ))
        )] | length == 1)
    ' "${receipt}" >/dev/null

    local checkpoint curriculum
    checkpoint="$(jq -er --argjson seed "${seed}" --arg mode "${mode}" \
        '.arms[] | select(.seed == $seed and .mode == $mode) | .checkpoint' "${receipt}")"
    curriculum="$(jq -er --argjson seed "${seed}" --arg mode "${mode}" \
        '.arms[] | select(.seed == $seed and .mode == $mode) | .curriculum_path' "${receipt}")"
    [[ -f "${checkpoint}" && -f "${curriculum}" ]]
}

run_or_reuse_training() {
    local seed="$1"
    local mode="$2"
    local receipt_dir="$3"
    mkdir -p "${receipt_dir}"

    local receipt
    if receipt="$(single_receipt "${receipt_dir}" 'curriculum_comparison_ne1024_*.json' 2>/dev/null)"; then
        validate_training_receipt "${receipt}" "${seed}" "${mode}"
        echo "reusing complete training receipt ${receipt}"
        return
    fi
    if find "${receipt_dir}" -maxdepth 1 -type f -name 'curriculum_comparison_ne1024_*.json' \
        -print -quit | grep -q .; then
        echo "partial or ambiguous training receipt set in ${receipt_dir}" >&2
        exit 1
    fi
    if [[ -e "${receipt_dir}/.started" ]]; then
        echo "training was previously started without a complete receipt: ${receipt_dir}" >&2
        echo "preserve it and file a from-scratch deviation; automatic retry is forbidden" >&2
        exit 1
    fi
    mkdir "${receipt_dir}/.started"
    assert_preregistered_state
    wait_for_idle_gpu
    python scripts/practice_utility/run_curriculum_comparison.py \
        --from-scratch \
        --num-envs 1024 \
        --iterations 8000 \
        --warmup-iterations 10 \
        --horizons 500 1000 2000 4000 6000 \
        --seeds "${seed}" \
        --modes "${mode}" \
        --termination-thresholds default \
        --wandb-project lucid-campaign \
        --motion-file "${RAT_MOTION}" \
        --smpl-motion-file dummy \
        --encoder "${RAT_ENCODER}" \
        --max-delay 8 \
        --delta-target 0.778 \
        --kp 1.0 \
        --ki 0.02 \
        --alpha 0.05 \
        --integral-max 1.0 \
        --return-floor 8.0 \
        --return-relative-drop 0.25 \
        --return-window 8 \
        --receipt-dir "${receipt_dir}" \
        --min-free-mib 6000 \
        --execute
    receipt="$(single_receipt "${receipt_dir}" 'curriculum_comparison_ne1024_*.json')"
    validate_training_receipt "${receipt}" "${seed}" "${mode}"
}

config_for_training_receipt() {
    local receipt="$1"
    local seed="$2"
    local mode="$3"
    local run_dir
    run_dir="$(jq -er --argjson seed "${seed}" --arg mode "${mode}" \
        '.arms[] | select(.seed == $seed and .mode == $mode) | .arm_spec.run_dir' "${receipt}")"
    if [[ "${run_dir}" = /* ]]; then
        printf '%s/config.yaml\n' "${run_dir}"
    else
        printf '%s/%s/config.yaml\n' "${RAT_REPO}" "${run_dir}"
    fi
}

freeze_or_reuse_checkpoint() {
    local receipt="$1"
    local config="$2"
    local seed="$3"
    local mode="$4"
    local out="${RAT_FREEZE_ROOT}/${mode}_s${seed}.json"
    if [[ ! -e "${out}" ]]; then
        assert_preregistered_state
        python scripts/practice_utility/freeze_training_checkpoint.py \
            --training-receipt "${receipt}" \
            --config "${config}" \
            --seed "${seed}" \
            --mode "${mode}" \
            --iterations 8000 \
            --make-read-only \
            --out "${out}"
    fi
    jq -e --argjson seed "${seed}" --arg mode "${mode}" '
        .state == "frozen_for_evaluation"
        and .seed == $seed and .mode == $mode
        and .iterations == 8000 and .resume_forbidden == true
        and .checkpoint.read_only == true
    ' "${out}" >/dev/null
    local expected_checkpoint
    expected_checkpoint="$(jq -er --argjson seed "${seed}" --arg mode "${mode}" \
        '.arms[] | select(.seed == $seed and .mode == $mode) | .checkpoint' "${receipt}")"
    if [[ "$(jq -er '.training_receipt.path' "${out}")" != "$(readlink -f "${receipt}")" \
        || "$(jq -er '.config.path' "${out}")" != "$(readlink -f "${config}")" \
        || "$(jq -er '.checkpoint.path' "${out}")" != "$(readlink -f "${expected_checkpoint}")" ]]; then
        echo "freeze manifest is not bound to the requested receipt/config/checkpoint: ${out}" >&2
        exit 1
    fi
    local section path expected
    for section in checkpoint config curriculum final_capsule training_receipt; do
        path="$(jq -er --arg section "${section}" '.[$section].path' "${out}")"
        expected="$(jq -er --arg section "${section}" '.[$section].sha256' "${out}")"
        assert_sha256 "${path}" "${expected}"
    done
    local checkpoint_mode
    checkpoint_mode="$(stat -c '%a' "$(jq -er '.checkpoint.path' "${out}")")"
    if (( (8#${checkpoint_mode}) & 8#222 )); then
        echo "frozen checkpoint regained write bits: $(jq -er '.checkpoint.path' "${out}")" >&2
        exit 1
    fi
}

validate_evaluation_receipt() {
    local receipt="$1"
    local seed="$2"
    local mode="$3"
    local eval_seed="$4"
    local checkpoint_sha="$5"
    local training_config="$6"
    local evaluator_sha training_config_source training_config_sha
    evaluator_sha="$(jq -er '.evaluation.evaluator_sha256' "${RAT_PREREG}")"
    training_config_source="$(readlink -f "${training_config}")"
    training_config_sha="$(sha256sum "${training_config_source}" | awk '{print $1}')"
    jq -e \
        --argjson seed "${seed}" \
        --arg mode "${mode}" \
        --argjson eval_seed "${eval_seed}" \
        --arg checkpoint_sha "${checkpoint_sha}" \
        --arg evaluator_sha "${evaluator_sha}" \
        --arg training_config_source "${training_config_source}" \
        --arg training_config_sha "${training_config_sha}" \
        --arg panel "${RAT_PANEL}" '
        def expected_presets: [
          "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
          "phys_125", "phys_150", "phys_175", "phys_200",
          "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms"
        ];
        (.verified | type == "array" and length > 0)
        and (.launcher_sha256 == $evaluator_sha)
        and (.protocol.num_envs == 512)
        and (.protocol.checkpoint_seeds == [$seed])
        and (.protocol.evaluation_seed_by_checkpoint_seed[($seed | tostring)] == $eval_seed)
        and (.protocol.modes == [$mode])
        and (.protocol.max_delay_capacity_steps == 12)
        and (.protocol.physics_step_ms == 5)
        and (.protocol.no_learning == true)
        and (.protocol.resolved_training_config.source == $training_config_source)
        and (.protocol.resolved_training_config.sha256 == $training_config_sha)
        and (.protocol.suite.motion_count == 512)
        and (.protocol.suite.replicate_panel.receipt == $panel)
        and (.protocol.suite.replicate_panel.replicates == 512)
        and (.protocol.suite.replicate_panel.alias_keys_sha256
             == "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430")
        and ((.runs | length) == 14)
        and ([.runs[] | select(.complete == true and .runtime.exit_code == 0)] | length == 14)
        and (([.runs[].preset] | sort) == (expected_presets | sort))
        and (([.runs[].mode] | unique) == [$mode])
        and (([.runs[].checkpoint_seed] | unique) == [$seed])
        and (([.runs[].evaluation_seed] | unique) == [$eval_seed])
        and (.checkpoint_sha256_before == .checkpoint_sha256_after)
        and (([.checkpoint_sha256_before[]] | unique) == [$checkpoint_sha])
        and (([.runs[].checkpoint_sha256] | unique) == [$checkpoint_sha])
    ' "${receipt}" >/dev/null
}

run_or_reuse_evaluation() {
    local training_receipt="$1"
    local training_config="$2"
    local seed="$3"
    local mode="$4"
    local eval_seed="$5"
    local receipt_dir="$6"
    mkdir -p "${receipt_dir}"

    local checkpoint checkpoint_sha receipt
    checkpoint="$(jq -er --argjson seed "${seed}" --arg mode "${mode}" \
        '.arms[] | select(.seed == $seed and .mode == $mode) | .checkpoint' "${training_receipt}")"
    checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    if receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json' 2>/dev/null)"; then
        validate_evaluation_receipt \
            "${receipt}" "${seed}" "${mode}" "${eval_seed}" "${checkpoint_sha}" \
            "${training_config}"
        echo "reusing complete evaluation receipt ${receipt}"
        return
    fi
    if find "${receipt_dir}" -maxdepth 1 -type f -name 'curriculum_robustness_ne512_*.json' \
        -print -quit | grep -q .; then
        echo "partial or ambiguous evaluation receipt set in ${receipt_dir}" >&2
        exit 1
    fi
    if [[ -e "${receipt_dir}/.started" ]]; then
        echo "evaluation was previously started without one valid receipt: ${receipt_dir}" >&2
        exit 1
    fi
    mkdir "${receipt_dir}/.started"
    assert_preregistered_state
    wait_for_idle_gpu
    python scripts/practice_utility/run_curriculum_robustness_eval.py \
        --training-receipt "${training_receipt}" \
        --training-config "${training_config}" \
        --panel-receipt "${RAT_PANEL}" \
        --num-envs 512 \
        --seeds "${seed}" \
        --modes "${mode}" \
        --eval-seed-base "${eval_seed}" \
        --max-delay 12 \
        --presets "${RAT_PRESETS[@]}" \
        --smpl-motion-file dummy \
        --artifact-root "${RAT_EVAL_ARTIFACT_ROOT}" \
        --log-dir "${RAT_EVAL_LOG_ROOT}" \
        --receipt-dir "${receipt_dir}" \
        --min-free-mib 6000 \
        --execute
    receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json')"
    validate_evaluation_receipt \
        "${receipt}" "${seed}" "${mode}" "${eval_seed}" "${checkpoint_sha}" \
        "${training_config}"
}

validate_screen_inputs() {
    local r8601_sha f8601_sha r8601_checkpoint f8601_checkpoint
    r8601_sha="$(jq -er '.checkpoint_sha256_before[]' "${RAT_SCREEN_RATCHET_EVAL}")"
    f8601_sha="$(jq -er '.checkpoint_sha256_before[]' "${RAT_SCREEN_FIXED_EVAL}")"
    validate_evaluation_receipt \
        "${RAT_SCREEN_RATCHET_EVAL}" 8601 lucid_ratchet_rg 8701 "${r8601_sha}" \
        "${RAT_SCREEN_RATCHET_CONFIG}"
    validate_evaluation_receipt \
        "${RAT_SCREEN_FIXED_EVAL}" 8601 fixed 8701 "${f8601_sha}" \
        "${RAT_FIXED_8601_CONFIG}"
    validate_training_receipt "${RAT_SCREEN_TRAINING}" 8601 lucid_ratchet_rg
    r8601_checkpoint="$(jq -er '.arms[] | select(.seed == 8601 and .mode == "lucid_ratchet_rg") | .checkpoint' "${RAT_SCREEN_TRAINING}")"
    f8601_checkpoint="$(jq -er '.arms[] | select(.seed == 8601 and .mode == "fixed") | .checkpoint' "${RAT_FIXED_8601_RECEIPT}")"
    assert_sha256 "${r8601_checkpoint}" "${r8601_sha}"
    assert_sha256 "${f8601_checkpoint}" "${f8601_sha}"
}

run_analysis() {
    local r8600_train="$1"
    local r8602_train="$2"
    local r8600_eval f8600_eval r8602_eval f8602_eval
    r8600_eval="$(single_receipt "${RAT_EVAL_R8600}" 'curriculum_robustness_ne512_*.json')"
    f8600_eval="$(single_receipt "${RAT_EVAL_F8600}" 'curriculum_robustness_ne512_*.json')"
    r8602_eval="$(single_receipt "${RAT_EVAL_R8602}" 'curriculum_robustness_ne512_*.json')"
    f8602_eval="$(single_receipt "${RAT_EVAL_F8602}" 'curriculum_robustness_ne512_*.json')"

    local -a eval_receipts=(
        "${r8600_eval}" "${f8600_eval}"
        "${RAT_SCREEN_RATCHET_EVAL}" "${RAT_SCREEN_FIXED_EVAL}"
        "${r8602_eval}" "${f8602_eval}"
    )
    local -a training_receipts=(
        "${r8600_train}" "${RAT_SCREEN_TRAINING}" "${r8602_train}"
    )

    if [[ -e "${RAT_ANALYSIS}" ]]; then
        # Recompute the deterministic receipt and compare every scientific
        # field before reusing an existing file after a reboot.
        local replay_dir replay
        replay_dir="$(mktemp -d /tmp/lucid-ratchet-confirmation-replay.XXXXXX)"
        replay="${replay_dir}/analysis.json"
        assert_preregistered_state
        python scripts/practice_utility/analyze_ratchet.py \
            --robustness-receipt "${eval_receipts[@]}" \
            --training-receipt "${training_receipts[@]}" \
            --out "${replay}" >/dev/null
        if ! cmp -s \
            <(jq -S 'del(.created_at)' "${RAT_ANALYSIS}") \
            <(jq -S 'del(.created_at)' "${replay}"); then
            echo "existing analysis does not reproduce from the exact frozen inputs" >&2
            exit 1
        fi
        chmod a-w "${RAT_ANALYSIS}"
        echo "reusing immutable final analysis ${RAT_ANALYSIS}"
        return
    fi
    assert_preregistered_state
    python scripts/practice_utility/analyze_ratchet.py \
        --robustness-receipt "${eval_receipts[@]}" \
        --training-receipt "${training_receipts[@]}" \
        --out "${RAT_ANALYSIS}"
    chmod a-w "${RAT_ANALYSIS}"
    jq -e '
        .instrument_audit.passed == true
        and .instrument_audit.cell_count == 84
        and .claim_scope.paired_training_seeds == ["8600", "8601", "8602"]
        and (.preregistered_decision.status == "pass" or .preregistered_decision.status == "fail")
        and .preregistered_decision.superiority_claim_authorized == false
    ' "${RAT_ANALYSIS}" >/dev/null
}

preflight() {
    assert_preregistered_state
    validate_screen_inputs

    validate_training_receipt "${RAT_FIXED_8600_RECEIPT}" 8600 fixed
    validate_training_receipt "${RAT_FIXED_8601_RECEIPT}" 8601 fixed
    echo "ratchet confirmation preflight passed"
}

main() {
    preflight

    freeze_or_reuse_checkpoint \
        "${RAT_FIXED_8600_RECEIPT}" "${RAT_FIXED_8600_CONFIG}" 8600 fixed
    freeze_or_reuse_checkpoint \
        "${RAT_SCREEN_TRAINING}" "${RAT_SCREEN_RATCHET_CONFIG}" 8601 lucid_ratchet_rg
    freeze_or_reuse_checkpoint \
        "${RAT_FIXED_8601_RECEIPT}" "${RAT_FIXED_8601_CONFIG}" 8601 fixed

    run_or_reuse_training 8600 lucid_ratchet_rg "${RAT_TRAIN_R8600}"
    local r8600_train r8600_config
    r8600_train="$(single_receipt "${RAT_TRAIN_R8600}" 'curriculum_comparison_ne1024_*.json')"
    r8600_config="$(config_for_training_receipt "${r8600_train}" 8600 lucid_ratchet_rg)"
    freeze_or_reuse_checkpoint "${r8600_train}" "${r8600_config}" 8600 lucid_ratchet_rg

    run_or_reuse_training 8602 fixed "${RAT_TRAIN_F8602}"
    local f8602_train f8602_config
    f8602_train="$(single_receipt "${RAT_TRAIN_F8602}" 'curriculum_comparison_ne1024_*.json')"
    f8602_config="$(config_for_training_receipt "${f8602_train}" 8602 fixed)"
    freeze_or_reuse_checkpoint "${f8602_train}" "${f8602_config}" 8602 fixed

    run_or_reuse_training 8602 lucid_ratchet_rg "${RAT_TRAIN_R8602}"
    local r8602_train r8602_config
    r8602_train="$(single_receipt "${RAT_TRAIN_R8602}" 'curriculum_comparison_ne1024_*.json')"
    r8602_config="$(config_for_training_receipt "${r8602_train}" 8602 lucid_ratchet_rg)"
    freeze_or_reuse_checkpoint "${r8602_train}" "${r8602_config}" 8602 lucid_ratchet_rg

    # Do not inspect capability until every preregistered training cell is
    # complete and frozen.  This prevents operational optional stopping.
    run_or_reuse_evaluation \
        "${r8600_train}" "${r8600_config}" 8600 lucid_ratchet_rg 8700 "${RAT_EVAL_R8600}"
    run_or_reuse_evaluation \
        "${RAT_FIXED_8600_RECEIPT}" "${RAT_FIXED_8600_CONFIG}" 8600 fixed 8700 "${RAT_EVAL_F8600}"
    run_or_reuse_evaluation \
        "${f8602_train}" "${f8602_config}" 8602 fixed 8702 "${RAT_EVAL_F8602}"
    run_or_reuse_evaluation \
        "${r8602_train}" "${r8602_config}" 8602 lucid_ratchet_rg 8702 "${RAT_EVAL_R8602}"

    run_analysis "${r8600_train}" "${r8602_train}"
    mkdir -p "${RAT_ROOT}/.complete"
    jq '{claim_scope, preregistered_decision, ratchet_vs_fixed, mechanism}' "${RAT_ANALYSIS}"
}

source "${RAT_ENV}"
export LUCID_GPU_WAIT_SECONDS=7200
cd "${RAT_REPO}"
if [[ "${1:-}" == "--preflight-only" ]]; then
    preflight
    exit 0
fi
main "$@"
