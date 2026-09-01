#!/usr/bin/env bash
# Run the future preregistered Tier-2 support-extension screen.
#
# This file is intentionally inert today. Activation requires an externally
# supplied SHA-256 for a future immutable preregistration. Each GPU cell has an
# independent receipt directory and a one-shot .started marker. A marker with
# no valid complete receipt is evidence of an interrupted cell: preserve it,
# file a deviation, and do not resume or silently retrain.
set -euo pipefail

readonly SUP_REPO="${LUCID_SUPPORT_SCREEN_REPO:-/home/linjiw/lucid-support-screen}"
# The bootstrap is deliberately not environment-overridable.  It is executed
# before Python is available, so both its absolute path and bytes are frozen in
# the preregistration and checked by the system shell before it is sourced.
readonly SUP_ENV="/home/linjiw/lucid/env/lucid_env.sh"
readonly SUP_DRIVER_PATH="$(/usr/bin/readlink -f "${BASH_SOURCE[0]}")"
# Claim-bearing output locations are part of the hashed driver contract.  An
# environment override would let a launch silently select a different set of
# reusable receipts and one-shot markers than the future preregistration names.
readonly SUP_ROOT="/home/linjiw/lucid-sonic/manifests/tier2_support_screen"
readonly SUP_PREREG="${LUCID_SUPPORT_SCREEN_PREREG:-/home/linjiw/lucid-sonic/manifests/lucid_tier2_support_screen_preregistration.json}"
readonly SUP_FREEZE_ROOT="${SUP_ROOT}/frozen_checkpoints"
readonly SUP_EVAL_ARTIFACT_ROOT="/home/linjiw/lucid-sonic/artifacts/tier2_support_screen_eval"
readonly SUP_EVAL_LOG_ROOT="/home/linjiw/lucid-sonic/outputs/tier2_support_screen_eval"
readonly SUP_ANALYSIS="${SUP_ROOT}/lucid_tier2_support_screen_analysis.json"

readonly SUP_TRAIN_FRESH="${SUP_ROOT}/training/fresh_fixed"
readonly SUP_TRAIN_FIXED150="${SUP_ROOT}/training/fixed_150"
readonly SUP_TRAIN_FIXEDU150="${SUP_ROOT}/training/fixed_u150"
readonly SUP_EVAL_HISTORICAL="${SUP_ROOT}/evaluation/historical_fixed"
readonly SUP_EVAL_FRESH="${SUP_ROOT}/evaluation/fresh_fixed"
readonly SUP_EVAL_FIXED150="${SUP_ROOT}/evaluation/fixed_150"
readonly SUP_EVAL_FIXEDU150="${SUP_ROOT}/evaluation/fixed_u150"

readonly -a SUP_PRESETS=(
    phys_000 phys_025 phys_050 phys_075 phys_100
    phys_125 phys_150 phys_175 phys_200
    lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms lat_60ms
)
readonly -a SUP_REQUIRED_CODE_FILES=(
    gear_sonic/research/practice_utility/dr_controller.py
    gear_sonic/research/practice_utility/dr_curriculum.py
    gear_sonic/research/practice_utility/dr_scaling.py
    gear_sonic/research/practice_utility/events_reset_safe.py
    gear_sonic/research/practice_utility/tace.py
    scripts/practice_utility/run_curriculum_comparison.py
    scripts/practice_utility/train_with_delay.py
    scripts/practice_utility/run_curriculum_robustness_eval.py
    scripts/practice_utility/eval_with_delay.py
    scripts/practice_utility/freeze_training_checkpoint.py
    scripts/practice_utility/analyze_support_screen.py
    scripts/practice_utility/run_support_screen.sh
)
readonly -a SUP_REQUIRED_INPUTS=(
    panel_receipt
    h_r2_analysis
    motion
    encoder
    historical_fixed_training
    historical_fixed_freeze_manifest
    historical_fixed_checkpoint
    historical_fixed_config
    historical_fixed_launcher
    environment_bootstrap
)

# Populated only after the future preregistration and every one of its hashes
# have passed. Keeping these unset before preflight prevents an accidental
# fallback to stale paths.
SUP_PANEL=""
SUP_PANEL_SHA=""
SUP_H_R2=""
SUP_H_R2_SHA=""
SUP_MOTION_FILE=""
SUP_MOTION_DIR=""
SUP_ENCODER=""
SUP_HISTORICAL_TRAINING=""
SUP_HISTORICAL_FREEZE=""
SUP_HISTORICAL_CHECKPOINT=""
SUP_HISTORICAL_CONFIG=""
SUP_EVALUATOR_SHA=""

die() {
    echo "$*" >&2
    exit 1
}

assert_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || die "invalid expected SHA-256 for ${path}"
    [[ -f "${path}" ]] || die "SHA-pinned file is missing: ${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] \
        || die "hash mismatch: ${path}: expected ${expected}, got ${actual}"
}

assert_no_write_bits() {
    local path="$1"
    local mode
    mode="$(stat -c '%a' "${path}")"
    if (( (8#${mode}) & 8#222 )); then
        die "immutable artifact regained write bits: ${path}"
    fi
}

assert_regular_single_link() {
    local path="$1"
    [[ -f "${path}" && ! -L "${path}" ]] \
        || die "claim-bearing artifact is not a regular file: ${path}"
    [[ "$(stat -c '%h' "${path}")" == "1" ]] \
        || die "claim-bearing artifact has multiple hard links: ${path}"
}

assert_unaliased_directory() {
    local path="$1"
    [[ -d "${path}" && ! -L "${path}" ]] \
        || die "claim-bearing output directory is missing or is a symlink: ${path}"
    [[ "$(readlink -f "${path}")" == "${path}" ]] \
        || die "claim-bearing output directory has a symlinked path component: ${path}"
}

frozen_input_path() {
    local key="$1"
    jq -er --arg key "${key}" '.frozen_inputs[$key].path' "${SUP_PREREG}"
}

assert_preregistration_schema() {
    jq -e \
        --arg repo "${SUP_REPO}" \
        --arg driver "${SUP_DRIVER_PATH}" \
        --arg environment "${SUP_ENV}" '
        .kind == "lucid_tier2_support_screen_preregistration"
        and .schema_version == 1
        and .frozen == true
        and .written_before_gpu == true
        and .code_state.worktree == $repo
        and .code_state.clean_detached_worktree_required == true
        and (.code_state.git_sha | type == "string" and test("^[0-9a-f]{40}$"))
        and (.code_state.file_sha256 | type == "object" and length > 0)
        and (.frozen_inputs | type == "object" and length > 0)
        and .execution.driver.path == $driver
        and .execution.driver.sha256
            == .code_state.file_sha256["scripts/practice_utility/run_support_screen.sh"]
        and .environment.bootstrap.path == $environment
        and .environment.bootstrap.sha256 == .frozen_inputs.environment_bootstrap.sha256
        and .environment.bootstrap.path == .frozen_inputs.environment_bootstrap.path
        and (.environment.bootstrap.sha256 | type == "string" and test("^[0-9a-f]{64}$"))
        and .design.training.from_scratch == true
        and .design.training.seed == 8600
        and .design.training.num_envs == 1024
        and .design.training.iterations == 8000
        and .design.training.warmup_iterations == 10
        and .design.training.order == ["fresh_fixed", "fixed_150", "fixed_u150"]
        and .design.training.role_to_mode == {
            "fresh_fixed": "fixed",
            "fixed_150": "fixed_150",
            "fixed_u150": "fixed_u150"
        }
        and .design.training.max_delay_capacity_steps == 12
        and .design.training.resume_allowed == false
        and .design.evaluation.num_envs == 512
        and .design.evaluation.checkpoint_seed == 8600
        and .design.evaluation.evaluation_seed == 8700
        and .design.evaluation.roles == [
            "historical_fixed", "fresh_fixed", "fixed_150", "fixed_u150"
        ]
        and .design.evaluation.presets == [
            "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
            "phys_125", "phys_150", "phys_175", "phys_200",
            "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms", "lat_60ms"
        ]
        and .design.evaluation.total_cells == 60
        and (.evaluation.evaluator_sha256 | type == "string"
             and test("^[0-9a-f]{64}$"))
        and (.historical_fixed.training_receipt_kind == "lucid_historical_training_cell_bridge"
             or .historical_fixed.training_receipt_kind == "lucid_three_arm_training_comparison")
        and (.historical_fixed.git_sha | type == "string" and test("^[0-9a-f]{40}$"))
        and (.historical_fixed.git_status_short | type == "array")
        and (.historical_fixed.launcher_sha256 | type == "string"
             and test("^[0-9a-f]{64}$"))
        and .historical_fixed.launcher_sha256
            == .frozen_inputs.historical_fixed_launcher.sha256
        and .analysis.script == "scripts/practice_utility/analyze_support_screen.py"
        and .analysis.screening_only == true
        and .analysis.directional_claim_authorized == false
        and .analysis.superiority_claim_authorized == false
    ' "${SUP_PREREG}" >/dev/null \
        || die "future Tier-2 preregistration does not satisfy the frozen supervisor schema"
}

assert_h_r2_passed() {
    local h_r2="$1"
    jq -e '
        .kind == "lucid_ratchet_analysis"
        and .instrument_audit.passed == true
        and .claim_scope.status == "three_seed_decision"
        and .claim_scope.noninferiority_decision_eligible == true
        and .preregistered_decision.status == "pass"
        and .preregistered_decision.paired_training_seeds == ["8600", "8601", "8602"]
        and .preregistered_decision.mechanism_pass == true
        and .preregistered_decision.capability_components_pass == true
        and .preregistered_decision.noninferiority_claim_authorized == true
        and .preregistered_decision.superiority_claim_authorized == false
        and .mechanism.summary.all_available_seeds_pass == true
    ' "${h_r2}" >/dev/null || die "Tier-2 remains gated: the frozen H_R2 analysis did not pass"
}

load_preregistered_paths() {
    SUP_PANEL="$(frozen_input_path panel_receipt)"
    SUP_PANEL_SHA="$(jq -er '.frozen_inputs.panel_receipt.sha256' "${SUP_PREREG}")"
    SUP_H_R2="$(frozen_input_path h_r2_analysis)"
    SUP_H_R2_SHA="$(jq -er '.frozen_inputs.h_r2_analysis.sha256' "${SUP_PREREG}")"
    SUP_MOTION_FILE="$(frozen_input_path motion)"
    SUP_MOTION_DIR="$(dirname "${SUP_MOTION_FILE}")"
    SUP_ENCODER="$(frozen_input_path encoder)"
    SUP_HISTORICAL_TRAINING="$(frozen_input_path historical_fixed_training)"
    SUP_HISTORICAL_FREEZE="$(frozen_input_path historical_fixed_freeze_manifest)"
    SUP_HISTORICAL_CHECKPOINT="$(frozen_input_path historical_fixed_checkpoint)"
    SUP_HISTORICAL_CONFIG="$(frozen_input_path historical_fixed_config)"
    SUP_EVALUATOR_SHA="$(jq -er '.evaluation.evaluator_sha256' "${SUP_PREREG}")"
}

assert_preregistered_state() {
    : "${LUCID_SUPPORT_SCREEN_PREREG_SHA256:?set the future frozen Tier-2 preregistration SHA-256}"
    assert_sha256 "${SUP_PREREG}" "${LUCID_SUPPORT_SCREEN_PREREG_SHA256}"
    assert_preregistration_schema

    local expected_git
    expected_git="$(jq -er '.code_state.git_sha' "${SUP_PREREG}")"
    [[ "$(git -C "${SUP_REPO}" rev-parse HEAD)" == "${expected_git}" ]] \
        || die "support-screen HEAD differs from preregistered commit ${expected_git}"
    if git -C "${SUP_REPO}" symbolic-ref -q HEAD >/dev/null; then
        die "support-screen worktree must be detached at the preregistered commit"
    fi
    git -C "${SUP_REPO}" diff --quiet \
        || die "support-screen worktree has unstaged tracked changes: ${SUP_REPO}"
    git -C "${SUP_REPO}" diff --cached --quiet \
        || die "support-screen worktree has staged changes: ${SUP_REPO}"
    [[ -z "$(git -C "${SUP_REPO}" status --porcelain --untracked-files=all)" ]] \
        || die "support-screen worktree is not clean: ${SUP_REPO}"

    local required relative expected key path
    for required in "${SUP_REQUIRED_CODE_FILES[@]}"; do
        jq -e --arg required "${required}" '.code_state.file_sha256 | has($required)' \
            "${SUP_PREREG}" >/dev/null \
            || die "preregistration does not pin required code file: ${required}"
    done
    while IFS=$'\t' read -r relative expected; do
        [[ "${relative}" != /* && "${relative}" != *".."* ]] \
            || die "unsafe preregistered code path: ${relative}"
        assert_sha256 "${SUP_REPO}/${relative}" "${expected}"
    done < <(
        jq -r '.code_state.file_sha256 | to_entries[] | [.key, .value] | @tsv' \
            "${SUP_PREREG}"
    )

    local expected_driver
    expected_driver="$(jq -er \
        '.code_state.file_sha256["scripts/practice_utility/run_support_screen.sh"]' \
        "${SUP_PREREG}")"
    [[ "${SUP_DRIVER_PATH}" == "$(readlink -f "${SUP_REPO}/scripts/practice_utility/run_support_screen.sh")" ]] \
        || die "the executed support-screen driver is not the preregistered worktree file"
    assert_sha256 "${SUP_DRIVER_PATH}" "${expected_driver}"

    for required in "${SUP_REQUIRED_INPUTS[@]}"; do
        jq -e --arg required "${required}" '.frozen_inputs | has($required)' \
            "${SUP_PREREG}" >/dev/null \
            || die "preregistration does not pin required input: ${required}"
    done
    while IFS=$'\t' read -r key path expected; do
        [[ -n "${key}" && -n "${path}" ]] || die "malformed frozen input entry"
        assert_sha256 "${path}" "${expected}"
    done < <(
        jq -r '.frozen_inputs | to_entries[] | [.key, .value.path, .value.sha256] | @tsv' \
            "${SUP_PREREG}"
    )

    load_preregistered_paths
    [[ "$(readlink -f "$(frozen_input_path environment_bootstrap)")" \
       == "$(readlink -f "${SUP_ENV}")" ]] \
        || die "environment bootstrap path differs from the frozen preregistration"
    assert_sha256 "${SUP_ENV}" \
        "$(jq -er '.frozen_inputs.environment_bootstrap.sha256' "${SUP_PREREG}")"
    [[ "${SUP_PANEL}" == "$(jq -er '.evaluation.panel_receipt' "${SUP_PREREG}")" ]] \
        || die "evaluation panel path differs from the frozen input"
    [[ "$(jq -er '.frozen_inputs.panel_receipt.sha256' "${SUP_PREREG}")" \
       == "$(jq -er '.evaluation.panel_sha256' "${SUP_PREREG}")" ]] \
        || die "evaluation panel SHA differs from the frozen input"
    [[ "${SUP_EVALUATOR_SHA}" \
       == "$(jq -er '.code_state.file_sha256["scripts/practice_utility/run_curriculum_robustness_eval.py"]' "${SUP_PREREG}")" ]] \
        || die "evaluator SHA is not the preregistered evaluator file SHA"
    assert_h_r2_passed "${SUP_H_R2}"
}

validate_live_data_contract() {
    python - "${SUP_PANEL}" "${SUP_MOTION_FILE}" \
        "$(jq -er '.frozen_inputs.motion.sha256' "${SUP_PREREG}")" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

panel_path = Path(sys.argv[1]).resolve()
motion_entry = Path(sys.argv[2])
motion_path = motion_entry.resolve()
motion_sha = sys.argv[3]


def fail(message: str) -> None:
    raise SystemExit(f"live panel/training-motion audit failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


try:
    panel = json.loads(panel_path.read_text())
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot load panel receipt: {error}")
if not isinstance(panel, dict):
    fail("panel receipt is not an object")
if panel.get("kind") != "lucid_replicate_panel" or panel.get("schema_version") != 1:
    fail("panel kind/schema differs")
if panel.get("replicates") != 512:
    fail("panel does not declare exactly 512 aliases")
if not motion_path.is_file() or sha256(motion_path) != motion_sha:
    fail("the frozen training motion is missing or its bytes changed")

training_entries = list(motion_entry.parent.iterdir())
if len(training_entries) != 1 or training_entries[0].resolve() != motion_path:
    fail("training motion directory is not an exact one-file directory")
if training_entries[0].suffix != ".pkl" or not training_entries[0].is_file():
    fail("the sole training motion entry is not a live .pkl file")

source_raw = panel.get("source_clip")
panel_dir_raw = panel.get("motion_file")
if not isinstance(source_raw, str) or not isinstance(panel_dir_raw, str):
    fail("panel source_clip or motion_file is missing")
source = Path(source_raw).resolve()
panel_dir = Path(panel_dir_raw).resolve()
if source != motion_path:
    fail("panel source path does not resolve to the frozen training motion")
if sha256(source) != motion_sha or sha256(source) != panel.get("source_clip_sha256"):
    fail("panel source bytes do not match both frozen hashes")
if source.stem != panel.get("motion_key"):
    fail("panel motion_key does not match the canonical source stem")
if not panel_dir.is_dir():
    fail("panel alias directory is missing")
entries = sorted(panel_dir.iterdir())
if len(entries) != 512:
    fail(f"panel alias directory has {len(entries)} entries, expected 512")
if not all(path.is_symlink() and path.suffix == ".pkl" and path.is_file() for path in entries):
    fail("panel directory is not exactly 512 live .pkl symlinks")
stems = sorted(path.stem for path in entries)
if len(set(stems)) != 512:
    fail("panel alias stems are not unique")
digest = hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest()
if digest != panel.get("alias_keys_sha256"):
    fail("live panel alias-stem digest differs from the receipt")
targets = {path.resolve(strict=True) for path in entries}
if targets != {motion_path}:
    fail("live panel aliases do not all target the frozen training motion")
PY
}

full_live_revalidation() {
    assert_preregistered_state
    assert_unaliased_directory "${SUP_ROOT}"
    validate_live_data_contract
    validate_historical_training
    validate_freeze_manifest \
        "${SUP_HISTORICAL_FREEZE}" "${SUP_HISTORICAL_TRAINING}" \
        "${SUP_HISTORICAL_CONFIG}" fixed historical_fixed
    assert_no_write_bits "${SUP_HISTORICAL_FREEZE}"
}

wait_for_idle_gpu() {
    local live
    while true; do
        live="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | sed '/^[[:space:]]*$/d')"
        if [[ -z "${live}" ]]; then
            return
        fi
        echo "waiting for compute GPU; live PIDs: ${live//$'\n'/,}" >&2
        sleep 30
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
        echo "expected exactly one ${pattern} in ${directory}; found ${#receipts[@]}" >&2
        return 1
    fi
    printf '%s\n' "${receipts[0]}"
}

validate_historical_training() {
    assert_regular_single_link "${SUP_HISTORICAL_TRAINING}"
    jq -e \
        --arg checkpoint "$(readlink -f "${SUP_HISTORICAL_CHECKPOINT}")" \
        --arg motion_dir "${SUP_MOTION_DIR}" \
        --arg kind "$(jq -er '.historical_fixed.training_receipt_kind' "${SUP_PREREG}")" \
        --arg git_sha "$(jq -er '.historical_fixed.git_sha' "${SUP_PREREG}")" \
        --arg launcher_sha "$(jq -er '.historical_fixed.launcher_sha256' "${SUP_PREREG}")" \
        --argjson git_status "$(jq -c '.historical_fixed.git_status_short' "${SUP_PREREG}")" '
        .kind == $kind
        and .schema_version == 1
        and .git_sha == $git_sha
        and .git_status_short == $git_status
        and .launcher_sha256 == $launcher_sha
        and (.verified | type == "array" and length > 0)
        and .config.checkpoint == null
        and .config.from_scratch == true
        and .config.num_envs == 1024
        and .config.iterations == 8000
        and .config.warmup_iterations == 10
        and .config.seeds == [8600]
        and .config.modes == ["fixed"]
        and .config.arm_order == [{"seed": 8600, "modes": ["fixed"]}]
        and .config.motion_file == $motion_dir
        and .config.smpl_motion_file == "dummy"
        and .config.event_preset == "tracking/lucid_curriculum"
        and .config.termination_thresholds == "default"
        and .config.consolidation_fraction == 0
        and .config.max_delay_steps == 8
        and .config.max_delay_ms == 40
        and .config.arms == {"fixed": ["fixed", 0, null]}
        and (.arms | type == "object" and length == 1)
        and ((.arms | keys) == (.runtime | keys))
        and ((.arms | keys) == (.commands | keys))
        and ([.arms[] | select(
            .seed == 8600 and .mode == "fixed"
            and .complete == true and .checkpoint_exported == true
            and .iterations_parsed == 8000 and .curriculum_rows == 8000
            and .consolidation_rows == 0 and .actuator_groups_swapped == 5
            and .checkpoint == $checkpoint
            and .arm_spec.curriculum_mode == "fixed"
            and .arm_spec.anchor_ratio == 0
            and .arm_spec.spread_strata == 1
            and .arm_spec.fixed_lambda == 1.0
            and .arm_spec.allow_extrapolation == false
        )] | length == 1)
        and ([.runtime[] | select(.exit_code == 0)] | length == 1)
    ' "${SUP_HISTORICAL_TRAINING}" >/dev/null \
        || die "historical fixed training bridge is not the frozen seed-8600 cell"
    assert_sha256 "${SUP_HISTORICAL_CHECKPOINT}" \
        "$(jq -er '.frozen_inputs.historical_fixed_checkpoint.sha256' "${SUP_PREREG}")"
    assert_sha256 "${SUP_HISTORICAL_CONFIG}" \
        "$(jq -er '.frozen_inputs.historical_fixed_config.sha256' "${SUP_PREREG}")"
    validate_training_command "${SUP_HISTORICAL_TRAINING}" historical_fixed fixed 8
    validate_fixed_curriculum \
        "$(jq -er '.arms[] | select(.seed == 8600 and .mode == "fixed") | .curriculum_path' \
            "${SUP_HISTORICAL_TRAINING}")" \
        historical_fixed
}

validate_training_command() {
    local receipt="$1"
    local role="$2"
    local mode="$3"
    local max_delay="$4"
    python - "${receipt}" "${role}" "${mode}" "${max_delay}" \
        "${SUP_MOTION_DIR}" "${SUP_ENCODER}" "${SUP_REPO}" <<'PY'
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1]).resolve()
role = sys.argv[2]
mode = sys.argv[3]
max_delay = sys.argv[4]
motion_dir = sys.argv[5]
encoder = sys.argv[6]
repo = Path(sys.argv[7]).resolve()
receipt = json.loads(receipt_path.read_text())


def fail(message: str) -> None:
    raise SystemExit(f"{role} training command audit failed: {message}")


arms = receipt.get("arms")
commands = receipt.get("commands")
if not isinstance(arms, dict) or len(arms) != 1 or not isinstance(commands, dict):
    fail("receipt does not contain one arm and one command")
branch_id, arm = next(iter(arms.items()))
if set(commands) != {branch_id} or arm.get("branch_id") != branch_id:
    fail("arm/command/branch identifiers differ")
command = commands[branch_id]
if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
    fail("command is not an argv string array")
if len(command) < 6 or command[2:5] != ["--max-delay", max_delay, "--"]:
    fail("delay-wrapper boundary is not exact")
if role != "historical_fixed":
    if Path(command[0]).resolve() != Path(sys.executable).resolve():
        fail("training Python executable differs from the pinned bootstrap environment")
    if Path(command[1]).resolve() != repo / "scripts/practice_utility/train_with_delay.py":
        fail("training wrapper is not the preregistered worktree file")
required = {
    "+exp=manager/universal_token/all_modes/sonic_release",
    "num_envs=1024",
    "headless=true",
    "seed=8600",
    "manager_env/events=tracking/lucid_curriculum",
    "++algo.config.num_learning_iterations=8000",
    "++algo.config.save_interval=100000",
    f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_dir}",
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
    f"++callbacks.practice_observer.encoder_path={encoder}",
    "++callbacks.lucid_curriculum.mode=fixed",
    "++callbacks.lucid_curriculum.initial_lambda=0.0",
    f"++callbacks.lucid_curriculum.fixed_lambda={'1.0' if role in {'historical_fixed', 'fresh_fixed'} else '1.5'}",
    "++callbacks.lucid_curriculum.warmup_iterations=10",
    "++callbacks.practice_capsule.horizons.final=8000",
}
missing = sorted(required - set(command))
if missing:
    fail(f"required argv entries are missing: {missing}")


def require_unique(prefix: str, expected: str) -> None:
    matches = [token for token in command if token.startswith(prefix)]
    if matches != [expected]:
        fail(f"argv field {prefix} is missing, duplicated, or overridden: {matches}")


for prefix, expected in (
    ("+exp=", "+exp=manager/universal_token/all_modes/sonic_release"),
    ("num_envs=", "num_envs=1024"),
    ("seed=", "seed=8600"),
    (
        "++algo.config.num_learning_iterations=",
        "++algo.config.num_learning_iterations=8000",
    ),
    (
        "++manager_env.commands.motion.motion_lib_cfg.motion_file=",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_dir}",
    ),
    (
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
    ),
    (
        "++callbacks.practice_observer.encoder_path=",
        f"++callbacks.practice_observer.encoder_path={encoder}",
    ),
    ("++callbacks.lucid_curriculum.mode=", "++callbacks.lucid_curriculum.mode=fixed"),
    (
        "++callbacks.lucid_curriculum.fixed_lambda=",
        f"++callbacks.lucid_curriculum.fixed_lambda={'1.0' if role in {'historical_fixed', 'fresh_fixed'} else '1.5'}",
    ),
    (
        "++callbacks.lucid_curriculum.warmup_iterations=",
        "++callbacks.lucid_curriculum.warmup_iterations=10",
    ),
):
    require_unique(prefix, expected)
if any(token.startswith("checkpoint=") for token in command):
    fail("from-scratch command unexpectedly contains a checkpoint")
if role in {"fixed_150", "fixed_u150"}:
    if "++callbacks.lucid_curriculum.allow_extrapolation=true" not in command:
        fail("extrapolation flag is missing")
elif any("allow_extrapolation" in token for token in command):
    fail("non-extrapolating fixed command contains extrapolation")
if role == "fixed_u150":
    for token in (
        "++callbacks.lucid_curriculum.spread_strata=8",
        "++callbacks.lucid_curriculum.anchor_seed=8600",
        "++callbacks.lucid_curriculum.stratum_sizes=[37,37,37,37,36,36,36,768]",
    ):
        if token not in command:
            fail(f"fixed_u150 command is missing {token}")
PY
}

validate_fixed_curriculum() {
    local curriculum="$1"
    local role="$2"
    python - "${curriculum}" "${role}" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
role = sys.argv[2]
if role in {"historical_fixed", "fresh_fixed"}:
    expected_lambda = 1.0
    expected_extrapolation = False
    expected_clamp = None
elif role in {"fixed_150", "fixed_u150"}:
    expected_lambda = 1.5
    expected_extrapolation = True
    expected_clamp = ["physics_material"]
else:
    raise SystemExit(f"unknown fixed curriculum role: {role}")

terms = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
rows = 0
warmup_rows = 0
post_warmup_rows = 0

def fail(message: str) -> None:
    raise SystemExit(f"{role} raw curriculum audit failed at row {rows}: {message}")

with path.open() as stream:
    for rows, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSON: {error}")
        if not isinstance(row, dict) or row.get("global_step") != rows:
            fail("global_step is missing or non-contiguous")
        if row.get("mode") != "fixed":
            fail("mode is not fixed")
        if not math.isclose(
            float(row.get("lambda", -1)), expected_lambda, rel_tol=0.0, abs_tol=1e-12
        ):
            fail(f"lambda is not fixed at {expected_lambda}")
        if row.get("consolidation", False) is not False:
            fail("a consolidation row replaced the frozen distribution")
        if set(row.get("scalable_terms") or []) != terms:
            fail("six scalable terms differ")
        if rows <= 10:
            warmup_rows += 1
            if row.get("warmup_hold") is not True:
                fail("one of the first 10 rows is not a warmup hold")
            if row.get("allow_extrapolation") is not None or row.get("physical_clamp") is not None:
                fail("warmup row contains unsupported extrapolation telemetry")
            if row.get("tace") is not None:
                fail("warmup row unexpectedly contains TACE telemetry")
            continue
        post_warmup_rows += 1
        if row.get("warmup_hold", False) is not False:
            fail("post-warmup row is marked as warmup")
        if row.get("allow_extrapolation", False) is not expected_extrapolation:
            fail("post-warmup extrapolation state differs")
        if row.get("physical_clamp") != expected_clamp:
            fail("post-warmup physical clamp state differs")
        if role != "fixed_u150" and row.get("tace") is not None:
            fail("unstratified fixed arm unexpectedly contains TACE telemetry")

if rows != 8000:
    fail(f"expected 8000 total rows, observed {rows}")
if warmup_rows != 10 or post_warmup_rows != 7990:
    fail(f"expected 10 warmup + 7990 post-warmup rows, observed {warmup_rows} + {post_warmup_rows}")
PY
}

validate_fixedu150_curriculum() {
    local curriculum="$1"
    local training_receipt="$2"
    python - "${curriculum}" "${training_receipt}" <<'PY'
import json
import math
from pathlib import Path
import sys

from gear_sonic.research.practice_utility import dr_scaling as DS

path = Path(sys.argv[1])
receipt = json.loads(Path(sys.argv[2]).read_text())
terms = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
sizes = [37, 37, 37, 37, 36, 36, 36, 768]
lambdas = [0.1875, 0.375, 0.5625, 0.75, 0.9375, 1.125, 1.3125, 1.5]
delay_ranges = [[0.0, 1.5], [0.0, 3.0], [0.0, 4.5], [0.0, 6.0],
                [0.0, 7.5], [0.0, 9.0], [0.0, 10.5], None]
rows = 0
post_warmup_tace_rows = 0
final_dispatch = None
final_tace = None
anchor_params_by_term = None

def fail(message: str) -> None:
    raise SystemExit(f"fixed_u150 raw curriculum audit failed at row {rows}: {message}")

def equivalent(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right

with path.open() as stream:
    for rows, line in enumerate(stream, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSON: {error}")
        if not isinstance(row, dict) or row.get("global_step") != rows:
            fail("global_step is missing or non-contiguous")
        if rows <= 10:
            if row.get("tace") is not None:
                fail("warmup row unexpectedly contains TACE telemetry")
            continue
        post_warmup_tace_rows += 1
        if not math.isclose(float(row.get("lambda", -1)), 1.5, rel_tol=0.0, abs_tol=1e-12):
            fail("applied lambda is not 1.5")
        if row.get("allow_extrapolation") is not True:
            fail("allow_extrapolation is not true")
        if row.get("physical_clamp") != ["physics_material"]:
            fail("physical clamp evidence differs")
        if set(row.get("scalable_terms") or []) != terms:
            fail("six scalable terms differ")
        tace = row.get("tace")
        if not isinstance(tace, dict):
            fail("post-warmup TACE telemetry is absent")
        if (
            tace.get("num_anchor") != 0
            or tace.get("num_focus") != 1024
            or not math.isclose(float(tace.get("anchor_ratio", math.nan)), 0.0,
                                rel_tol=0.0, abs_tol=1e-12)
            or tace.get("num_strata") != 8
            or tace.get("stratum_sizes") != sizes
            or tace.get("stratum_lambdas") != lambdas
            or tace.get("consolidating") is not False
        ):
            fail("cohort sizes, applied lambdas, or consolidation state differ")
        dispatch = tace.get("dispatch")
        if not isinstance(dispatch, dict) or set(dispatch) != terms:
            fail("dispatcher does not contain exactly all six terms")
        final_dispatch = dispatch
        final_tace = tace
        current_anchor_params = {}
        for term in sorted(terms):
            telemetry = dispatch.get(term)
            if not isinstance(telemetry, dict) or telemetry.get("term") != term:
                fail(f"{term} dispatcher identity differs")
            params = telemetry.get("stratum_params")
            if (
                telemetry.get("num_strata") != 8
                or not isinstance(params, list)
                or len(params) != 8
                or params[-1] is not None
                or not all(isinstance(value, dict) for value in params[:-1])
            ):
                fail(f"{term} did not recompute all eight stratum parameters")
            anchor_params = telemetry.get("anchor_params")
            if not isinstance(anchor_params, dict) or not anchor_params:
                fail(f"{term} anchor_params baseline is absent")
            if not set(anchor_params).issubset(DS.RANGE_NOMINALS):
                fail(f"{term} anchor_params contains unsupported range fields")
            current_anchor_params[term] = anchor_params
            for index, applied_lambda in enumerate(lambdas[:-1]):
                expected_params = DS.scaled_term_params(
                    anchor_params, applied_lambda, allow_extrapolation=True
                )
                expected_params, _ = DS.clamp_params_physical(expected_params)
                if not equivalent(params[index], expected_params):
                    fail(f"{term} stratum {index} params do not recompute from anchor_params")
            counts = telemetry.get("env_counts")
            if not isinstance(counts, dict):
                fail(f"{term} dispatcher counts are absent")
            for key, count in counts.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    fail(f"{term} {key} counter is invalid")
        if anchor_params_by_term is None:
            anchor_params_by_term = current_anchor_params
        elif not equivalent(current_anchor_params, anchor_params_by_term):
            fail("dispatcher anchor_params changed during the run")
        observed_delay = [
            value.get("delay_range") if isinstance(value, dict) else None
            for value in dispatch["randomize_action_delay"]["stratum_params"]
        ]
        if observed_delay != delay_ranges:
            fail("per-stratum delay ranges differ")

if rows != 8000:
    fail(f"expected 8000 total rows, observed {rows}")
if post_warmup_tace_rows != 7990:
    fail(f"expected 7990 post-warmup TACE rows, observed {post_warmup_tace_rows}")
if not isinstance(final_dispatch, dict):
    fail("final TACE dispatcher is absent")
arms = receipt.get("arms")
matches = [
    arm
    for arm in (arms.values() if isinstance(arms, dict) else [])
    if isinstance(arm, dict) and arm.get("seed") == 8600 and arm.get("mode") == "fixed_u150"
]
if len(matches) != 1 or not equivalent(matches[0].get("tace_final"), final_tace):
    fail("training receipt tace_final does not equal the final raw curriculum row")
for term in sorted(terms):
    counts = final_dispatch[term]["env_counts"]
    for index in range(8):
        if counts[f"focus_s{index}"] <= 0:
            fail(f"final {term} focus_s{index} counter was never exercised")
PY
}

validate_training_receipt() {
    local receipt="$1"
    local role="$2"
    local mode="$3"
    assert_regular_single_link "${receipt}"
    local fixed_lambda allow_extrapolation spread expected_live_max anchor_seed clamp
    if [[ "${role}" == "fresh_fixed" ]]; then
        fixed_lambda="1.0"
        allow_extrapolation="false"
        spread="1"
        expected_live_max="8"
        anchor_seed="null"
        clamp="null"
    elif [[ "${role}" == "fixed_150" ]]; then
        fixed_lambda="1.5"
        allow_extrapolation="true"
        spread="1"
        expected_live_max="12"
        anchor_seed="null"
        clamp='["physics_material"]'
    elif [[ "${role}" == "fixed_u150" ]]; then
        fixed_lambda="1.5"
        allow_extrapolation="true"
        spread="8"
        expected_live_max="12"
        anchor_seed="8600"
        clamp='["physics_material"]'
    else
        die "unknown support-screen training role: ${role}"
    fi

    jq -e \
        --arg role "${role}" \
        --arg mode "${mode}" \
        --arg git_sha "$(jq -er '.code_state.git_sha' "${SUP_PREREG}")" \
        --arg launcher_sha "$(jq -er \
            '.code_state.file_sha256["scripts/practice_utility/run_curriculum_comparison.py"]' \
            "${SUP_PREREG}")" \
        --arg motion_dir "${SUP_MOTION_DIR}" \
        --argjson fixed_lambda "${fixed_lambda}" \
        --argjson allow_extrapolation "${allow_extrapolation}" \
        --argjson spread "${spread}" \
        --argjson expected_live_max "${expected_live_max}" \
        --argjson anchor_seed "${anchor_seed}" \
        --argjson clamp "${clamp}" '
        def terms: [
            "add_joint_default_pos", "base_com", "physics_material", "push_robot",
            "randomize_action_delay", "randomize_rigid_body_mass"
        ];
        def sizes: [37, 37, 37, 37, 36, 36, 36, 768];
        def lambdas: [0.1875, 0.375, 0.5625, 0.75, 0.9375, 1.125, 1.3125, 1.5];
        .kind == "lucid_three_arm_training_comparison"
        and .schema_version == 1
        and .git_sha == $git_sha
        and .git_status_short == []
        and .launcher_sha256 == $launcher_sha
        and (.verified | type == "array" and length > 0)
        and .config.checkpoint == null
        and .config.from_scratch == true
        and .config.num_envs == 1024
        and .config.iterations == 8000
        and .config.warmup_iterations == 10
        and .config.seeds == [8600]
        and .config.modes == [$mode]
        and .config.arm_order == [{"seed": 8600, "modes": [$mode]}]
        and .config.event_preset == "tracking/lucid_curriculum"
        and .config.motion_file == $motion_dir
        and .config.smpl_motion_file == "dummy"
        and .config.arms == {($mode): ["fixed", 0, null]}
        and .config.termination_thresholds == "default"
        and .config.consolidation_fraction == 0
        and .config.max_delay_steps == 12
        and .config.max_delay_ms == 60
        and ((.arms | length) == 1)
        and ((.runtime | keys) == (.arms | keys))
        and ((.commands | keys) == (.arms | keys))
        and ([.arms[] | select(
            .seed == 8600 and .mode == $mode
            and .complete == true and .checkpoint_exported == true
            and .iterations_parsed == 8000 and .curriculum_rows == 8000
            and .consolidation_rows == 0 and .actuator_groups_swapped == 5
            and ((.scalable_terms | sort) == (terms | sort))
            and .final_lambda == $fixed_lambda
            and .arm_spec.curriculum_mode == "fixed"
            and .arm_spec.anchor_ratio == 0.0
            and .arm_spec.anchor_seed == $anchor_seed
            and .arm_spec.yoked_source == null
            and .arm_spec.yoked_cross_seed == false
            and .arm_spec.term_lambda_overrides == {}
            and .arm_spec.spread_strata == $spread
            and .arm_spec.return_guard == "absolute"
            and .arm_spec.monotonic == false
            and .arm_spec.fixed_lambda == $fixed_lambda
            and .arm_spec.allow_extrapolation == $allow_extrapolation
            and .arm_spec.physical_clamp == $clamp
            and .arm_spec.signal == "gap"
            and .arm_spec.margin == null
            and .arm_spec.term_lambda_caps == {}
            and .arm_spec.max_delay_steps == 12
            and .live_delay_final.action_delay_actuator_groups == 5
            and .live_delay_final.action_delay_num_lags == 5120
            and .live_delay_final.action_delay_min_steps == 0
            and .live_delay_final.action_delay_max_steps == $expected_live_max
            and (.live_delay_final.action_delay_nonzero_fraction | type == "number"
                 and . > 0 and . <= 1)
            and (.live_delay_final.action_delay_histogram | type == "array"
                 and length == ($expected_live_max + 1)
                 and all(.[]; type == "number" and floor == . and . >= 0)
                 and add == 5120)
            and ($role != "fixed_u150" or (
                .arm_spec.stratum_sizes == sizes
                and .arm_spec.stratum_lambdas == lambdas
                and .arm_spec.top_fraction == 0.75
                and .tace_final.num_anchor == 0
                and .tace_final.num_focus == 1024
                and .tace_final.num_strata == 8
                and .tace_final.stratum_sizes == sizes
                and .tace_final.stratum_lambdas == lambdas
                and .expand_contract.passed == true
                and .expand_contract.errors == []
            ))
            and ($role == "fixed_u150" or (
                .arm_spec.stratum_sizes == null
                and .arm_spec.stratum_lambdas == null
                and .arm_spec.top_fraction == null
                and .tace_final == null
            ))
        )] | length == 1)
        and ([.runtime[] | select(.exit_code == 0)] | length == 1)
    ' "${receipt}" >/dev/null || die "invalid ${role} training receipt: ${receipt}"

    local checkpoint curriculum
    checkpoint="$(jq -er --arg mode "${mode}" \
        '.arms[] | select(.seed == 8600 and .mode == $mode) | .checkpoint' "${receipt}")"
    curriculum="$(jq -er --arg mode "${mode}" \
        '.arms[] | select(.seed == 8600 and .mode == $mode) | .curriculum_path' "${receipt}")"
    [[ -f "${checkpoint}" && -f "${curriculum}" ]] \
        || die "${role} receipt does not resolve to checkpoint and curriculum files"
    validate_training_command "${receipt}" "${role}" "${mode}" 12
    validate_fixed_curriculum "${curriculum}" "${role}"
    if [[ "${role}" == "fixed_u150" ]]; then
        validate_fixedu150_curriculum "${curriculum}" "${receipt}"
    fi
}

run_or_reuse_training() {
    local role="$1"
    local mode="$2"
    local receipt_dir="$3"
    local receipt
    full_live_revalidation
    mkdir -p "${receipt_dir}"
    assert_unaliased_directory "${receipt_dir}"
    if receipt="$(single_receipt "${receipt_dir}" 'curriculum_comparison_ne1024_*.json' 2>/dev/null)"; then
        validate_training_receipt "${receipt}" "${role}" "${mode}"
        echo "reusing complete ${role} training receipt ${receipt}"
        return
    fi
    if find "${receipt_dir}" -maxdepth 1 -type f -name 'curriculum_comparison_ne1024_*.json' \
        -print -quit | grep -q .; then
        die "partial or ambiguous ${role} training receipts in ${receipt_dir}"
    fi
    if [[ -e "${receipt_dir}/.started" ]]; then
        die "${role} training was started without one valid complete receipt; no resume or automatic retry is allowed"
    fi
    wait_for_idle_gpu
    full_live_revalidation
    [[ ! -e "${receipt_dir}/.started" ]] \
        || die "${role} training marker appeared while waiting; refusing a concurrent launch"
    mkdir "${receipt_dir}/.started"
    python scripts/practice_utility/run_curriculum_comparison.py \
        --from-scratch \
        --num-envs 1024 \
        --iterations 8000 \
        --warmup-iterations 10 \
        --horizons 500 1000 2000 4000 6000 \
        --seeds 8600 \
        --modes "${mode}" \
        --termination-thresholds default \
        --wandb-project lucid-campaign \
        --motion-file "${SUP_MOTION_DIR}" \
        --smpl-motion-file dummy \
        --encoder "${SUP_ENCODER}" \
        --max-delay 12 \
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
    validate_training_receipt "${receipt}" "${role}" "${mode}"
}

config_for_training_receipt() {
    local receipt="$1"
    local mode="$2"
    local run_dir config
    run_dir="$(jq -er --arg mode "${mode}" \
        '.arms[] | select(.seed == 8600 and .mode == $mode) | .arm_spec.run_dir' "${receipt}")"
    if [[ "${run_dir}" = /* ]]; then
        config="${run_dir}/config.yaml"
    else
        config="${SUP_REPO}/${run_dir}/config.yaml"
    fi
    [[ -f "${config}" ]] || die "resolved training config is missing: ${config}"
    readlink -f "${config}"
}

validate_freeze_manifest() {
    local manifest="$1"
    local training_receipt="$2"
    local config="$3"
    local mode="$4"
    local role="$5"
    local expected_git
    expected_git="$(jq -er '.code_state.git_sha' "${SUP_PREREG}")"
    python - "${manifest}" "${training_receipt}" "${config}" "${mode}" \
        "${role}" "${expected_git}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import stat
import sys

manifest_entry = Path(sys.argv[1])
training_receipt = Path(sys.argv[2]).resolve(strict=True)
config = Path(sys.argv[3]).resolve(strict=True)
mode = sys.argv[4]
role = sys.argv[5]
expected_git = sys.argv[6]


def fail(message: str) -> None:
    raise SystemExit(f"{role} freeze-manifest audit failed: {message}")


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if manifest_entry.is_symlink() or not manifest_entry.is_file():
    fail("manifest is not a regular, non-symlink file")
manifest_lstat = manifest_entry.lstat()
if manifest_lstat.st_nlink != 1:
    fail(f"manifest has {manifest_lstat.st_nlink} hard links")
if stat.S_IMODE(manifest_lstat.st_mode) & 0o222:
    fail("manifest has write bits")
manifest_path = manifest_entry.resolve(strict=True)
manifest = load_object(manifest_path)
receipt = load_object(training_receipt)

expected_top = {
    "kind": "lucid_frozen_training_checkpoint",
    "schema_version": 1,
    "state": "frozen_for_evaluation",
    "evaluation_only": True,
    "seed": 8600,
    "mode": mode,
    "iterations": 8000,
    "resume_forbidden": True,
}
for key, expected in expected_top.items():
    if manifest.get(key) != expected:
        fail(f"{key} differs: {manifest.get(key)!r} != {expected!r}")
if not isinstance(manifest.get("verified"), list) or not manifest["verified"]:
    fail("verified evidence is absent")
if role != "historical_fixed":
    code = manifest.get("code") or {}
    if code.get("git_sha") != expected_git or code.get("git_status_short") != "":
        fail("new freeze was not created from the preregistered clean Git state")

arms = receipt.get("arms")
if not isinstance(arms, dict):
    fail("training receipt arms are absent")
matches = [
    arm
    for arm in arms.values()
    if isinstance(arm, dict) and arm.get("seed") == 8600 and arm.get("mode") == mode
]
if len(matches) != 1:
    fail(f"training receipt has {len(matches)} matching seed/mode arms")
arm = matches[0]
expected_paths = {
    "checkpoint": Path(arm["checkpoint"]).resolve(strict=True),
    "config": config,
    "curriculum": Path(arm["curriculum_path"]).resolve(strict=True),
    "final_capsule": Path(arm["capsule"]).resolve(strict=True),
    "training_receipt": training_receipt,
}
sha_re = re.compile(r"[0-9a-f]{64}")
for section_name, expected_path in expected_paths.items():
    section = manifest.get(section_name)
    if not isinstance(section, dict):
        fail(f"{section_name} section is absent")
    raw_path = section.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        fail(f"{section_name}.path is absent")
    observed_path = Path(raw_path).resolve(strict=True)
    if observed_path != expected_path:
        fail(f"{section_name} is not bound to the selected training arm")
    expected_sha = section.get("sha256")
    if not isinstance(expected_sha, str) or sha_re.fullmatch(expected_sha) is None:
        fail(f"{section_name}.sha256 is invalid")
    if sha256(observed_path) != expected_sha:
        fail(f"{section_name} bytes differ from its manifest hash")
    size = section.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size != observed_path.stat().st_size:
        fail(f"{section_name}.size_bytes differs")

checkpoint = expected_paths["checkpoint"]
checkpoint_mode = stat.S_IMODE(checkpoint.stat().st_mode)
if manifest["checkpoint"].get("read_only") is not True or checkpoint_mode & 0o222:
    fail("checkpoint is not read-only")
if manifest["checkpoint"].get("mode_octal") != oct(checkpoint_mode):
    fail("checkpoint mode_octal differs from the live file")
if manifest["curriculum"].get("rows") != 8000:
    fail("curriculum row count is not 8000")
PY
}

freeze_or_reuse_checkpoint() {
    local role="$1"
    local training_receipt="$2"
    local config="$3"
    local mode="$4"
    local manifest="${SUP_FREEZE_ROOT}/${role}.json"
    local created="false"
    assert_preregistered_state
    mkdir -p "${SUP_FREEZE_ROOT}"
    assert_unaliased_directory "${SUP_FREEZE_ROOT}"
    if [[ ! -e "${manifest}" ]]; then
        python scripts/practice_utility/freeze_training_checkpoint.py \
            --training-receipt "${training_receipt}" \
            --config "${config}" \
            --seed 8600 \
            --mode "${mode}" \
            --iterations 8000 \
            --make-read-only \
            --out "${manifest}" >/dev/null
        created="true"
    fi
    validate_freeze_manifest \
        "${manifest}" "${training_receipt}" "${config}" "${mode}" "${role}"
    if [[ "${created}" == "true" ]]; then
        chmod a-w "${manifest}"
    else
        assert_no_write_bits "${manifest}"
    fi
    printf '%s\n' "${manifest}"
}

validate_adjacent_config() {
    local checkpoint="$1"
    local config="$2"
    local expected_sha="$3"
    local source destination
    source="$(readlink -f "${config}")"
    destination="$(dirname "$(readlink -f "${checkpoint}")")/config.yaml"
    [[ -f "${destination}" ]] \
        || die "checkpoint-adjacent config is absent before the GPU marker: ${destination}"
    [[ "$(readlink -f "${destination}")" == "${source}" ]] \
        || die "checkpoint-adjacent config is not linked to the frozen config: ${destination}"
    assert_sha256 "${source}" "${expected_sha}"
    assert_sha256 "${destination}" "${expected_sha}"
}

prepare_adjacent_config() {
    local checkpoint="$1"
    local config="$2"
    local expected_sha="$3"
    local source destination parent
    source="$(readlink -f "${config}")"
    destination="$(dirname "$(readlink -f "${checkpoint}")")/config.yaml"
    parent="$(dirname "${destination}")"
    assert_sha256 "${source}" "${expected_sha}"
    if [[ ! -e "${destination}" && ! -L "${destination}" ]]; then
        [[ -w "${parent}" ]] \
            || die "checkpoint directory is not writable for pre-marker config installation: ${parent}"
        ln -s "${source}" "${destination}"
    fi
    validate_adjacent_config "${checkpoint}" "${source}" "${expected_sha}"
}

prepare_freeze_adjacent_config() {
    local manifest="$1"
    prepare_adjacent_config \
        "$(jq -er '.checkpoint.path' "${manifest}")" \
        "$(jq -er '.config.path' "${manifest}")" \
        "$(jq -er '.config.sha256' "${manifest}")"
}

validate_freeze_adjacent_config() {
    local manifest="$1"
    validate_adjacent_config \
        "$(jq -er '.checkpoint.path' "${manifest}")" \
        "$(jq -er '.config.path' "${manifest}")" \
        "$(jq -er '.config.sha256' "${manifest}")"
}

validate_evaluation_metrics() {
    local receipt="$1"
    local role="$2"
    local mode="$3"
    python - "${receipt}" "${role}" "${mode}" "${SUP_PANEL}" "${SUP_REPO}" \
        "${SUP_EVAL_ARTIFACT_ROOT}" "${SUP_EVAL_LOG_ROOT}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from gear_sonic.research.practice_utility import dr_scaling as DS

receipt_path = Path(sys.argv[1]).resolve()
role = sys.argv[2]
mode = sys.argv[3]
panel_path = Path(sys.argv[4]).resolve()
repo = Path(sys.argv[5]).resolve()
artifact_root = Path(sys.argv[6]).resolve()
log_root = Path(sys.argv[7]).resolve()
receipt = json.loads(receipt_path.read_text())
panel = json.loads(panel_path.read_text())
terms = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
latency_steps = {
    "lat_10ms": 2,
    "lat_20ms": 4,
    "lat_30ms": 6,
    "lat_40ms": 8,
    "lat_50ms": 10,
    "lat_60ms": 12,
}
physics_levels = {
    "phys_000": 0.0,
    "phys_025": 0.25,
    "phys_050": 0.5,
    "phys_075": 0.75,
    "phys_100": 1.0,
    "phys_125": 1.25,
    "phys_150": 1.5,
    "phys_175": 1.75,
    "phys_200": 2.0,
}
expected_presets = set(physics_levels) | set(latency_steps)
non_latency_terms = terms - {"randomize_action_delay"}

def fail(run_id: str, message: str) -> None:
    raise SystemExit(f"raw evaluation audit failed for {run_id}: {message}")

metrics_paths: set[Path] = set()
raw_by_preset: dict[str, dict] = {}
summary_by_preset: dict[str, dict] = {}
commands = receipt.get("commands")
if not isinstance(commands, dict) or set(commands) != set(receipt.get("runs", {})):
    raise SystemExit(f"{role} evaluation commands/runs keysets differ")
experiment_id = receipt.get("experiment_id")
if not isinstance(experiment_id, str) or re.fullmatch(
    r"curriculum_robustness_ne512_[0-9]{8}_[0-9]{6}", experiment_id
) is None:
    raise SystemExit(f"{role} evaluation experiment_id differs from the frozen launcher schema")
for run_id, run in receipt.get("runs", {}).items():
    preset = run.get("preset")
    if preset not in expected_presets or preset in raw_by_preset:
        fail(run_id, f"unexpected or duplicate preset {preset!r}")
    expected_run_id = f"{experiment_id}_s8600_{mode}_{preset}"
    if run_id != expected_run_id:
        fail(run_id, f"branch id differs from {expected_run_id}")
    expected_output = artifact_root / experiment_id / "seed_8600" / mode / preset
    expected_metrics = expected_output / "metrics_eval.json"
    expected_log = log_root / f"{run_id}.log"
    command = commands.get(run_id)
    if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
        fail(run_id, "command is not an argv string array")
    if (
        len(command) < 6
        or Path(command[0]).resolve() != Path(sys.executable).resolve()
        or Path(command[1]).resolve()
        != repo / "scripts/practice_utility/eval_with_delay.py"
        or command[2:5] != ["--max-delay", "12", "--"]
    ):
        fail(run_id, "evaluator wrapper/interpreter boundary differs")
    expected_event = (
        "tracking/lucid_curriculum" if preset in physics_levels else "tracking/lucid_eval_clean"
    )
    required_command = {
        f"checkpoint={run.get('checkpoint')}",
        "+num_envs=512",
        "+headless=true",
        "+use_wandb=false",
        "+seed=8700",
        f"+manager_env/events={expected_event}",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={Path(panel['motion_file']).resolve()}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
        "++callbacks.practice_eval.eval_only=true",
        f"++callbacks.practice_eval.output_dir={expected_output}",
        f"++callbacks.practice_eval.preset_id={preset}",
        f"++callbacks.practice_eval.branch_id={run_id}",
    }
    if not required_command.issubset(command):
        fail(run_id, "command does not encode the exact checkpoint/panel/preset cell")

    def require_unique(prefix: str, expected: str) -> None:
        matches = [token for token in command if token.startswith(prefix)]
        if matches != [expected]:
            fail(run_id, f"argv field {prefix} is missing, duplicated, or overridden: {matches}")

    for prefix, expected in (
        ("checkpoint=", f"checkpoint={run.get('checkpoint')}"),
        ("+num_envs=", "+num_envs=512"),
        ("+seed=", "+seed=8700"),
        ("+manager_env/events=", f"+manager_env/events={expected_event}"),
        (
            "++manager_env.commands.motion.motion_lib_cfg.motion_file=",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={Path(panel['motion_file']).resolve()}",
        ),
        (
            "++callbacks.practice_eval.output_dir=",
            f"++callbacks.practice_eval.output_dir={expected_output}",
        ),
        (
            "++callbacks.practice_eval.preset_id=",
            f"++callbacks.practice_eval.preset_id={preset}",
        ),
        (
            "++callbacks.practice_eval.branch_id=",
            f"++callbacks.practice_eval.branch_id={run_id}",
        ),
    ):
        require_unique(prefix, expected)
    if preset in physics_levels:
        require_unique(
            "++callbacks.practice_eval.non_latency_dr_scale=",
            f"++callbacks.practice_eval.non_latency_dr_scale={physics_levels[preset]}",
        )
        require_unique(
            "++callbacks.practice_eval.fixed_latency_steps=",
            "++callbacks.practice_eval.fixed_latency_steps=0",
        )
    else:
        if any(
            token.startswith("++callbacks.practice_eval.non_latency_dr_scale=")
            for token in command
        ):
            fail(run_id, "latency-cell command contains a non-latency scale override")
        require_unique(
            "++callbacks.practice_eval.fixed_latency_steps=",
            f"++callbacks.practice_eval.fixed_latency_steps={latency_steps[preset]}",
        )

    metrics_entry = Path(run.get("metrics_path", ""))
    if not metrics_entry.is_absolute():
        metrics_entry = receipt_path.parent / metrics_entry
    if metrics_entry.is_symlink():
        fail(run_id, "metrics_path is a symlink")
    metrics_path = metrics_entry.resolve()
    if metrics_path != expected_metrics.resolve():
        fail(run_id, "metrics_path is outside its exact role/preset artifact directory")
    if (
        not metrics_path.is_file()
        or metrics_path.stat().st_nlink != 1
        or metrics_path in metrics_paths
    ):
        fail(run_id, "metrics_path is missing, linked, or reused")
    metrics_paths.add(metrics_path)
    log_path_raw = run.get("log_path")
    if not isinstance(log_path_raw, str) or not log_path_raw:
        fail(run_id, "log_path is absent")
    log_entry = Path(log_path_raw)
    if log_entry.is_symlink():
        fail(run_id, "log_path is a symlink")
    log_path = log_entry.resolve()
    if (
        log_path != expected_log.resolve()
        or not log_path.is_file()
        or log_path.stat().st_nlink != 1
    ):
        fail(run_id, "log_path is outside its exact branch path or is linked")
    metrics = json.loads(metrics_path.read_text())
    arrays = metrics.get("eval/all_metrics_dict")
    if not isinstance(arrays, dict):
        fail(run_id, "eval/all_metrics_dict is absent")
    motion_keys = arrays.get("motion_keys")
    terminated = arrays.get("terminated")
    progress = arrays.get("progress")
    if (
        not isinstance(motion_keys, list)
        or len(motion_keys) != 512
        or len(set(motion_keys)) != 512
        or not isinstance(terminated, list)
        or len(terminated) != 512
        or not all(isinstance(value, bool) for value in terminated)
        or not isinstance(progress, list)
        or len(progress) != 512
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            for value in progress
        )
    ):
        fail(run_id, "the three scored episode arrays are not exact length-512 arrays")
    motion_digest = hashlib.sha256(
        ("\n".join(sorted(motion_keys)) + "\n").encode()
    ).hexdigest()
    if motion_digest != panel.get("alias_keys_sha256"):
        fail(run_id, "raw motion keys do not match the frozen live panel digest")
    failed_idxes = metrics.get("failed_idxes")
    expected_failed = [index for index, value in enumerate(terminated) if value]
    if failed_idxes != expected_failed:
        fail(run_id, "failed_idxes does not equal the terminated-array indices")
    failed_keys = metrics.get("failed_keys")
    if failed_keys != [motion_keys[index] for index in expected_failed]:
        fail(run_id, "failed_keys does not reconcile with failed_idxes and motion_keys")

    success = 1.0 - len(expected_failed) / 512.0
    progress_rate = sum(float(value) for value in progress) / 512.0
    for index, (terminated_value, progress_value) in enumerate(zip(terminated, progress)):
        if (not terminated_value) != (float(progress_value) >= 1.0):
            fail(run_id, f"episode {index} termination/progress disagree")
    raw_success = metrics.get("eval/success/success_rate")
    raw_progress = metrics.get("eval/success/progress_rate")
    summary = run.get("summary") or {}
    for label, observed, expected in (
        ("raw success", raw_success, success),
        ("summary success", summary.get("success_rate"), success),
        ("raw progress", raw_progress, progress_rate),
        ("summary progress", summary.get("progress_rate"), progress_rate),
    ):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            fail(run_id, f"{label} does not reconcile from episode arrays")
    if summary.get("failed_count") != len(expected_failed):
        fail(run_id, "summary failed_count does not reconcile")
    if set(metrics.get("eval/protocol/active_dr_terms") or []) != terms:
        fail(run_id, "raw active DR terms differ")
    if set(summary.get("active_dr_terms") or []) != terms:
        fail(run_id, "summary active DR terms differ")
    raw_ranges = metrics.get("eval/protocol/dr_ranges")
    if not isinstance(raw_ranges, dict) or set(raw_ranges) != terms:
        fail(run_id, "raw DR ranges do not contain exactly all six terms")
    if summary.get("dr_ranges") != raw_ranges:
        fail(run_id, "summary DR ranges do not reconcile with raw metrics")
    if metrics.get("eval/protocol/preset_id") != run.get("preset"):
        fail(run_id, "raw preset id differs from the receipt run")
    if metrics.get("eval/protocol/branch_id") != run_id:
        fail(run_id, "raw branch id differs from the receipt run")

    expected_delay = latency_steps.get(preset, 0)
    expected_histogram = [0] * (expected_delay + 1)
    expected_histogram[expected_delay] = 2560
    raw_delay = {
        key.removeprefix("eval/delay/"): value
        for key, value in metrics.items()
        if key.startswith("eval/delay/")
    }
    summary_delay = summary.get("delay") or {}
    expected_delay_fields = {
        "action_delay_actuator_groups": 5,
        "action_delay_num_lags": 2560,
        "action_delay_min_steps": expected_delay,
        "action_delay_max_steps": expected_delay,
        "action_delay_histogram": expected_histogram,
    }
    for key, expected in expected_delay_fields.items():
        if raw_delay.get(key) != expected or summary_delay.get(key) != expected:
            fail(run_id, f"raw/summary delay field {key} differs from preset mechanics")
    expected_mean = float(expected_delay)
    expected_nonzero = 0.0 if expected_delay == 0 else 1.0
    for key, expected in (
        ("action_delay_mean_steps", expected_mean),
        ("action_delay_nonzero_fraction", expected_nonzero),
    ):
        if not isinstance(raw_delay.get(key), (int, float)) or not math.isclose(
            float(raw_delay[key]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            fail(run_id, f"raw delay field {key} differs")
    if summary_delay != raw_delay:
        fail(run_id, "summary delay object does not exactly reconcile with raw delay telemetry")
    process_histogram = raw_delay.get("action_delay_process_histogram")
    if process_histogram is not None:
        assignments = raw_delay.get("action_delay_process_assignments")
        if (
            isinstance(assignments, bool)
            or not isinstance(assignments, int)
            or assignments <= 0
            or process_histogram != [0] * expected_delay + [assignments]
            or not math.isclose(
                float(raw_delay.get("action_delay_process_mean_steps", math.nan)),
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            fail(run_id, "delay process telemetry is inconsistent")

    fixed_report = metrics.get("eval/protocol/fixed_latency_report")
    if (
        not isinstance(fixed_report, dict)
        or not math.isclose(
            float(fixed_report.get("requested_steps", math.nan)),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or fixed_report.get("pinned_terms") != ["randomize_action_delay"]
        or metrics.get("eval/protocol/fixed_latency_steps") != expected_delay
    ):
        fail(run_id, "fixed-latency protocol metadata differs")
    if preset in physics_levels:
        scale = physics_levels[preset]
        if not math.isclose(
            float(metrics.get("eval/protocol/non_latency_dr_scale", math.nan)),
            scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            fail(run_id, "raw non-latency DR scale differs")
        scale_report = metrics.get("eval/protocol/dr_scale_report")
        if (
            not isinstance(scale_report, dict)
            or not math.isclose(
                float(scale_report.get("lambda_value", math.nan)),
                scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or set(scale_report.get("scaled_terms") or []) != non_latency_terms
            or scale_report.get("num_scaled") != 5
            or scale_report.get("skipped_startup_terms") != []
            or scale_report.get("skipped_unknown_params") != []
        ):
            fail(run_id, "raw DR scale report differs")
    elif (
        metrics.get("eval/protocol/non_latency_dr_scale") is not None
        or metrics.get("eval/protocol/dr_scale_report") is not None
    ):
        fail(run_id, "latency-only cell unexpectedly reports non-latency scaling")

    raw_by_preset[preset] = {"dr_ranges": raw_ranges, "delay": raw_delay}
    summary_by_preset[preset] = summary

if len(metrics_paths) != 15 or set(raw_by_preset) != expected_presets:
    raise SystemExit(f"raw evaluation audit expected 15 unique metrics files, got {len(metrics_paths)}")


def equivalent(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


baseline = raw_by_preset["phys_100"]["dr_ranges"]
for preset, scale in physics_levels.items():
    observed = raw_by_preset[preset]["dr_ranges"]
    for term in sorted(non_latency_terms):
        expected = DS.scaled_term_params(baseline[term], scale, allow_extrapolation=True)
        expected, _ = DS.clamp_params_physical(expected)
        if not equivalent(observed[term], expected):
            raise SystemExit(f"{role}.{preset} live DR range for {term} differs")
    if not equivalent(observed["randomize_action_delay"], {"delay_range": [0.0, 0.0]}):
        raise SystemExit(f"{role}.{preset} physics cell did not pin latency to zero")

nominal = raw_by_preset["phys_000"]["dr_ranges"]
for preset, steps in latency_steps.items():
    observed = raw_by_preset[preset]["dr_ranges"]
    for term in sorted(non_latency_terms):
        if not equivalent(observed[term], nominal[term]):
            raise SystemExit(f"{role}.{preset} non-latency range is not nominal")
    if not equivalent(
        observed["randomize_action_delay"], {"delay_range": [float(steps), float(steps)]}
    ):
        raise SystemExit(f"{role}.{preset} live fixed-latency range differs")

aggregate = receipt.get("mode_summary")
if not isinstance(aggregate, dict) or set(aggregate) != expected_presets:
    raise SystemExit(f"{role} mode_summary preset set differs")
for preset in sorted(expected_presets):
    modes = aggregate[preset]
    if not isinstance(modes, dict) or set(modes) != {mode}:
        raise SystemExit(f"{role}.{preset} mode_summary mode set differs")
    block = modes[mode]
    if block.get("num_runs") != 1 or not isinstance(block.get("metrics"), dict):
        raise SystemExit(f"{role}.{preset} mode_summary run count differs")
    summary = summary_by_preset[preset]
    for metric, metric_block in block["metrics"].items():
        if not isinstance(metric_block, dict):
            raise SystemExit(f"{role}.{preset}.{metric} aggregate is not an object")
        expected = summary.get(metric)
        if (
            metric_block.get("per_checkpoint_seed") != {"8600": expected}
            or metric_block.get("mean") != expected
            or metric_block.get("sample_std") is not None
        ):
            raise SystemExit(f"{role}.{preset}.{metric} run/aggregate reconciliation failed")
PY
}

validate_evaluation_receipt() {
    local receipt="$1"
    local role="$2"
    local mode="$3"
    local training_receipt="$4"
    local config="$5"
    local freeze_manifest="$6"
    assert_regular_single_link "${receipt}"
    local checkpoint checkpoint_sha config_source config_sha installed
    checkpoint="$(jq -er '.checkpoint.path' "${freeze_manifest}")"
    checkpoint_sha="$(jq -er '.checkpoint.sha256' "${freeze_manifest}")"
    config_source="$(readlink -f "${config}")"
    config_sha="$(jq -er '.config.sha256' "${freeze_manifest}")"
    jq -e \
        --arg role "${role}" \
        --arg mode "${mode}" \
        --arg training_receipt "$(readlink -f "${training_receipt}")" \
        --arg checkpoint "$(readlink -f "${checkpoint}")" \
        --arg checkpoint_sha "${checkpoint_sha}" \
        --arg config_source "${config_source}" \
        --arg config_sha "${config_sha}" \
        --arg evaluator_sha "${SUP_EVALUATOR_SHA}" \
        --arg git_sha "$(jq -er '.code_state.git_sha' "${SUP_PREREG}")" \
        --arg panel "${SUP_PANEL}" \
        --arg panel_dir "$(readlink -f "$(jq -er '.motion_file' "${SUP_PANEL}")")" \
        --arg panel_motion_key "$(jq -er '.motion_key' "${SUP_PANEL}")" \
        --arg panel_source_sha "$(jq -er '.source_clip_sha256' "${SUP_PANEL}")" \
        --arg panel_alias_sha "$(jq -er '.alias_keys_sha256' "${SUP_PANEL}")" \
        --arg panel_pool_sha "$(jq -er '.pool_sha256' "${SUP_PANEL}")" \
        --arg panel_split_sha "$(jq -er '.split_sha256' "${SUP_PANEL}")" \
        --arg panel_partition "$(jq -er '.partition' "${SUP_PANEL}")" \
        --arg training_experiment_id "$(jq -er '.experiment_id' "${training_receipt}")" '
        def presets: [
            "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
            "phys_125", "phys_150", "phys_175", "phys_200",
            "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms", "lat_60ms"
        ];
        def preset_metadata: {
            "phys_000": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 0.0, "fixed_latency_steps": 0},
            "phys_025": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 0.25, "fixed_latency_steps": 0},
            "phys_050": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 0.5, "fixed_latency_steps": 0},
            "phys_075": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 0.75, "fixed_latency_steps": 0},
            "phys_100": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 1.0, "fixed_latency_steps": 0},
            "phys_125": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 1.25, "fixed_latency_steps": 0},
            "phys_150": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 1.5, "fixed_latency_steps": 0},
            "phys_175": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 1.75, "fixed_latency_steps": 0},
            "phys_200": {"event_preset": "tracking/lucid_curriculum", "non_latency_dr_scale": 2.0, "fixed_latency_steps": 0},
            "lat_10ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 2},
            "lat_20ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 4},
            "lat_30ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 6},
            "lat_40ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 8},
            "lat_50ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 10},
            "lat_60ms": {"event_preset": "tracking/lucid_eval_clean", "fixed_latency_steps": 12}
        };
        def rate: type == "number" and . >= 0 and . <= 1;
        .kind == "lucid_frozen_checkpoint_robustness_evaluation"
        and .schema_version == 1
        and .git_sha == $git_sha
        and .git_status_short == []
        and (.verified | type == "array" and length > 0)
        and .launcher_sha256 == $evaluator_sha
        and .training_receipt == $training_receipt
        and .training_experiment_id == $training_experiment_id
        and .protocol.num_envs == 512
        and .protocol.checkpoint_seeds == [8600]
        and .protocol.evaluation_seed_by_checkpoint_seed == {"8600": 8700}
        and .protocol.modes == [$mode]
        and .protocol.presets == preset_metadata
        and .protocol.max_delay_capacity_steps == 12
        and .protocol.physics_step_ms == 5
        and .protocol.no_learning == true
        and .protocol.suite.motion_count == 512
        and .protocol.suite.motion_file == $panel_dir
        and .protocol.suite.motion_keys_sha256 == $panel_alias_sha
        and .protocol.suite.pool_sha256 == $panel_pool_sha
        and .protocol.suite.split_sha256 == $panel_split_sha
        and .protocol.suite.split_linkage == "replicate-panel"
        and .protocol.suite.partition == $panel_partition
        and .protocol.suite.replicate_panel.receipt == $panel
        and .protocol.suite.replicate_panel.motion_key == $panel_motion_key
        and .protocol.suite.replicate_panel.source_clip_sha256 == $panel_source_sha
        and .protocol.suite.replicate_panel.replicates == 512
        and .protocol.suite.replicate_panel.alias_keys_sha256 == $panel_alias_sha
        and .protocol.resolved_training_config.source == $config_source
        and .protocol.resolved_training_config.sha256 == $config_sha
        and (.protocol.resolved_training_config.installed | length) == 1
        and ((.runs | length) == 15)
        and ((.commands | keys | sort) == (.runs | keys | sort))
        and ([.runs[] | select(
            .complete == true and .runtime.exit_code == 0
            and .checkpoint_seed == 8600 and .evaluation_seed == 8700
            and .mode == $mode and .checkpoint == $checkpoint
            and .checkpoint_sha256 == $checkpoint_sha
            and .summary.motion_count == 512
            and (.summary.success_rate | rate)
            and (.summary.progress_rate | rate)
        )] | length == 15)
        and (([.runs[].preset] | sort) == (presets | sort))
        and ((.mode_summary | keys | sort) == (presets | sort))
        and .checkpoint_sha256_before == {($checkpoint): $checkpoint_sha}
        and .checkpoint_sha256_after == .checkpoint_sha256_before
    ' "${receipt}" >/dev/null || die "invalid ${role} evaluation receipt: ${receipt}"

    installed="$(jq -er '.protocol.resolved_training_config.installed[0]' "${receipt}")"
    assert_sha256 "${installed}" "${config_sha}"
    assert_sha256 "${checkpoint}" "${checkpoint_sha}"
    assert_sha256 "${config_source}" "${config_sha}"
    validate_evaluation_metrics "${receipt}" "${role}" "${mode}"
}

run_or_reuse_evaluation() {
    local role="$1"
    local training_receipt="$2"
    local config="$3"
    local mode="$4"
    local freeze_manifest="$5"
    local receipt_dir="$6"
    local receipt
    full_live_revalidation
    validate_freeze_manifest \
        "${freeze_manifest}" "${training_receipt}" "${config}" "${mode}" "${role}"
    validate_freeze_adjacent_config "${freeze_manifest}"
    mkdir -p "${receipt_dir}"
    assert_unaliased_directory "${receipt_dir}"
    if receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json' 2>/dev/null)"; then
        validate_evaluation_receipt \
            "${receipt}" "${role}" "${mode}" "${training_receipt}" "${config}" \
            "${freeze_manifest}"
        echo "reusing complete ${role} evaluation receipt ${receipt}"
        return
    fi
    if find "${receipt_dir}" -maxdepth 1 -type f -name 'curriculum_robustness_ne512_*.json' \
        -print -quit | grep -q .; then
        die "partial or ambiguous ${role} evaluation receipts in ${receipt_dir}"
    fi
    if [[ -e "${receipt_dir}/.started" ]]; then
        die "${role} evaluation was started without one valid complete receipt; no resume or automatic retry is allowed"
    fi
    wait_for_idle_gpu
    full_live_revalidation
    if [[ "${role}" != "historical_fixed" ]]; then
        validate_training_receipt "${training_receipt}" "${role}" "${mode}"
    fi
    validate_freeze_manifest \
        "${freeze_manifest}" "${training_receipt}" "${config}" "${mode}" "${role}"
    validate_freeze_adjacent_config "${freeze_manifest}"
    [[ ! -e "${receipt_dir}/.started" ]] \
        || die "${role} evaluation marker appeared while waiting; refusing a concurrent launch"
    mkdir "${receipt_dir}/.started"
    python scripts/practice_utility/run_curriculum_robustness_eval.py \
        --training-receipt "${training_receipt}" \
        --training-config "${config}" \
        --panel-receipt "${SUP_PANEL}" \
        --num-envs 512 \
        --seeds 8600 \
        --modes "${mode}" \
        --eval-seed-base 8700 \
        --max-delay 12 \
        --presets "${SUP_PRESETS[@]}" \
        --smpl-motion-file dummy \
        --artifact-root "${SUP_EVAL_ARTIFACT_ROOT}" \
        --log-dir "${SUP_EVAL_LOG_ROOT}" \
        --receipt-dir "${receipt_dir}" \
        --min-free-mib 6000 \
        --execute
    receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json')"
    validate_evaluation_receipt \
        "${receipt}" "${role}" "${mode}" "${training_receipt}" "${config}" \
        "${freeze_manifest}"
}

run_analyzer() {
    local out="$1"
    local historical_eval="$2"
    local fresh_eval="$3"
    local fixed150_eval="$4"
    local fixedu150_eval="$5"
    local fresh_training="$6"
    local fixed150_training="$7"
    local fixedu150_training="$8"
    local historical_freeze="$9"
    local fresh_freeze="${10}"
    local fixed150_freeze="${11}"
    local fixedu150_freeze="${12}"
    python scripts/practice_utility/analyze_support_screen.py \
        --historical-fixed "${historical_eval}" \
        --fresh-fixed "${fresh_eval}" \
        --fixed-150 "${fixed150_eval}" \
        --fixed-u150 "${fixedu150_eval}" \
        --fresh-fixed-training "${fresh_training}" \
        --fixed-150-training "${fixed150_training}" \
        --fixed-u150-training "${fixedu150_training}" \
        --preregistration "${SUP_PREREG}" \
        --expected-preregistration-sha "${LUCID_SUPPORT_SCREEN_PREREG_SHA256}" \
        --historical-fixed-freeze-manifest "${historical_freeze}" \
        --fresh-fixed-freeze-manifest "${fresh_freeze}" \
        --fixed-150-freeze-manifest "${fixed150_freeze}" \
        --fixed-u150-freeze-manifest "${fixedu150_freeze}" \
        --out "${out}"
}

validate_analysis() {
    local analysis="$1"
    jq -e '
        .kind == "lucid_tier2_support_screen_analysis"
        and .instrument_audit.passed == true
        and .instrument_audit.unique_cells == 60
        and .instrument_audit.cross_role_live_dr.passed == true
        and .instrument_audit.cross_role_live_dr.roles == [
            "historical_fixed", "fresh_fixed", "fixed_150", "fixed_u150"
        ]
        and (.instrument_audit.cross_role_live_dr.canonical_sha256_by_preset
             | type == "object" and length == 15)
        and .claim_scope.status == "screening_only"
        and .claim_scope.directional_claim_authorized == false
        and .claim_scope.superiority_claim_authorized == false
        and .decision.screening_only == true
        and .decision.directional_claim_authorized == false
        and .decision.superiority_claim_authorized == false
        and (.decision.status == "screen_pass"
             or .decision.status == "screen_fail"
             or .decision.status == "invalid_bridge")
    ' "${analysis}" >/dev/null || die "support-screen analyzer produced an invalid receipt"
}

run_analysis() {
    local fresh_training="$1"
    local fixed150_training="$2"
    local fixedu150_training="$3"
    local historical_freeze="$4"
    local fresh_freeze="$5"
    local fixed150_freeze="$6"
    local fixedu150_freeze="$7"
    local historical_eval fresh_eval fixed150_eval fixedu150_eval
    historical_eval="$(single_receipt "${SUP_EVAL_HISTORICAL}" 'curriculum_robustness_ne512_*.json')"
    fresh_eval="$(single_receipt "${SUP_EVAL_FRESH}" 'curriculum_robustness_ne512_*.json')"
    fixed150_eval="$(single_receipt "${SUP_EVAL_FIXED150}" 'curriculum_robustness_ne512_*.json')"
    fixedu150_eval="$(single_receipt "${SUP_EVAL_FIXEDU150}" 'curriculum_robustness_ne512_*.json')"

    # This is the last trust boundary before any analysis output or temporary
    # replay directory exists. Re-read every live/frozen input and all 60 raw
    # cells so a reboot, retargeted symlink, or concurrent edit cannot be
    # hidden behind receipts validated earlier in the campaign.
    full_live_revalidation
    validate_training_receipt "${fresh_training}" fresh_fixed fixed
    validate_training_receipt "${fixed150_training}" fixed_150 fixed_150
    validate_training_receipt "${fixedu150_training}" fixed_u150 fixed_u150
    validate_freeze_manifest \
        "${historical_freeze}" "${SUP_HISTORICAL_TRAINING}" \
        "${SUP_HISTORICAL_CONFIG}" fixed historical_fixed
    validate_freeze_manifest "${fresh_freeze}" "${fresh_training}" \
        "$(config_for_training_receipt "${fresh_training}" fixed)" fixed fresh_fixed
    validate_freeze_manifest "${fixed150_freeze}" "${fixed150_training}" \
        "$(config_for_training_receipt "${fixed150_training}" fixed_150)" fixed_150 fixed_150
    validate_freeze_manifest "${fixedu150_freeze}" "${fixedu150_training}" \
        "$(config_for_training_receipt "${fixedu150_training}" fixed_u150)" fixed_u150 fixed_u150
    validate_freeze_adjacent_config "${historical_freeze}"
    validate_freeze_adjacent_config "${fresh_freeze}"
    validate_freeze_adjacent_config "${fixed150_freeze}"
    validate_freeze_adjacent_config "${fixedu150_freeze}"
    validate_evaluation_receipt \
        "${historical_eval}" historical_fixed fixed "${SUP_HISTORICAL_TRAINING}" \
        "${SUP_HISTORICAL_CONFIG}" "${historical_freeze}"
    validate_evaluation_receipt \
        "${fresh_eval}" fresh_fixed fixed "${fresh_training}" \
        "$(config_for_training_receipt "${fresh_training}" fixed)" "${fresh_freeze}"
    validate_evaluation_receipt \
        "${fixed150_eval}" fixed_150 fixed_150 "${fixed150_training}" \
        "$(config_for_training_receipt "${fixed150_training}" fixed_150)" \
        "${fixed150_freeze}"
    validate_evaluation_receipt \
        "${fixedu150_eval}" fixed_u150 fixed_u150 "${fixedu150_training}" \
        "$(config_for_training_receipt "${fixedu150_training}" fixed_u150)" \
        "${fixedu150_freeze}"

    assert_preregistered_state
    validate_live_data_contract
    local replay_dir replay
    replay_dir="$(mktemp -d /tmp/lucid-support-screen-replay.XXXXXX)"
    replay="${replay_dir}/analysis.json"
    if [[ -e "${SUP_ANALYSIS}" ]]; then
        assert_regular_single_link "${SUP_ANALYSIS}"
        assert_no_write_bits "${SUP_ANALYSIS}"
    else
        run_analyzer \
            "${SUP_ANALYSIS}" "${historical_eval}" "${fresh_eval}" "${fixed150_eval}" \
            "${fixedu150_eval}" "${fresh_training}" "${fixed150_training}" \
            "${fixedu150_training}" "${historical_freeze}" "${fresh_freeze}" \
            "${fixed150_freeze}" "${fixedu150_freeze}"
        validate_analysis "${SUP_ANALYSIS}"
        chmod a-w "${SUP_ANALYSIS}"
    fi
    assert_regular_single_link "${SUP_ANALYSIS}"

    # Immutable analysis/recompute parity is mandatory on first creation and
    # every reuse after a reboot. Only the wall-clock creation timestamp may differ.
    run_analyzer \
        "${replay}" "${historical_eval}" "${fresh_eval}" "${fixed150_eval}" \
        "${fixedu150_eval}" "${fresh_training}" "${fixed150_training}" \
        "${fixedu150_training}" "${historical_freeze}" "${fresh_freeze}" \
        "${fixed150_freeze}" "${fixedu150_freeze}" >/dev/null
    if ! cmp -s \
        <(jq -S 'del(.created_at)' "${SUP_ANALYSIS}") \
        <(jq -S 'del(.created_at)' "${replay}"); then
        die "immutable support-screen analysis does not reproduce from exact frozen inputs"
    fi
    validate_analysis "${SUP_ANALYSIS}"
    assert_no_write_bits "${SUP_ANALYSIS}"
    rm -f "${replay}"
    rmdir "${replay_dir}"
}

preflight() {
    full_live_revalidation
    echo "Tier-2 support-screen preflight passed; no GPU cell has started"
}

main() {
    preflight

    # One process and one receipt boundary per training cell. Do not evaluate
    # or inspect capability until all three new policies have finished training.
    run_or_reuse_training fresh_fixed fixed "${SUP_TRAIN_FRESH}"
    run_or_reuse_training fixed_150 fixed_150 "${SUP_TRAIN_FIXED150}"
    run_or_reuse_training fixed_u150 fixed_u150 "${SUP_TRAIN_FIXEDU150}"

    local fresh_training fixed150_training fixedu150_training
    local fresh_config fixed150_config fixedu150_config
    fresh_training="$(single_receipt "${SUP_TRAIN_FRESH}" 'curriculum_comparison_ne1024_*.json')"
    fixed150_training="$(single_receipt "${SUP_TRAIN_FIXED150}" 'curriculum_comparison_ne1024_*.json')"
    fixedu150_training="$(single_receipt "${SUP_TRAIN_FIXEDU150}" 'curriculum_comparison_ne1024_*.json')"
    fresh_config="$(config_for_training_receipt "${fresh_training}" fixed)"
    fixed150_config="$(config_for_training_receipt "${fixed150_training}" fixed_150)"
    fixedu150_config="$(config_for_training_receipt "${fixedu150_training}" fixed_u150)"

    # All training is complete before any checkpoint/config freeze or eval. The
    # four manifests bind every evaluated checkpoint and config by SHA-256.
    local historical_freeze fresh_freeze fixed150_freeze fixedu150_freeze
    historical_freeze="${SUP_HISTORICAL_FREEZE}"
    validate_freeze_manifest \
        "${historical_freeze}" "${SUP_HISTORICAL_TRAINING}" \
        "${SUP_HISTORICAL_CONFIG}" fixed historical_fixed
    fresh_freeze="$(freeze_or_reuse_checkpoint \
        fresh_fixed "${fresh_training}" "${fresh_config}" fixed)"
    fixed150_freeze="$(freeze_or_reuse_checkpoint \
        fixed_150 "${fixed150_training}" "${fixed150_config}" fixed_150)"
    fixedu150_freeze="$(freeze_or_reuse_checkpoint \
        fixed_u150 "${fixedu150_training}" "${fixedu150_config}" fixed_u150)"

    # The evaluator may create config.yaml beside a checkpoint.  Materialize
    # and verify those links before any one-shot evaluation marker, so no
    # provenance-affecting filesystem write can occur between marker and GPU
    # launch. Existing foreign or mismatched configs are never replaced.
    prepare_freeze_adjacent_config "${historical_freeze}"
    prepare_freeze_adjacent_config "${fresh_freeze}"
    prepare_freeze_adjacent_config "${fixed150_freeze}"
    prepare_freeze_adjacent_config "${fixedu150_freeze}"

    # Exact 15-cell k512 ladders, all paired on evaluation seed 8700.
    run_or_reuse_evaluation \
        historical_fixed "${SUP_HISTORICAL_TRAINING}" "${SUP_HISTORICAL_CONFIG}" fixed \
        "${historical_freeze}" "${SUP_EVAL_HISTORICAL}"
    run_or_reuse_evaluation \
        fresh_fixed "${fresh_training}" "${fresh_config}" fixed \
        "${fresh_freeze}" "${SUP_EVAL_FRESH}"
    run_or_reuse_evaluation \
        fixed_150 "${fixed150_training}" "${fixed150_config}" fixed_150 \
        "${fixed150_freeze}" "${SUP_EVAL_FIXED150}"
    run_or_reuse_evaluation \
        fixed_u150 "${fixedu150_training}" "${fixedu150_config}" fixed_u150 \
        "${fixedu150_freeze}" "${SUP_EVAL_FIXEDU150}"

    # Recheck every frozen bundle after all evaluation processes have exited.
    validate_freeze_manifest \
        "${historical_freeze}" "${SUP_HISTORICAL_TRAINING}" \
        "${SUP_HISTORICAL_CONFIG}" fixed historical_fixed
    validate_freeze_manifest \
        "${fresh_freeze}" "${fresh_training}" "${fresh_config}" fixed fresh_fixed
    validate_freeze_manifest \
        "${fixed150_freeze}" "${fixed150_training}" "${fixed150_config}" fixed_150 fixed_150
    validate_freeze_manifest \
        "${fixedu150_freeze}" "${fixedu150_training}" "${fixedu150_config}" \
        fixed_u150 fixed_u150

    run_analysis \
        "${fresh_training}" "${fixed150_training}" "${fixedu150_training}" \
        "${historical_freeze}" "${fresh_freeze}" "${fixed150_freeze}" \
        "${fixedu150_freeze}"
    mkdir -p "${SUP_ROOT}/.complete"
    jq '{claim_scope, historical_bridge, candidate_screens, decision}' "${SUP_ANALYSIS}"
}

# Activation is checked before sourcing the simulator environment, changing
# directories, creating markers, or touching the GPU.
: "${LUCID_SUPPORT_SCREEN_PREREG_SHA256:?set the future frozen Tier-2 preregistration SHA-256}"
# Do not let an inherited PATH substitute jq/git/hash tools at the only trust
# boundary that precedes execution of the frozen environment bootstrap.  The
# bootstrap then installs the preregistered Python environment for all later
# validation and launch commands.
PATH=/usr/bin:/bin assert_preregistered_state
source "${SUP_ENV}"
cd "${SUP_REPO}"
export LUCID_GPU_WAIT_SECONDS=7200
assert_preregistered_state

if [[ "${1:-}" == "--preflight-only" ]]; then
    preflight
    exit 0
fi
main "$@"
