#!/usr/bin/env bash
# Run the post-H_R2, descriptive historical lucid_rg bridge.
#
# This supervisor is deliberately dormant until a future preregistration is
# frozen and its SHA-256 is supplied by the operator.  It never trains a policy.
# Each of the three historical evaluations has its own one-shot .started marker;
# a marker without one complete valid receipt is fail-closed and may not resume
# or retry.  H_R2 capability may have passed or failed, but all three H_R0
# mechanism gates must have passed before this descriptive bridge can activate.
set -euo pipefail

readonly HIST_REPO="${LUCID_RATCHET_HISTORICAL_BRIDGE_REPO:-/home/linjiw/lucid-ratchet-historical-bridge}"
readonly HIST_ENV="/home/linjiw/lucid/env/lucid_env.sh"
readonly HIST_ROOT="${LUCID_RATCHET_HISTORICAL_BRIDGE_ROOT:-/home/linjiw/lucid-sonic/manifests/ratchet_historical_bridge_20260901}"
readonly HIST_PREREG="${LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG:-/home/linjiw/lucid-sonic/manifests/lucid_ratchet_historical_bridge_preregistration_20260901.json}"
readonly HIST_STAGE_ROOT="${HIST_ROOT}/staged_historical_lucid_rg"
readonly HIST_FREEZE_ROOT="${HIST_ROOT}/frozen_historical_checkpoints"
readonly HIST_EVAL_ROOT="${HIST_ROOT}/evaluation"
readonly HIST_EVAL_ARTIFACT_ROOT="/home/linjiw/lucid-sonic/artifacts/ratchet_historical_bridge_eval_20260901"
readonly HIST_EVAL_LOG_ROOT="/home/linjiw/lucid-sonic/outputs/ratchet_historical_bridge_20260901"
readonly HIST_ANALYSIS="${HIST_ROOT}/lucid_ratchet_historical_bridge_analysis.json"

readonly EXPECTED_EVALUATOR_SHA256="308e24150e4d4f03d0abf0dc6a427063ac662904bb3a7765488a9bff63cd94ca"
readonly EXPECTED_PANEL_SHA256="e2e61933405e6701b0563eb4df793b6faf5c90d8ae5b7d8fc1e11f47142aefd7"
readonly EXPECTED_MOTION_SHA256="a7f10e7aa26e53cc4e346151d4ccd74e932e3aafa1cfaaac77dab8b8eec40929"
readonly EXPECTED_H_R2_AMENDMENT_SHA256="2064bf7a16ca159092c6ebeabfbf09bc2fe3c1b30ce359a64505503a83786044"
readonly EXPECTED_INSTRUMENT_BASE_GIT_SHA="ca057e658acc59773e798057980b827d65988441"
readonly EXPECTED_PANEL_ALIAS_KEYS_SHA256="4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430"
readonly EXPECTED_ENV_SHA256="aa1827d1b415cb21f8aadddc8a8985f62f3f1fc6807b96246ea2dab39d11d743"

readonly -a EXPECTED_INSTRUMENT_ADDITIONS=(
    scripts/practice_utility/analyze_ratchet_historical_bridge.py
    scripts/practice_utility/run_ratchet_historical_bridge.sh
    tests/practice_utility/test_analyze_ratchet_historical_bridge.py
    tests/practice_utility/test_ratchet_historical_bridge_driver.py
)

readonly -A EXPECTED_LUCID_CHECKPOINT_SHA256=(
    [8600]="95aadf780c6bdf90e3d78e90b7ef14ee8a3b03a8362e776f39ea1408dc71fd2a"
    [8601]="e8ece9de91b5d73ea7ef920cc27047068ee1a25ea804d8c7001cf603fb31d70e"
    [8602]="aced3185ca7804d39e67d6223dd47f033808ea449500c1690b8f5d8f41613bf3"
)
readonly -A EXPECTED_LUCID_CONFIG_SHA256=(
    [8600]="4c0b49de050a4c09b687e339cdbed11e4f2a5a3b2130edd3e08649681ce369ff"
    [8601]="9997fe633cf33c319314a8fb28f239c8d70a15e9470209b828f1e591abce3568"
    [8602]="a3cd711fd0456fad745dc9a6b732a38461d63489818f4b5a22c754e9cfb9efb9"
)
readonly -A EXPECTED_LUCID_CURRICULUM_SHA256=(
    [8600]="e37dbdd0da02b42c81dac055d1f41e1a11911a84d062e6be11baeacd092413aa"
    [8601]="3e98983a34b8896fd45a8a72d032ad22048c4f517a7135f25018b0579b0b6e0d"
    [8602]="27d861498121a4b879d6cc47b1016f50e321bcd93db4a5458761e59a603d0537"
)

readonly -a HIST_SEEDS=(8600 8601 8602)
readonly -a HIST_PRESETS=(
    phys_000 phys_025 phys_050 phys_075 phys_100
    phys_125 phys_150 phys_175 phys_200
    lat_10ms lat_20ms lat_30ms lat_40ms lat_50ms
)

die() {
    echo "$*" >&2
    exit 1
}

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

assert_sha256() {
    local path="$1"
    local expected="$2"
    [[ -f "${path}" ]] || die "pinned file is missing: ${path}"
    local actual
    actual="$(sha256 "${path}")"
    [[ "${actual}" == "${expected}" ]] \
        || die "hash mismatch: ${path}: expected ${expected}, got ${actual}"
}

assert_no_write_bits() {
    local path="$1"
    local mode
    mode="$(stat -c '%a' "${path}")"
    (( (8#${mode}) & 8#222 == 0 )) || die "frozen file regained write bits: ${path}"
}

assert_regular_single_link() {
    local path="$1"
    [[ -f "${path}" && ! -L "${path}" ]] \
        || die "expected a regular non-symlink file: ${path}"
    [[ "$(stat -c '%h' "${path}")" == 1 ]] \
        || die "file is a forbidden hardlink: ${path}"
}

assert_real_directory() {
    local path="$1"
    [[ -d "${path}" && ! -L "${path}" ]] \
        || die "expected a non-symlink directory: ${path}"
}

frozen_input_path() {
    local key="$1"
    local path
    path="$(jq -er --arg key "${key}" '.frozen_inputs[$key].path' "${HIST_PREREG}")"
    [[ "${path}" = /* ]] || die "frozen input ${key} is not an absolute path"
    printf '%s\n' "${path}"
}

frozen_input_sha() {
    local key="$1"
    jq -er --arg key "${key}" '.frozen_inputs[$key].sha256' "${HIST_PREREG}"
}

assert_frozen_input() {
    local key="$1"
    assert_sha256 "$(frozen_input_path "${key}")" "$(frozen_input_sha "${key}")"
}

h_r2_freeze_key() {
    local mode="$1"
    local seed="$2"
    printf 'h_r2_%s_s%s_freeze\n' "${mode}" "${seed}"
}

h_r2_evaluation_key() {
    local mode="$1"
    local seed="$2"
    printf 'h_r2_%s_s%s_evaluation\n' "${mode}" "${seed}"
}

historical_key() {
    local seed="$1"
    local kind="$2"
    printf 'historical_lucid_s%s_%s\n' "${seed}" "${kind}"
}

staged_bundle_dir() {
    printf '%s/lucid_rg_s%s\n' "${HIST_STAGE_ROOT}" "$1"
}

validate_instrument_code_closure() {
    git -C "${HIST_REPO}" cat-file -e "${EXPECTED_INSTRUMENT_BASE_GIT_SHA}^{commit}" \
        || die "historical instrument base commit is unavailable"
    git -C "${HIST_REPO}" merge-base --is-ancestor \
        "${EXPECTED_INSTRUMENT_BASE_GIT_SHA}" HEAD \
        || die "historical instrument base is not an ancestor of HEAD"

    local -a observed=()
    local -a expected=()
    local path index
    mapfile -t observed < <(
        git -C "${HIST_REPO}" diff --name-status --no-renames \
            "${EXPECTED_INSTRUMENT_BASE_GIT_SHA}..HEAD" | LC_ALL=C sort
    )
    for path in "${EXPECTED_INSTRUMENT_ADDITIONS[@]}"; do
        expected+=("${path}")
    done
    mapfile -t expected < <(
        for path in "${expected[@]}"; do
            printf 'A\t%s\n' "${path}"
        done | LC_ALL=C sort
    )
    [[ ${#observed[@]} == ${#expected[@]} ]] \
        || die "historical instrument differs from the four-file additive closure"
    for index in "${!expected[@]}"; do
        [[ "${observed[${index}]}" == "${expected[${index}]}" ]] \
            || die "historical instrument diff is not the exact four-file additive closure"
    done

    # Name-status alone admits committed symlinks and unexpected executable
    # modes.  The frozen instrument is four ordinary blobs with one executable
    # supervisor; bind those Git modes as part of the ca057 closure.
    local expected_mode entry metadata mode object object_sha recorded_path
    for path in "${EXPECTED_INSTRUMENT_ADDITIONS[@]}"; do
        expected_mode=100644
        if [[ "${path}" == scripts/practice_utility/run_ratchet_historical_bridge.sh ]]; then
            expected_mode=100755
        fi
        entry="$(git -C "${HIST_REPO}" ls-tree HEAD -- "${path}")"
        IFS=$'\t' read -r metadata recorded_path <<<"${entry}"
        read -r mode object object_sha <<<"${metadata}"
        [[ "${mode}" == "${expected_mode}" && "${object}" == blob \
            && "${recorded_path}" == "${path}" ]] \
            || die "historical instrument Git mode/type differs for ${path}"
        assert_regular_single_link "${HIST_REPO}/${path}"
    done
}

assert_preregistered_state() {
    : "${LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG_SHA256:?set the future frozen historical-bridge preregistration SHA-256}"
    assert_sha256 "${HIST_PREREG}" "${LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG_SHA256}"

    jq -e \
        --arg repo "${HIST_REPO}" \
        --arg stage_root "${HIST_STAGE_ROOT}" \
        --arg instrument_base "${EXPECTED_INSTRUMENT_BASE_GIT_SHA}" \
        --arg env_path "${HIST_ENV}" \
        --arg env_sha "${EXPECTED_ENV_SHA256}" \
        --arg evaluator_sha "${EXPECTED_EVALUATOR_SHA256}" \
        --arg panel_sha "${EXPECTED_PANEL_SHA256}" \
        --arg motion_sha "${EXPECTED_MOTION_SHA256}" '
        .kind == "lucid_ratchet_historical_bridge_preregistration"
        and .schema_version == 1
        and .state == "frozen_before_bridge_evaluation"
        and .claim_scope.classification == "posthoc_descriptive"
        and .claim_scope.binding == false
        and .claim_scope.alters_H_R2 == false
        and .claim_scope.inference == "none"
        and .claim_scope.noninferiority_claim_authorized == false
        and .claim_scope.superiority_claim_authorized == false
        and .activation.h_r2_capability_status_allowed == ["pass", "fail"]
        and .activation.require_all_h_r0_mechanism_gates == true
        and .instrument.num_envs == 512
        and .instrument.cells_per_historical_checkpoint == 14
        and .instrument.total_new_cells == 42
        and .instrument.total_analysis_cells == 126
        and .instrument.evaluator_sha256 == $evaluator_sha
        and .instrument.panel_sha256 == $panel_sha
        and .instrument.motion_sha256 == $motion_sha
        and .instrument.evaluation_seed_by_training_seed
             == {"8600": 8700, "8601": 8701, "8602": 8702}
        and .instrument.presets == [
          "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
          "phys_125", "phys_150", "phys_175", "phys_200",
          "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms"
        ]
        and .code_state.worktree == $repo
        and .code_state.clean_detached_worktree_required == true
        and .code_state.instrument_base_git_sha == $instrument_base
        and .code_state.allowed_changes_from_instrument_base == [
          {"status": "A", "path": "scripts/practice_utility/analyze_ratchet_historical_bridge.py"},
          {"status": "A", "path": "scripts/practice_utility/run_ratchet_historical_bridge.sh"},
          {"status": "A", "path": "tests/practice_utility/test_analyze_ratchet_historical_bridge.py"},
          {"status": "A", "path": "tests/practice_utility/test_ratchet_historical_bridge_driver.py"}
        ]
        and .staging.root == $stage_root
        and .staging.bundle_layout == "checkpoint_with_adjacent_true_config"
        and .staging.allowed_copy_methods == ["copy", "reflink"]
        and .staging.hardlinks_allowed == false
        and .staging.source_artifact_mutation_allowed == false
        and .environment.path == $env_path
        and .environment.sha256 == $env_sha
        and .frozen_inputs.lucid_env == {"path": $env_path, "sha256": $env_sha}
        and (.code_state.git_sha | type == "string" and length == 40)
    ' "${HIST_PREREG}" >/dev/null \
        || die "historical-bridge preregistration contract differs"

    local expected_git
    expected_git="$(jq -er '.code_state.git_sha' "${HIST_PREREG}")"
    [[ "$(git -C "${HIST_REPO}" rev-parse HEAD)" == "${expected_git}" ]] \
        || die "historical-bridge HEAD differs from preregistered commit ${expected_git}"
    if git -C "${HIST_REPO}" symbolic-ref -q HEAD >/dev/null; then
        die "historical-bridge worktree is not detached: ${HIST_REPO}"
    fi
    git -C "${HIST_REPO}" diff --quiet
    git -C "${HIST_REPO}" diff --cached --quiet
    [[ -z "$(git -C "${HIST_REPO}" status --porcelain --untracked-files=all)" ]] \
        || die "historical-bridge worktree is not clean: ${HIST_REPO}"
    validate_instrument_code_closure

    local relative expected
    while IFS=$'\t' read -r relative expected; do
        assert_sha256 "${HIST_REPO}/${relative}" "${expected}"
    done < <(
        jq -r '.code_state.file_sha256 | to_entries[] | [.key, .value] | @tsv' \
            "${HIST_PREREG}"
    )
    for relative in \
        scripts/practice_utility/run_curriculum_robustness_eval.py \
        scripts/practice_utility/analyze_ratchet.py \
        scripts/practice_utility/analyze_ratchet_historical_bridge.py \
        scripts/practice_utility/freeze_training_checkpoint.py \
        scripts/practice_utility/run_ratchet_historical_bridge.sh \
        tests/practice_utility/test_analyze_ratchet_historical_bridge.py \
        tests/practice_utility/test_ratchet_historical_bridge_driver.py; do
        jq -e --arg path "${relative}" '.code_state.file_sha256[$path] | type == "string"' \
            "${HIST_PREREG}" >/dev/null \
            || die "preregistration does not pin required code file ${relative}"
    done
    local running_driver expected_driver_sha
    running_driver="$(readlink -f "${BASH_SOURCE[0]}")"
    [[ "${running_driver}" \
        == "$(readlink -f "${HIST_REPO}/scripts/practice_utility/run_ratchet_historical_bridge.sh")" ]] \
        || die "executed historical driver is outside the preregistered worktree"
    expected_driver_sha="$(jq -er \
        '.code_state.file_sha256["scripts/practice_utility/run_ratchet_historical_bridge.sh"]' \
        "${HIST_PREREG}")"
    assert_sha256 "${running_driver}" "${expected_driver_sha}"

    while IFS=$'\t' read -r path expected; do
        [[ "${path}" = /* ]] || die "frozen input is not absolute: ${path}"
        assert_sha256 "${path}" "${expected}"
    done < <(
        jq -r '.frozen_inputs | to_entries[] | [.value.path, .value.sha256] | @tsv' \
            "${HIST_PREREG}"
    )

    local required=(lucid_env panel_receipt motion h_r2_amendment h_r2_analysis)
    local seed mode kind
    for seed in "${HIST_SEEDS[@]}"; do
        for mode in fixed ratchet; do
            required+=("$(h_r2_freeze_key "${mode}" "${seed}")")
            required+=("$(h_r2_evaluation_key "${mode}" "${seed}")")
        done
        for kind in bridge checkpoint config curriculum; do
            required+=("$(historical_key "${seed}" "${kind}")")
        done
    done
    local key
    for key in "${required[@]}"; do
        assert_frozen_input "${key}"
    done
    [[ "$(frozen_input_path lucid_env)" == "${HIST_ENV}" ]] \
        || die "future preregistration does not pin the exact environment bootstrap path"
    [[ "$(frozen_input_sha lucid_env)" == "${EXPECTED_ENV_SHA256}" ]] \
        || die "future preregistration does not pin the exact environment bootstrap SHA"
    assert_sha256 "${HIST_ENV}" "${EXPECTED_ENV_SHA256}"

    local evaluator analyzer
    evaluator="${HIST_REPO}/scripts/practice_utility/run_curriculum_robustness_eval.py"
    analyzer="${HIST_REPO}/scripts/practice_utility/analyze_ratchet_historical_bridge.py"
    [[ "$(sha256 "${evaluator}")" == "${EXPECTED_EVALUATOR_SHA256}" ]] \
        || die "worktree evaluator is not byte-identical to H_R2"
    [[ "$(sha256 "${evaluator}")" == "$(jq -er '.instrument.evaluator_sha256' "${HIST_PREREG}")" ]] \
        || die "preregistered evaluator hash differs from the worktree"
    [[ "$(sha256 "${analyzer}")" == "$(jq -er '.instrument.analyzer_sha256' "${HIST_PREREG}")" ]] \
        || die "preregistered historical analyzer hash differs from the worktree"
}

validate_h_r2_gate() {
    local analysis amendment
    analysis="$(frozen_input_path h_r2_analysis)"
    amendment="$(frozen_input_path h_r2_amendment)"
    [[ "$(frozen_input_sha h_r2_amendment)" == "${EXPECTED_H_R2_AMENDMENT_SHA256}" ]] \
        || die "H_R2 amendment pin differs from the frozen amendment"

    jq -e '
        .kind == "lucid_ratchet_analysis"
        and .instrument_audit.passed == true
        and .instrument_audit.cell_count == 84
        and .instrument_audit.paired_training_seeds == ["8600", "8601", "8602"]
        and .claim_scope.status == "three_seed_decision"
        and .claim_scope.paired_training_seeds == ["8600", "8601", "8602"]
        and .claim_scope.noninferiority_decision_eligible == true
        and (.preregistered_decision.status == "pass"
             or .preregistered_decision.status == "fail")
        and .preregistered_decision.mechanism_complete == true
        and .preregistered_decision.mechanism_pass == true
        and .preregistered_decision.paired_training_seeds == ["8600", "8601", "8602"]
        and .preregistered_decision.noninferiority_decision_eligible == true
        and .preregistered_decision.superiority_claim_authorized == false
        and .mechanism.summary.per_seed_all_gates_pass
             == {"8600": true, "8601": true, "8602": true}
        and .mechanism.summary.all_available_seeds_pass == true
        and ([.inputs.robustness_receipts[]] | length == 6)
        and ([.inputs.training_receipts[]] | length == 3)
    ' "${analysis}" >/dev/null || die "terminal H_R2/H_R0 activation audit failed"

    # The future preregistration directly pins the same six receipts that the
    # immutable H_R2 analysis records.  No substituted or newly scored H_R2
    # receipt is admitted to the nine-receipt descriptive union.
    if ! diff -u \
        <(
            jq -r '.inputs.robustness_receipts[] | [.path, .sha256] | @tsv' \
                "${analysis}" | sort
        ) \
        <(
            local seed mode key
            for seed in "${HIST_SEEDS[@]}"; do
                for mode in fixed ratchet; do
                    key="$(h_r2_evaluation_key "${mode}" "${seed}")"
                    printf '%s\t%s\n' "$(frozen_input_path "${key}")" "$(frozen_input_sha "${key}")"
                done
            done | sort
        ) >/dev/null; then
        die "preregistered six-receipt H_R2 set differs from the terminal analysis"
    fi

    local -a freezes=()
    local seed mode
    for seed in "${HIST_SEEDS[@]}"; do
        for mode in fixed ratchet; do
            freezes+=("$(frozen_input_path "$(h_r2_freeze_key "${mode}" "${seed}")")")
        done
    done
    python - "${analysis}" "${amendment}" "${freezes[@]}" <<'PY'
from pathlib import Path
import sys

from scripts.practice_utility import analyze_ratchet as ratchet
from scripts.practice_utility import analyze_ratchet_historical_bridge as bridge

analysis_path = Path(sys.argv[1])
amendment_path = Path(sys.argv[2])
freeze_paths = [Path(value) for value in sys.argv[3:]]
analysis, robustness, training, _ = bridge.audit_terminal_h_r2(analysis_path)
instrument = ratchet.audit_instrument(robustness)
freezes = bridge.audit_h_r2_freeze_manifests(freeze_paths, training, instrument)
bridge.audit_h_r2_amendment(amendment_path, freezes, robustness, training)
assert analysis["preregistered_decision"]["status"] in ("pass", "fail")
assert analysis["preregistered_decision"]["mechanism_pass"] is True
PY
}

validate_panel_alias_tree() {
    local panel="$1"
    local motion="$2"
    local alias_sha="$3"
    local motion_sha="$4"
    python - "${panel}" "${motion}" "${alias_sha}" "${motion_sha}" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

panel_path = Path(sys.argv[1])
motion_path = Path(sys.argv[2])
expected_alias_sha = sys.argv[3]
expected_motion_sha = sys.argv[4]


def fail(message: str) -> None:
    raise SystemExit(f"frozen panel alias-tree audit failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


panel = json.loads(panel_path.read_text())
try:
    motion = motion_path.resolve(strict=True)
    source = Path(panel["source_clip"]).resolve(strict=True)
    motion_dir = Path(panel["motion_file"]).resolve(strict=True)
except (KeyError, OSError) as error:
    fail(f"a canonical path is missing: {error}")
if not motion.is_file() or source != motion:
    fail("panel source_clip is not the preregistered motion file")
if sha256(motion) != expected_motion_sha:
    fail("canonical source-clip bytes differ")
if not motion_dir.is_dir():
    fail("motion_file is not a directory")

aliases = sorted(motion_dir.glob("*.pkl"), key=lambda path: path.name)
entries = sorted(motion_dir.iterdir(), key=lambda path: path.name)
if len(entries) != 512 or entries != aliases:
    fail(
        "panel directory must contain exactly the 512 .pkl aliases and no other entries; "
        f"observed {len(entries)} total entries and {len(aliases)} top-level .pkl entries"
    )
if any(not alias.is_symlink() for alias in aliases):
    fail("every .pkl alias must be a symlink")
stems = [alias.stem for alias in aliases]
stem_sha = hashlib.sha256(("\n".join(sorted(stems)) + "\n").encode()).hexdigest()
if stem_sha != expected_alias_sha:
    fail(f"alias-stem digest differs: {stem_sha}")
try:
    targets = {alias.resolve(strict=True) for alias in aliases}
except OSError as error:
    fail(f"an alias target is missing: {error}")
if targets != {motion}:
    fail("the aliases do not all resolve to the one canonical source clip")
if sha256(next(iter(targets))) != expected_motion_sha:
    fail("resolved alias-target bytes differ")
PY
}

validate_panel_and_motion() {
    local panel motion
    panel="$(frozen_input_path panel_receipt)"
    motion="$(frozen_input_path motion)"
    [[ "$(frozen_input_sha panel_receipt)" == "${EXPECTED_PANEL_SHA256}" ]] \
        || die "historical bridge panel is not the exact H_R2 panel"
    [[ "$(frozen_input_sha motion)" == "${EXPECTED_MOTION_SHA256}" ]] \
        || die "historical bridge motion is not the exact H_R2 motion"
    jq -e --arg motion_sha "${EXPECTED_MOTION_SHA256}" '
        .kind == "lucid_replicate_panel"
        and .schema_version == 1
        and (.verified | type == "array" and length > 0)
        and .replicates == 512
        and .motion_key == "walk_hands_on_back_loop_002__A066_M"
        and (.source_clip | type == "string" and length > 0)
        and .source_clip_sha256 == $motion_sha
        and .alias_keys_sha256
             == "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430"
        and (.motion_file | type == "string" and length > 0)
        and (.pool_sha256 | type == "string" and length == 64)
        and (.split_sha256 | type == "string" and length == 64)
        and (.partition | type == "string" and length > 0)
    ' "${panel}" >/dev/null || die "frozen H_R2 panel identity differs"
    local panel_source
    panel_source="$(jq -er '.source_clip' "${panel}")"
    assert_sha256 "${panel_source}" "${EXPECTED_MOTION_SHA256}"
    [[ "$(readlink -f "${panel_source}")" == "$(readlink -f "${motion}")" ]] \
        || die "panel source clip is not the frozen H_R2 motion path"
    validate_panel_alias_tree \
        "${panel}" "${motion}" "${EXPECTED_PANEL_ALIAS_KEYS_SHA256}" \
        "${EXPECTED_MOTION_SHA256}"
}

validate_checkpoint_config_adjacency() {
    local checkpoint="$1"
    local config="$2"
    local checkpoint_sha="$3"
    local config_sha="$4"
    local adjacent
    adjacent="$(dirname "${checkpoint}")/config.yaml"

    [[ -f "${checkpoint}" && ! -L "${checkpoint}" ]] \
        || die "staged checkpoint must be a regular non-symlink file: ${checkpoint}"
    [[ -f "${config}" && ! -L "${config}" ]] \
        || die "staged true config must be a regular non-symlink file: ${config}"
    [[ "$(readlink -f "${config}")" == "$(readlink -f "${adjacent}")" ]] \
        || die "staged true config is not checkpoint.parent/config.yaml"
    [[ "$(stat -c '%h' "${checkpoint}")" == 1 ]] \
        || die "staged checkpoint is a forbidden hardlink: ${checkpoint}"
    [[ "$(stat -c '%h' "${config}")" == 1 ]] \
        || die "staged true config is a forbidden hardlink: ${config}"
    assert_sha256 "${checkpoint}" "${checkpoint_sha}"
    assert_sha256 "${config}" "${config_sha}"
    assert_sha256 "${adjacent}" "${config_sha}"
}

validate_staged_bundle() {
    local seed="$1"
    local require_read_only="${2:-false}"
    local checkpoint config expected_dir
    checkpoint="$(frozen_input_path "$(historical_key "${seed}" checkpoint)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    expected_dir="$(staged_bundle_dir "${seed}")"
    assert_real_directory "${HIST_STAGE_ROOT}"
    assert_real_directory "${expected_dir}"
    [[ "$(dirname "$(readlink -f "${expected_dir}")")" \
        == "$(readlink -f "${HIST_STAGE_ROOT}")" ]] \
        || die "historical seed ${seed} bundle escapes the staged root"
    [[ "${checkpoint}" == "${expected_dir}/final_checkpoint.pt" ]] \
        || die "historical seed ${seed} checkpoint is not in its preregistered staged bundle"
    [[ "${config}" == "${expected_dir}/config.yaml" ]] \
        || die "historical seed ${seed} config is not adjacent in its staged bundle"
    validate_checkpoint_config_adjacency \
        "${checkpoint}" "${config}" \
        "${EXPECTED_LUCID_CHECKPOINT_SHA256[${seed}]}" \
        "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}"
    local -a entries=()
    mapfile -t entries < <(find "${expected_dir}" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
    [[ "${entries[*]}" == "config.yaml final_checkpoint.pt" ]] \
        || die "historical seed ${seed} staged bundle contains unexpected entries"
    if [[ "${require_read_only}" == true ]]; then
        assert_no_write_bits "${checkpoint}"
        assert_no_write_bits "${config}"
        assert_no_write_bits "${expected_dir}"
    fi
}

validate_historical_source() {
    local seed="$1"
    local bridge checkpoint config curriculum capsule capsule_sha
    bridge="$(frozen_input_path "$(historical_key "${seed}" bridge)")"
    checkpoint="$(frozen_input_path "$(historical_key "${seed}" checkpoint)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    curriculum="$(frozen_input_path "$(historical_key "${seed}" curriculum)")"
    capsule="$(jq -er '.arms[].capsule' "${bridge}")"
    capsule_sha="$(jq -er '.sha256.final_capsule' "${bridge}")"

    # The old evaluator insists on checkpoint.parent/config.yaml.  The original
    # s8600 export has a known adjacent symlink to the wrong off-arm config, so
    # the future preregistration must bind a distinct copied/reflinked checkpoint
    # plus its true regular adjacent config.  Original evidence is never mutated.
    validate_staged_bundle "${seed}"

    [[ "$(frozen_input_sha "$(historical_key "${seed}" checkpoint)")" \
        == "${EXPECTED_LUCID_CHECKPOINT_SHA256[${seed}]}" ]] \
        || die "historical seed ${seed} checkpoint is not the predeclared B8000 policy"
    [[ "$(frozen_input_sha "$(historical_key "${seed}" config)")" \
        == "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}" ]] \
        || die "historical seed ${seed} true config is not the predeclared run config"
    [[ "$(frozen_input_sha "$(historical_key "${seed}" curriculum)")" \
        == "${EXPECTED_LUCID_CURRICULUM_SHA256[${seed}]}" ]] \
        || die "historical seed ${seed} curriculum is not the predeclared trace"

    jq -e \
        --argjson seed "${seed}" \
        --arg checkpoint "$(readlink -f "${checkpoint}")" \
        --arg config "$(readlink -f "${config}")" \
        --arg curriculum "$(readlink -f "${curriculum}")" \
        --arg capsule "$(readlink -f "${capsule}")" \
        --arg capsule_sha "${capsule_sha}" \
        --arg checkpoint_sha "${EXPECTED_LUCID_CHECKPOINT_SHA256[${seed}]}" \
        --arg config_sha "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}" \
        --arg curriculum_sha "${EXPECTED_LUCID_CURRICULUM_SHA256[${seed}]}" \
        --arg motion_sha "${EXPECTED_MOTION_SHA256}" '
        .kind == "lucid_historical_training_cell_bridge"
        and .schema_version == 1
        and (.verified | type == "array" and length > 0)
        and .resolved_config == $config
        and .config.from_scratch == true
        and .config.num_envs == 1024
        and .config.iterations == 8000
        and .config.warmup_iterations == 10
        and .config.seeds == [$seed]
        and .config.modes == ["lucid_rg"]
        and .config.consolidation_fraction == 0
        and .config.max_delay_steps == 8
        and (.arms | length == 1)
        and ([.arms[] | select(
            .seed == $seed and .mode == "lucid_rg"
            and .complete == true and .checkpoint_exported == true
            and .iterations_parsed == 8000 and .curriculum_rows == 8000
            and .checkpoint == $checkpoint
            and .curriculum_path == $curriculum
            and .capsule == $capsule
            and .arm_spec.curriculum_mode == "lucid"
            and .arm_spec.anchor_ratio == 0
            and .arm_spec.spread_strata == 1
            and .arm_spec.return_guard == "relative"
            and .arm_spec.monotonic == false
            and .arm_spec.allow_extrapolation == false
            and .arm_spec.margin == null
        )] | length == 1)
        and .sha256.checkpoint == $checkpoint_sha
        and .sha256.resolved_config == $config_sha
        and .sha256.curriculum == $curriculum_sha
        and .sha256.motion == $motion_sha
        and .sha256.final_capsule == $capsule_sha
    ' "${bridge}" >/dev/null || die "historical seed ${seed} training bridge differs"

    assert_sha256 "${capsule}" "${capsule_sha}"
}

validate_all_historical_sources() {
    local seed
    for seed in "${HIST_SEEDS[@]}"; do
        validate_historical_source "${seed}"
    done
    local unique
    unique="$({
        for seed in "${HIST_SEEDS[@]}"; do
            frozen_input_path "$(historical_key "${seed}" checkpoint)"
            frozen_input_path "$(historical_key "${seed}" config)"
            frozen_input_path "$(historical_key "${seed}" curriculum)"
        done
    } | sort -u | wc -l)"
    [[ "${unique}" == 9 ]] || die "historical checkpoint/config/curriculum inputs overlap"
}

historical_freeze_path() {
    printf '%s/lucid_rg_s%s.json\n' "${HIST_FREEZE_ROOT}" "$1"
}

validate_historical_freeze() {
    local seed="$1"
    local manifest="$2"
    local require_read_only="${3:-true}"
    local bridge checkpoint config curriculum capsule capsule_sha
    bridge="$(frozen_input_path "$(historical_key "${seed}" bridge)")"
    checkpoint="$(frozen_input_path "$(historical_key "${seed}" checkpoint)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    curriculum="$(frozen_input_path "$(historical_key "${seed}" curriculum)")"
    capsule="$(jq -er '.arms[].capsule' "${bridge}")"
    capsule_sha="$(jq -er '.sha256.final_capsule' "${bridge}")"
    jq -e \
        --argjson seed "${seed}" \
        --arg bridge "$(readlink -f "${bridge}")" \
        --arg checkpoint "$(readlink -f "${checkpoint}")" \
        --arg config "$(readlink -f "${config}")" \
        --arg curriculum "$(readlink -f "${curriculum}")" \
        --arg capsule "$(readlink -f "${capsule}")" \
        --arg capsule_sha "${capsule_sha}" \
        --arg bridge_sha "$(sha256 "${bridge}")" \
        --arg checkpoint_sha "${EXPECTED_LUCID_CHECKPOINT_SHA256[${seed}]}" \
        --arg config_sha "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}" \
        --arg curriculum_sha "${EXPECTED_LUCID_CURRICULUM_SHA256[${seed}]}" '
        .kind == "lucid_frozen_training_checkpoint"
        and .schema_version == 1
        and .state == "frozen_for_evaluation"
        and .evaluation_only == true
        and .resume_forbidden == true
        and .seed == $seed
        and .mode == "lucid_rg"
        and .iterations == 8000
        and .checkpoint.path == $checkpoint
        and .checkpoint.sha256 == $checkpoint_sha
        and .checkpoint.read_only == true
        and .config.path == $config
        and .config.sha256 == $config_sha
        and .curriculum.path == $curriculum
        and .curriculum.sha256 == $curriculum_sha
        and .curriculum.rows == 8000
        and .final_capsule.path == $capsule
        and .final_capsule.sha256 == $capsule_sha
        and .training_receipt.path == $bridge
        and .training_receipt.sha256 == $bridge_sha
        and (.verified | type == "array" and length > 0)
    ' "${manifest}" >/dev/null || die "historical seed ${seed} freeze manifest differs"

    local section path expected
    for section in checkpoint config curriculum final_capsule training_receipt; do
        path="$(jq -er --arg section "${section}" '.[$section].path' "${manifest}")"
        expected="$(jq -er --arg section "${section}" '.[$section].sha256' "${manifest}")"
        assert_sha256 "${path}" "${expected}"
    done
    if [[ "${require_read_only}" == true ]]; then
        # Only the distinct staged checkpoint/config bundle is made immutable.
        # The hash-pinned historical bridge, curriculum, and capsule are original
        # evidence and this supervisor must never chmod or otherwise mutate them.
        assert_no_write_bits "$(jq -er '.checkpoint.path' "${manifest}")"
        assert_no_write_bits "$(jq -er '.config.path' "${manifest}")"
        assert_no_write_bits "${manifest}"
    fi
}

freeze_or_reuse_historical() {
    local seed="$1"
    local bridge config manifest
    bridge="$(frozen_input_path "$(historical_key "${seed}" bridge)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    manifest="$(historical_freeze_path "${seed}")"

    assert_preregistered_state
    validate_h_r2_gate
    validate_all_historical_sources
    if [[ -L "${manifest}" ]]; then
        die "historical freeze manifest path is a symlink: ${manifest}"
    elif [[ ! -e "${manifest}" ]]; then
        python scripts/practice_utility/freeze_training_checkpoint.py \
            --training-receipt "${bridge}" \
            --config "${config}" \
            --seed "${seed}" \
            --mode lucid_rg \
            --iterations 8000 \
            --make-read-only \
            --out "${manifest}"
    fi
    assert_regular_single_link "${manifest}"

    # Validate every bound path and hash before chmod can touch any path from
    # either a newly written or previously existing manifest.
    validate_historical_freeze "${seed}" "${manifest}" false
    validate_staged_bundle "${seed}"

    # Freeze only the copied/reflinked staged files.  The original curriculum,
    # capsule, and bridge stay byte-pinned but their modes are left untouched.
    local section path
    for section in checkpoint config; do
        path="$(jq -er --arg section "${section}" '.[$section].path' "${manifest}")"
        chmod a-w "${path}"
    done
    chmod a-w "$(staged_bundle_dir "${seed}")"
    chmod a-w "${manifest}"
    validate_historical_freeze "${seed}" "${manifest}"
    validate_staged_bundle "${seed}" true
}

validate_all_historical_freezes() {
    local -a bridges=()
    local seed
    for seed in "${HIST_SEEDS[@]}"; do
        validate_historical_freeze "${seed}" "$(historical_freeze_path "${seed}")"
        validate_staged_bundle "${seed}" true
        bridges+=("$(frozen_input_path "$(historical_key "${seed}" bridge)")")
    done
    python - "${bridges[@]}" <<'PY'
from pathlib import Path
import sys

from scripts.practice_utility import analyze_ratchet_historical_bridge as bridge

records, mechanisms = bridge.audit_historical_bridges([Path(value) for value in sys.argv[1:]])
assert sorted(records) == ["8600", "8601", "8602"]
assert sorted(mechanisms) == ["8600", "8601", "8602"]
PY
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

validate_historical_evaluation_evidence() {
    local receipt="$1"
    local seed="$2"
    local eval_seed="$3"
    local expected_git="$4"
    local bridge="$5"
    local checkpoint="$6"
    local checkpoint_sha="$7"
    local alias_sha="$8"
    local artifact_root="$9"
    local log_root="${10}"
    local repo_root="${11}"
    local panel_path="${12}"
    local config_path="${13}"
    local config_sha="${14}"
    python - \
        "${receipt}" "${seed}" "${eval_seed}" "${expected_git}" \
        "${EXPECTED_EVALUATOR_SHA256}" "${bridge}" "${checkpoint}" \
        "${checkpoint_sha}" "${alias_sha}" "${artifact_root}" "${log_root}" \
        "${repo_root}" "${panel_path}" "${config_path}" "${config_sha}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

from gear_sonic.research.practice_utility import dr_scaling as DS

receipt_path = Path(sys.argv[1])
seed = int(sys.argv[2])
eval_seed = int(sys.argv[3])
expected_git = sys.argv[4]
expected_launcher = sys.argv[5]
bridge = Path(sys.argv[6]).resolve()
checkpoint = Path(sys.argv[7]).resolve()
checkpoint_sha = sys.argv[8]
alias_sha = sys.argv[9]
artifact_root = Path(sys.argv[10]).resolve()
log_root = Path(sys.argv[11]).resolve()
repo_root = Path(sys.argv[12]).resolve()
panel_path = Path(sys.argv[13]).resolve()
config_path = Path(sys.argv[14]).resolve()
config_sha = sys.argv[15]
presets = (
    "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
    "phys_125", "phys_150", "phys_175", "phys_200",
    "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms",
)
metrics = (
    "success_rate",
    "progress_rate",
    "mpjpe_g",
    "mpjpe_l",
    "foot_slip_per_step_m",
    "undesired_contact_rate",
    "torque_saturation",
    "energy_proxy",
)
raw_metric = {
    "success_rate": "eval/success/success_rate",
    "progress_rate": "eval/success/progress_rate",
    "mpjpe_g": "eval/all/mpjpe_g",
    "mpjpe_l": "eval/all/mpjpe_l",
    "foot_slip_per_step_m": "eval/quality/foot_slip_per_step_m",
    "undesired_contact_rate": "eval/quality/undesired_contact_rate",
    "torque_saturation": "eval/quality/torque_saturation",
    "energy_proxy": "eval/quality/energy_proxy",
}
terms = {
    "add_joint_default_pos",
    "base_com",
    "physics_material",
    "push_robot",
    "randomize_action_delay",
    "randomize_rigid_body_mass",
}
non_latency_terms = terms - {"randomize_action_delay"}
physics_levels = {
    "phys_000": 0.00,
    "phys_025": 0.25,
    "phys_050": 0.50,
    "phys_075": 0.75,
    "phys_100": 1.00,
    "phys_125": 1.25,
    "phys_150": 1.50,
    "phys_175": 1.75,
    "phys_200": 2.00,
}
latency_steps = {
    "lat_10ms": 2,
    "lat_20ms": 4,
    "lat_30ms": 6,
    "lat_40ms": 8,
    "lat_50ms": 10,
}
old_evaluator_preset_metadata = {
    "id_clean": "six channels collapsed to LUCID lambda=0 nominal",
    "dr_full": "fresh draws from the complete six-channel training envelope",
    "latency_60ms": (
        "full five non-latency DR channels plus fixed 60 ms latency, "
        "beyond the 0-40 ms train range"
    ),
}


def fail(message: str) -> None:
    raise SystemExit(f"historical evaluation evidence audit failed: {message}")


def same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1e-12
        )
    return left == right


def nested_same(left, right) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(nested_same(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(nested_same(a, b) for a, b in zip(left, right, strict=True))
        )
    return same(left, right)


def rate(value, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        fail(f"{label} is not a finite [0,1] rate")
    return float(value)


receipt = json.loads(receipt_path.read_text())
historical_bridge = json.loads(bridge.read_text())
panel = json.loads(panel_path.read_text())
if receipt.get("kind") != "lucid_frozen_checkpoint_robustness_evaluation":
    fail("kind differs")
if receipt.get("schema_version") != 1:
    fail("schema_version differs")
if receipt.get("launcher_sha256") != expected_launcher:
    fail("launcher SHA differs")
if receipt.get("git_sha") != expected_git:
    fail("git SHA differs from the preregistered worktree")
if receipt.get("git_status_short") != []:
    fail("evaluation worktree was not clean")
if not isinstance(receipt.get("verified"), list) or not receipt["verified"]:
    fail("evaluator receipt is not verified")
experiment_id = receipt.get("experiment_id")
if not isinstance(experiment_id, str) or not experiment_id:
    fail("experiment_id is absent")
try:
    linked_bridge = Path(receipt["training_receipt"]).resolve()
except (KeyError, TypeError):
    fail("training_receipt linkage is absent")
if linked_bridge != bridge:
    fail("training_receipt does not link this seed's historical bridge")
if receipt.get("training_experiment_id") != historical_bridge.get("experiment_id"):
    fail("training_experiment_id does not reconcile with the historical bridge")

protocol = receipt.get("protocol")
if not isinstance(protocol, dict):
    fail("protocol is absent")
if protocol.get("presets") != old_evaluator_preset_metadata:
    fail("old-evaluator preset metadata differs")
if (
    protocol.get("num_envs") != 512
    or protocol.get("checkpoint_seeds") != [seed]
    or protocol.get("evaluation_seed_by_checkpoint_seed") != {str(seed): eval_seed}
    or protocol.get("modes") != ["lucid_rg"]
    or protocol.get("max_delay_capacity_steps") != 12
    or protocol.get("physics_step_ms") != 5
    or protocol.get("no_learning") is not True
):
    fail("protocol identity differs")
resolved_config = protocol.get("resolved_training_config")
if not isinstance(resolved_config, dict) or resolved_config != {
    "source": str(config_path),
    "sha256": config_sha,
    "installed": [str(config_path)],
}:
    fail("resolved training-config lineage differs")
suite = protocol.get("suite")
if not isinstance(suite, dict):
    fail("suite lineage is absent")
try:
    suite_motion = Path(suite["motion_file"]).resolve()
    panel_motion = Path(panel["motion_file"]).resolve()
    linked_panel = Path(suite["replicate_panel"]["receipt"]).resolve()
except (KeyError, TypeError):
    fail("suite/panel path lineage is absent")
expected_replicate = {
    "receipt": str(panel_path),
    "motion_key": panel.get("motion_key"),
    "source_clip_sha256": panel.get("source_clip_sha256"),
    "replicates": panel.get("replicates"),
    "alias_keys_sha256": panel.get("alias_keys_sha256"),
}
if (
    suite_motion != panel_motion
    or linked_panel != panel_path
    or suite.get("motion_count") != 512
    or suite.get("motion_keys_sha256") != alias_sha
    or suite.get("pool_sha256") != panel.get("pool_sha256")
    or suite.get("split_sha256") != panel.get("split_sha256")
    or suite.get("partition") != panel.get("partition")
    or suite.get("split_linkage") != "replicate-panel"
    or suite.get("replicate_panel") != expected_replicate
):
    fail("suite does not exactly reconcile with the frozen panel")

runs = receipt.get("runs")
if not isinstance(runs, dict) or len(runs) != len(presets):
    fail("run mapping is not the exact 14-cell ladder")
commands = receipt.get("commands")
if not isinstance(commands, dict) or set(commands) != set(runs):
    fail("evaluator commands/runs keyspaces differ")
seen_presets: set[str] = set()
metrics_paths: set[Path] = set()
runs_by_preset: dict[str, dict] = {}
ranges_by_preset: dict[str, dict] = {}
for run_id, run in runs.items():
    if not isinstance(run, dict):
        fail(f"run {run_id} is not an object")
    preset = run.get("preset")
    if preset not in presets or preset in seen_presets:
        fail(f"run {run_id} has a missing/duplicate preset")
    seen_presets.add(preset)
    runs_by_preset[preset] = run
    expected_run_id = f"{experiment_id}_s{seed}_lucid_rg_{preset}"
    if run_id != expected_run_id:
        fail(f"run {run_id} does not have the evaluator's exact branch identity")
    if (
        run.get("checkpoint_seed") != seed
        or run.get("evaluation_seed") != eval_seed
        or run.get("mode") != "lucid_rg"
        or run.get("complete") is not True
        or (run.get("runtime") or {}).get("exit_code") != 0
    ):
        fail(f"run {run_id} identity/completion differs")
    try:
        run_checkpoint = Path(run["checkpoint"]).resolve()
    except (KeyError, TypeError):
        fail(f"run {run_id} checkpoint path is absent")
    if run_checkpoint != checkpoint or run.get("checkpoint_sha256") != checkpoint_sha:
        fail(f"run {run_id} checkpoint lineage differs")

    expected_event = (
        "tracking/lucid_curriculum" if preset in physics_levels else "tracking/lucid_eval_clean"
    )
    expected_output_dir = artifact_root / experiment_id / f"seed_{seed}" / "lucid_rg" / preset
    recorded_command = commands.get(run_id)
    if (
        not isinstance(recorded_command, list)
        or not recorded_command
        or not all(isinstance(token, str) for token in recorded_command)
        or Path(recorded_command[0]).resolve() != Path(sys.executable).resolve()
    ):
        fail(f"run {run_id} evaluator interpreter differs from the pinned environment")
    expected_command = [
        recorded_command[0],
        str(repo_root / "scripts/practice_utility/eval_with_delay.py"),
        "--max-delay",
        "12",
        "--",
        f"checkpoint={checkpoint}",
        "+num_envs=512",
        "+headless=true",
        "+use_wandb=false",
        f"+seed={eval_seed}",
        f"+manager_env/events={expected_event}",
        "+use_encoder=g1",
        "+eval_callbacks=[practice_eval]",
        "+run_eval_loop=false",
        "++manager_env.config.train_only_events=[]",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={panel_motion}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
        (
            "++callbacks.practice_eval._target_="
            "gear_sonic.research.practice_utility.eval_callback.PracticeRobustnessEvalCallback"
        ),
        "++callbacks.practice_eval.eval_frequency=1",
        "++callbacks.practice_eval.eval_only=true",
        f"++callbacks.practice_eval.output_dir={expected_output_dir}",
        f"++callbacks.practice_eval.preset_id={preset}",
        f"++callbacks.practice_eval.branch_id={run_id}",
    ]
    if preset in physics_levels:
        expected_command.extend(
            [
                f"++callbacks.practice_eval.non_latency_dr_scale={physics_levels[preset]}",
                "++callbacks.practice_eval.fixed_latency_steps=0",
            ]
        )
    else:
        expected_command.append(
            f"++callbacks.practice_eval.fixed_latency_steps={latency_steps[preset]}"
        )
    if recorded_command != expected_command:
        fail(f"run {run_id} evaluator command differs from the exact frozen cell")

    summary = run.get("summary")
    if not isinstance(summary, dict):
        fail(f"run {run_id} summary is absent")
    rate(summary.get("success_rate"), f"{preset}.summary.success_rate")
    rate(summary.get("progress_rate"), f"{preset}.summary.progress_rate")
    raw_path_value = run.get("metrics_path")
    if not isinstance(raw_path_value, str) or not raw_path_value:
        fail(f"run {run_id} raw metrics path is absent")
    raw_recorded = Path(raw_path_value)
    if (
        not raw_recorded.is_absolute()
        or raw_recorded.is_symlink()
        or not raw_recorded.is_file()
        or raw_recorded.stat().st_nlink != 1
    ):
        fail(f"run {run_id} raw metrics file is not a regular single-link absolute file")
    raw_path = raw_recorded.resolve()
    expected_raw_path = (
        artifact_root
        / experiment_id
        / f"seed_{seed}"
        / "lucid_rg"
        / preset
        / "metrics_eval.json"
    ).resolve()
    if raw_path != expected_raw_path:
        fail(f"run {run_id} raw metrics path is outside its exact evaluator output cell")
    if not raw_path.is_file() or raw_path in metrics_paths:
        fail(f"run {run_id} raw metrics path is missing/reused")
    metrics_paths.add(raw_path)
    expected_log_path = (log_root / f"{run_id}.log").resolve()
    try:
        log_recorded = Path(run["log_path"])
    except (KeyError, TypeError):
        fail(f"run {run_id} log path is absent")
    if (
        not log_recorded.is_absolute()
        or log_recorded.is_symlink()
        or not log_recorded.is_file()
        or log_recorded.stat().st_nlink != 1
    ):
        fail(f"run {run_id} log file is not a regular single-link absolute file")
    log_path = log_recorded.resolve()
    if log_path != expected_log_path:
        fail(f"run {run_id} log path is outside its exact evaluator output cell")
    raw = json.loads(raw_path.read_text())
    if raw.get("eval/protocol/preset_id") != preset:
        fail(f"run {run_id} raw preset differs")
    if raw.get("eval/protocol/branch_id") != run_id:
        fail(f"run {run_id} raw branch differs")
    if set(raw.get("eval/protocol/active_dr_terms") or []) != terms:
        fail(f"run {run_id} raw active DR terms differ")
    if set(summary.get("active_dr_terms") or []) != terms:
        fail(f"run {run_id} summary active DR terms differ")

    ranges = raw.get("eval/protocol/dr_ranges")
    if not isinstance(ranges, dict) or set(ranges) != terms:
        fail(f"run {run_id} raw DR ranges differ")
    ranges_by_preset[preset] = ranges
    expected_steps = 0 if preset in physics_levels else latency_steps[preset]
    fixed_report = raw.get("eval/protocol/fixed_latency_report")
    if (
        not isinstance(fixed_report, dict)
        or not same(fixed_report.get("requested_steps"), expected_steps)
        or fixed_report.get("pinned_terms") != ["randomize_action_delay"]
        or raw.get("eval/protocol/fixed_latency_steps") != expected_steps
    ):
        fail(f"run {run_id} fixed-latency protocol differs")
    if preset in physics_levels:
        scale = physics_levels[preset]
        report = raw.get("eval/protocol/dr_scale_report")
        if (
            not same(raw.get("eval/protocol/non_latency_dr_scale"), scale)
            or not isinstance(report, dict)
            or not same(report.get("lambda_value"), scale)
            or set(report.get("scaled_terms") or []) != non_latency_terms
            or report.get("num_scaled") != 5
            or report.get("skipped_startup_terms") != []
            or report.get("skipped_unknown_params") != []
        ):
            fail(f"run {run_id} physics scaling protocol differs")
    elif (
        raw.get("eval/protocol/non_latency_dr_scale") is not None
        or raw.get("eval/protocol/dr_scale_report") is not None
    ):
        fail(f"run {run_id} latency cell contains a physics scaling request")

    delay = {
        key.removeprefix("eval/delay/"): value
        for key, value in raw.items()
        if key.startswith("eval/delay/")
    }
    expected_lags = 512 * 5
    expected_delay = {
        "action_delay_actuator_groups": 5,
        "action_delay_num_lags": expected_lags,
        "action_delay_min_steps": expected_steps,
        "action_delay_max_steps": expected_steps,
    }
    if any(delay.get(key) != value for key, value in expected_delay.items()):
        fail(f"run {run_id} live delay identity differs")
    if (
        not same(delay.get("action_delay_mean_steps"), expected_steps)
        or not same(
            delay.get("action_delay_nonzero_fraction"), 0.0 if expected_steps == 0 else 1.0
        )
        or delay.get("action_delay_histogram") != [0] * expected_steps + [expected_lags]
    ):
        fail(f"run {run_id} live delay histogram differs")
    process_histogram = delay.get("action_delay_process_histogram")
    if process_histogram is not None:
        assignments = delay.get("action_delay_process_assignments")
        if (
            not isinstance(assignments, int)
            or isinstance(assignments, bool)
            or assignments <= 0
            or process_histogram != [0] * expected_steps + [assignments]
            or not same(delay.get("action_delay_process_mean_steps"), expected_steps)
        ):
            fail(f"run {run_id} live process-delay histogram differs")
    if not nested_same(summary.get("dr_ranges"), ranges):
        fail(f"run {run_id} raw/summary DR ranges differ")
    if not nested_same(summary.get("delay"), delay):
        fail(f"run {run_id} raw/summary delay telemetry differs")

    arrays = raw.get("eval/all_metrics_dict")
    if not isinstance(arrays, dict):
        fail(f"run {run_id} episode arrays are absent")
    motion_keys = arrays.get("motion_keys")
    terminated = arrays.get("terminated")
    progress = arrays.get("progress")
    if not isinstance(motion_keys, list) or len(motion_keys) != 512:
        fail(f"run {run_id} motion_keys length differs")
    if len(set(motion_keys)) != 512 or not all(
        isinstance(value, str) and value for value in motion_keys
    ):
        fail(f"run {run_id} motion_keys are invalid/nonunique")
    observed_alias_sha = hashlib.sha256(
        ("\n".join(sorted(motion_keys)) + "\n").encode()
    ).hexdigest()
    if observed_alias_sha != alias_sha:
        fail(f"run {run_id} raw panel digest differs")
    if not isinstance(terminated, list) or len(terminated) != 512 or not all(
        isinstance(value, bool) for value in terminated
    ):
        fail(f"run {run_id} terminated array differs")
    if not isinstance(progress, list) or len(progress) != 512:
        fail(f"run {run_id} progress array length differs")
    progress_values = [rate(value, f"{preset}.raw.progress") for value in progress]
    if any((not stopped) != (value >= 1.0) for stopped, value in zip(terminated, progress_values, strict=True)):
        fail(f"run {run_id} termination/progress arrays disagree")
    failed_indices = [index for index, value in enumerate(terminated) if value]
    if raw.get("failed_idxes") != failed_indices:
        fail(f"run {run_id} failed_idxes do not reconcile")
    if raw.get("failed_keys") != [motion_keys[index] for index in failed_indices]:
        fail(f"run {run_id} failed_keys do not reconcile")
    success_rate = 1.0 - len(failed_indices) / 512.0
    progress_rate = sum(progress_values) / 512.0
    if not same(raw.get("eval/success/success_rate"), success_rate):
        fail(f"run {run_id} raw success does not reconcile")
    if not same(raw.get("eval/success/progress_rate"), progress_rate):
        fail(f"run {run_id} raw progress does not reconcile")
    if not same(summary.get("success_rate"), success_rate):
        fail(f"run {run_id} summary success does not reconcile")
    if not same(summary.get("progress_rate"), progress_rate):
        fail(f"run {run_id} summary progress does not reconcile")
    if summary.get("motion_count") != 512 or summary.get("failed_count") != len(failed_indices):
        fail(f"run {run_id} summary counts do not reconcile")
    for metric in metrics:
        expected_value = raw.get(raw_metric[metric])
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            expected_value = float(expected_value)
        elif expected_value is not None:
            fail(f"run {run_id} raw metric {metric} has invalid type")
        if not same(summary.get(metric), expected_value):
            fail(f"run {run_id} raw/summary metric {metric} differs")

if seen_presets != set(presets) or len(metrics_paths) != len(presets):
    fail("14 unique presets/raw files were not observed")

baseline = ranges_by_preset["phys_100"]
for preset, scale in physics_levels.items():
    observed = ranges_by_preset[preset]
    for term in sorted(non_latency_terms):
        expected = DS.scaled_term_params(baseline[term], scale, allow_extrapolation=True)
        expected, _ = DS.clamp_params_physical(expected)
        if not nested_same(observed[term], expected):
            fail(f"{preset} live DR range differs for {term}")
    if not nested_same(observed["randomize_action_delay"], {"delay_range": [0.0, 0.0]}):
        fail(f"{preset} live delay range is not zero")
nominal = ranges_by_preset["phys_000"]
for preset, steps in latency_steps.items():
    observed = ranges_by_preset[preset]
    for term in sorted(non_latency_terms):
        if not nested_same(observed[term], nominal[term]):
            fail(f"{preset} non-latency DR drifted from nominal for {term}")
    if not nested_same(
        observed["randomize_action_delay"], {"delay_range": [float(steps), float(steps)]}
    ):
        fail(f"{preset} live delay range differs")

mode_summary = receipt.get("mode_summary")
if not isinstance(mode_summary, dict) or set(mode_summary) != set(presets):
    fail("mode_summary preset keys differ")
for preset in presets:
    modes = mode_summary[preset]
    if not isinstance(modes, dict) or set(modes) != {"lucid_rg"}:
        fail(f"{preset} mode_summary mode differs")
    block = modes["lucid_rg"]
    if block.get("num_runs") != 1 or set(block.get("metrics") or {}) != set(metrics):
        fail(f"{preset} mode_summary shape differs")
    summary = runs_by_preset[preset]["summary"]
    for metric in metrics:
        aggregate = block["metrics"][metric]
        expected_value = summary.get(metric)
        if set(aggregate.get("per_checkpoint_seed") or {}) != {str(seed)}:
            fail(f"{preset}.{metric} aggregate seed mapping differs")
        if not same(aggregate["per_checkpoint_seed"][str(seed)], expected_value):
            fail(f"{preset}.{metric} aggregate per-seed value differs")
        if not same(aggregate.get("mean"), expected_value):
            fail(f"{preset}.{metric} aggregate mean differs")
        if aggregate.get("sample_std") is not None:
            fail(f"{preset}.{metric} one-seed std is not null")
PY
}

validate_historical_evaluation() {
    local receipt="$1"
    local seed="$2"
    local eval_seed="$3"
    local bridge config checkpoint checkpoint_sha panel panel_motion expected_git
    bridge="$(frozen_input_path "$(historical_key "${seed}" bridge)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    checkpoint="$(frozen_input_path "$(historical_key "${seed}" checkpoint)")"
    checkpoint_sha="${EXPECTED_LUCID_CHECKPOINT_SHA256[${seed}]}"
    panel="$(frozen_input_path panel_receipt)"
    panel_motion="$(jq -er '.motion_file' "${panel}")"
    expected_git="$(jq -er '.code_state.git_sha' "${HIST_PREREG}")"

    assert_regular_single_link "${receipt}"

    jq -e \
        --argjson seed "${seed}" \
        --argjson eval_seed "${eval_seed}" \
        --arg evaluator_sha "${EXPECTED_EVALUATOR_SHA256}" \
        --arg checkpoint "$(readlink -f "${checkpoint}")" \
        --arg checkpoint_sha "${checkpoint_sha}" \
        --arg config "$(readlink -f "${config}")" \
        --arg config_sha "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}" \
        --arg panel "${panel}" \
        --arg panel_motion "${panel_motion}" \
        --arg expected_git "${expected_git}" \
        --arg bridge "$(readlink -f "${bridge}")" '
        def expected_presets: [
          "phys_000", "phys_025", "phys_050", "phys_075", "phys_100",
          "phys_125", "phys_150", "phys_175", "phys_200",
          "lat_10ms", "lat_20ms", "lat_30ms", "lat_40ms", "lat_50ms"
        ];
        .kind == "lucid_frozen_checkpoint_robustness_evaluation"
        and .schema_version == 1
        and (.verified | type == "array" and length > 0)
        and .launcher_sha256 == $evaluator_sha
        and .git_sha == $expected_git
        and .git_status_short == []
        and .training_receipt == $bridge
        and .protocol.num_envs == 512
        and .protocol.checkpoint_seeds == [$seed]
        and .protocol.evaluation_seed_by_checkpoint_seed == {($seed | tostring): $eval_seed}
        and .protocol.modes == ["lucid_rg"]
        and .protocol.max_delay_capacity_steps == 12
        and .protocol.physics_step_ms == 5
        and .protocol.no_learning == true
        and .protocol.resolved_training_config.source == $config
        and .protocol.resolved_training_config.sha256 == $config_sha
        and .protocol.resolved_training_config.installed == [$config]
        and .protocol.suite.motion_file == $panel_motion
        and .protocol.suite.motion_count == 512
        and .protocol.suite.motion_keys_sha256
             == "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430"
        and .protocol.suite.split_linkage == "replicate-panel"
        and .protocol.suite.replicate_panel.receipt == $panel
        and .protocol.suite.replicate_panel.motion_key
             == "walk_hands_on_back_loop_002__A066_M"
        and .protocol.suite.replicate_panel.replicates == 512
        and .protocol.suite.replicate_panel.source_clip_sha256
             == "a7f10e7aa26e53cc4e346151d4ccd74e932e3aafa1cfaaac77dab8b8eec40929"
        and .protocol.suite.replicate_panel.alias_keys_sha256
             == "4b0fae026d8763e5cb1a39957ab8131e5372e1d47d4ec7e526791b76fe7f1430"
        and ((.runs | length) == 14)
        and (([.runs[].preset] | sort) == (expected_presets | sort))
        and ([.runs[] | select(
            .complete == true and .runtime.exit_code == 0
            and .mode == "lucid_rg"
            and .checkpoint_seed == $seed
            and .evaluation_seed == $eval_seed
            and .checkpoint == $checkpoint
            and .checkpoint_sha256 == $checkpoint_sha
            and .summary.motion_count == 512
            and (.summary.success_rate | type == "number")
            and (.summary.progress_rate | type == "number")
        )] | length == 14)
        and .checkpoint_sha256_before == .checkpoint_sha256_after
        and .checkpoint_sha256_before == {($checkpoint): $checkpoint_sha}
    ' "${receipt}" >/dev/null || die "historical seed ${seed} evaluation receipt differs"

    validate_historical_evaluation_evidence \
        "${receipt}" "${seed}" "${eval_seed}" "${expected_git}" "${bridge}" \
        "${checkpoint}" "${checkpoint_sha}" "${EXPECTED_PANEL_ALIAS_KEYS_SHA256}" \
        "${HIST_EVAL_ARTIFACT_ROOT}" "${HIST_EVAL_LOG_ROOT}" "${HIST_REPO}" \
        "${panel}" "${config}" "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}"

    # The analyzer also checks every run/aggregate metric pair.  Here, make
    # receipt reuse fail before another GPU cell if an installed config drifted.
    local installed
    while IFS= read -r installed; do
        assert_sha256 "${installed}" "${EXPECTED_LUCID_CONFIG_SHA256[${seed}]}"
    done < <(jq -er '.protocol.resolved_training_config.installed[]' "${receipt}")
    assert_sha256 "${checkpoint}" "${checkpoint_sha}"
    assert_no_write_bits "${checkpoint}"
    assert_sha256 "${bridge}" "$(frozen_input_sha "$(historical_key "${seed}" bridge)")"
    assert_sha256 "${panel}" "${EXPECTED_PANEL_SHA256}"
}

run_or_reuse_historical_evaluation() {
    local seed="$1"
    local eval_seed="$2"
    local receipt_dir="${HIST_EVAL_ROOT}/lucid_rg_s${seed}"
    local bridge config panel
    bridge="$(frozen_input_path "$(historical_key "${seed}" bridge)")"
    config="$(frozen_input_path "$(historical_key "${seed}" config)")"
    panel="$(frozen_input_path panel_receipt)"

    # Recheck every activation and freeze invariant before creating a marker or
    # touching the GPU; a long queue wait cannot silently widen the contract.
    assert_preregistered_state
    validate_h_r2_gate
    validate_panel_and_motion
    validate_all_historical_freezes
    mkdir -p "${receipt_dir}"

    local receipt
    if receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json' 2>/dev/null)"; then
        validate_historical_evaluation "${receipt}" "${seed}" "${eval_seed}"
        echo "reusing complete historical evaluation receipt ${receipt}"
        return
    fi
    if find "${receipt_dir}" -maxdepth 1 -type f \
        -name 'curriculum_robustness_ne512_*.json' -print -quit | grep -q .; then
        die "partial or ambiguous historical evaluation receipt set in ${receipt_dir}"
    fi
    if [[ -e "${receipt_dir}/.started" ]]; then
        die "historical evaluation was started without one valid complete receipt; preserve it, no resume or automatic retry is allowed"
    fi

    # Waiting is not a launched cell.  Reboots or interrupts while merely queued
    # must not burn the one-shot marker.  Revalidate the whole contract after the
    # wait, then atomically mark immediately before the evaluator process.
    wait_for_idle_gpu
    assert_preregistered_state
    validate_h_r2_gate
    validate_all_historical_freezes
    validate_staged_bundle "${seed}" true
    validate_panel_and_motion
    mkdir "${receipt_dir}/.started"
    python scripts/practice_utility/run_curriculum_robustness_eval.py \
        --training-receipt "${bridge}" \
        --training-config "${config}" \
        --panel-receipt "${panel}" \
        --num-envs 512 \
        --seeds "${seed}" \
        --modes lucid_rg \
        --eval-seed-base "${eval_seed}" \
        --max-delay 12 \
        --presets "${HIST_PRESETS[@]}" \
        --smpl-motion-file dummy \
        --artifact-root "${HIST_EVAL_ARTIFACT_ROOT}" \
        --log-dir "${HIST_EVAL_LOG_ROOT}" \
        --receipt-dir "${receipt_dir}" \
        --min-free-mib 6000 \
        --execute
    receipt="$(single_receipt "${receipt_dir}" 'curriculum_robustness_ne512_*.json')"
    validate_historical_evaluation "${receipt}" "${seed}" "${eval_seed}"
    validate_panel_and_motion
}

validate_bridge_analysis() {
    local analysis="$1"
    jq -e '
        .kind == "lucid_ratchet_historical_bridge_analysis"
        and .schema_version == 1
        and .claim_scope.classification == "posthoc_descriptive"
        and .claim_scope.binding == false
        and .claim_scope.alters_H_R2 == false
        and .claim_scope.inference == "none"
        and .claim_scope.noninferiority_claim_authorized == false
        and .claim_scope.superiority_claim_authorized == false
        and .activation.satisfied == true
        and .activation.h_r0_mechanism_pass == true
        and (.activation.h_r2_status_observed == "pass"
             or .activation.h_r2_status_observed == "fail")
        and .instrument_audit.passed == true
        and .instrument_audit.cell_count == 126
        and .instrument_audit.expected_cell_count == 126
        and .instrument_audit.per_mode_cell_count
             == {"fixed": 42, "lucid_ratchet_rg": 42, "lucid_rg": 42}
        and .instrument_audit.h_r2_parent_audit.cell_count == 84
        and (.inputs.h_r2_freeze_manifests | length) == 6
        and (.inputs.historical_robustness_receipts | length) == 3
        and (.inputs.historical_training_bridges | length) == 3
    ' "${analysis}" >/dev/null || die "historical bridge analyzer produced an invalid receipt"
}

run_analyzer() {
    local out="$1"
    shift
    local -a historical_receipts=("${@:1:3}")
    local -a h_r2_freezes=("${@:4:6}")
    local -a historical_bridges=("${@:10:3}")
    python scripts/practice_utility/analyze_ratchet_historical_bridge.py \
        --h-r2-analysis "$(frozen_input_path h_r2_analysis)" \
        --h-r2-amendment "$(frozen_input_path h_r2_amendment)" \
        --h-r2-freeze-manifest "${h_r2_freezes[@]}" \
        --historical-robustness-receipt "${historical_receipts[@]}" \
        --historical-training-bridge "${historical_bridges[@]}" \
        --out "${out}"
}

validate_analysis_inputs() {
    assert_preregistered_state
    validate_h_r2_gate
    validate_panel_and_motion
    validate_all_historical_freezes
    local seed receipt
    for seed in "${HIST_SEEDS[@]}"; do
        receipt="$(single_receipt "${HIST_EVAL_ROOT}/lucid_rg_s${seed}" \
            'curriculum_robustness_ne512_*.json')"
        validate_historical_evaluation "${receipt}" "${seed}" "$((seed + 100))"
    done
}

publish_exclusive_read_only() {
    local source="$1"
    local destination="$2"
    python - "${source}" "${destination}" <<'PY'
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
payload = source.read_bytes()
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
except BaseException:
    raise
PY
    chmod a-w "${destination}"
}

run_analysis() {
    local -a historical_receipts=()
    local -a h_r2_freezes=()
    local -a historical_bridges=()
    local seed mode key
    for seed in "${HIST_SEEDS[@]}"; do
        historical_receipts+=(
            "$(single_receipt "${HIST_EVAL_ROOT}/lucid_rg_s${seed}" \
                'curriculum_robustness_ne512_*.json')"
        )
        historical_bridges+=("$(frozen_input_path "$(historical_key "${seed}" bridge)")")
        for mode in fixed ratchet; do
            key="$(h_r2_freeze_key "${mode}" "${seed}")"
            h_r2_freezes+=("$(frozen_input_path "${key}")")
        done
    done
    [[ ${#historical_receipts[@]} == 3 ]] || die "exactly three historical receipts required"
    [[ ${#h_r2_freezes[@]} == 6 ]] || die "exactly six H_R2 freezes required"
    [[ ${#historical_bridges[@]} == 3 ]] || die "exactly three historical bridges required"

    # The analyzer reads the six exact H_R2 evaluation receipts from the
    # immutable parent analysis and adds exactly these three new receipts:
    # nine evaluation receipts / 126 cells in total.
    validate_analysis_inputs

    local candidate_dir candidate replay_dir replay
    candidate_dir="$(mktemp -d /tmp/lucid-ratchet-historical-candidate.XXXXXX)"
    candidate="${candidate_dir}/analysis.json"
    run_analyzer "${candidate}" \
        "${historical_receipts[@]}" "${h_r2_freezes[@]}" "${historical_bridges[@]}" \
        >/dev/null
    validate_bridge_analysis "${candidate}"
    # The analyzer and every input are mutable files until publication.  Close
    # the candidate-generation TOCTOU window before the O_EXCL claim write.
    validate_analysis_inputs
    if [[ -L "${HIST_ANALYSIS}" ]]; then
        die "historical bridge analysis path is a symlink: ${HIST_ANALYSIS}"
    elif [[ -e "${HIST_ANALYSIS}" ]]; then
        assert_regular_single_link "${HIST_ANALYSIS}"
        assert_no_write_bits "${HIST_ANALYSIS}"
    else
        publish_exclusive_read_only "${candidate}" "${HIST_ANALYSIS}"
        assert_regular_single_link "${HIST_ANALYSIS}"
    fi

    # Recompute after publication (and on every reuse) and compare all
    # scientific/provenance fields; only the wall-clock timestamp may differ.
    validate_analysis_inputs
    replay_dir="$(mktemp -d /tmp/lucid-ratchet-historical-replay.XXXXXX)"
    replay="${replay_dir}/analysis.json"
    run_analyzer "${replay}" \
        "${historical_receipts[@]}" "${h_r2_freezes[@]}" "${historical_bridges[@]}" \
        >/dev/null
    validate_bridge_analysis "${replay}"
    validate_analysis_inputs
    if ! cmp -s \
        <(jq -S 'del(.created_at)' "${HIST_ANALYSIS}") \
        <(jq -S 'del(.created_at)' "${replay}"); then
        die "immutable historical bridge analysis does not reproduce from exact frozen inputs"
    fi
    validate_bridge_analysis "${HIST_ANALYSIS}"
    assert_no_write_bits "${HIST_ANALYSIS}"
    rm -f "${candidate}" "${replay}"
    rmdir "${candidate_dir}" "${replay_dir}"
}

preflight() {
    assert_preregistered_state
    validate_h_r2_gate
    validate_panel_and_motion
    validate_all_historical_sources
    echo "historical ratchet bridge preflight passed; no write or GPU cell has started"
}

main() {
    preflight
    assert_preregistered_state
    mkdir -p "${HIST_FREEZE_ROOT}" "${HIST_EVAL_ROOT}"

    # All three staged checkpoint/true-config pairs are frozen and the original
    # bridge/curriculum/capsule evidence is re-hashed before any capability cell.
    freeze_or_reuse_historical 8600
    freeze_or_reuse_historical 8601
    freeze_or_reuse_historical 8602
    validate_all_historical_freezes

    # Serial, one-shot 14-cell ladders with the exact H_R2 seed mapping.
    run_or_reuse_historical_evaluation 8600 8700
    run_or_reuse_historical_evaluation 8601 8701
    run_or_reuse_historical_evaluation 8602 8702

    run_analysis
    mkdir -p "${HIST_ROOT}/.complete"
    jq '{claim_scope, activation, instrument_audit, descriptive_comparisons, collapsed_seed_interaction}' \
        "${HIST_ANALYSIS}"
}

# Activation is checked before sourcing the simulator environment, changing
# directories, creating output paths/markers, or touching the GPU.
: "${LUCID_RATCHET_HISTORICAL_BRIDGE_PREREG_SHA256:?set the future frozen historical-bridge preregistration SHA-256}"
# The environment bootstrap creates data-root directories, so the SHA-pinned
# preregistration and clean detached worktree gate must pass before sourcing it.
assert_preregistered_state
source "${HIST_ENV}"
cd "${HIST_REPO}"
export LUCID_GPU_WAIT_SECONDS=7200

if [[ "${1:-}" == "--preflight-only" ]]; then
    preflight
    exit 0
fi
main "$@"
