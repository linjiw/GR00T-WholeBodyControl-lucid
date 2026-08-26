# Track A claim-readiness debug

- Started: 2026-08-26 10:48 America/New_York
- Mode: `autoresearch debug`
- Iteration budget: 5
- Objective: make the frozen `probe_screen_v1_late` campaign claim-grade runnable
  without changing its scientific treatment, opening the final split, or crossing Gate B.
- Keep metric: a dry-run campaign receipt must prove symmetric capsule restart, settled
  origin, last-4 efficacy, frozen Gate A/B thresholds, per-branch provenance, and a real
  frozen latent-gap feature path; focused CPU tests must pass.
- Discard condition: any change that changes the frozen intervention, consumes final-test
  data, substitutes training reward for the claimed deployment estimand without an explicit
  caveat, or builds gated estimator/allocator machinery.

## Iteration 0 — audit

Status: baseline rejected for launch.

Findings:

- `run_branch.py` starts from a checkpoint only and does not restore a capsule.
- It has no campaign orchestrator, capacity gate, log capture, or JSON branch receipts.
- It does not attach the observer, so it cannot provide the claimed frozen latent-gap proxy.
- `build_utility_labels.py` currently sees only manifest-side failure/sampling/motion
  features and defaults to training reward rather than claim-grade deployment `J_eff`.
- Gate A reads the older epsilon-zero paired-delta report rather than the measured settled
  cross-seed last-4 floor.
- The code's Gate B means "no simple proxy is sufficient, therefore an estimator is
  authorized," while `fable.md` uses "Gate B passes" to mean the latent gap predicts
  utility. This semantic mismatch must be resolved in receipts and paper language.

Evidence: 978 focused tests passed; the GPU was idle with 30,607 MiB free; no probe branch,
utility-label, Gate A, or Gate B receipt exists as of this audit.

## Iteration 1 — manifest-aligned origin creation

Status: kept.

Added `create_probe_origins.py`, which resumes the established settled step-24 checkpoint,
creates one full capsule/checkpoint/sampler snapshot per seed at the same absolute step,
and freezes a two-window stability rule before execution. A later manifest must be selected
from the context intersection across those snapshots. The driver remains dry-run by default
and explicitly leaves latent-feature, J_eff, and Gate A artifacts unverified.

Verification: the final origin suite has 13 focused tests; an actual-checkpoint dry run
resolved source step 24 and produced the expected absolute target step 36 command.

## Iteration 2 — provenance falsification

Status: kept.

The first origin builder still trusted two self-reported facts: its motion-tree paths and
the sampler timeline rate. The hardened version resolves the robot symlink to the exact
pool `source_root`, hashes both the 512-file robot and SMPL trees, requires identical
motion-key support, forces Hydra's live `target_fps`, and makes the callback verify that
value before snapshotting. Claim-grade manifest creation now rescans the same source
bytes and refuses a dirty worktree.

Verification: real debug512 robot/SMPL support is 512/512; pool rescan is exact; dry run
is 24→36 with the live target-FPS override.

## Iteration 3 — estimand falsification

Status: kept as a blocker, not repaired by assumption.

The frozen screen's two shared controls do not carry a target kernel, so their live dose
reports cannot measure per-context baseline completed-kernel steps. The advertised
realized-extra-dose denominator was therefore not executable. A second audit found that
the directional latent test was only a method-name string. Preflight now remains blocked
on both facts, and the low-level branch runner can execute only with an explicit
`--exploratory` acknowledgement.

## Iteration 4 — false claim-output falsification

Status: kept.

The label builder previously could call aggregate inputs `claim_grade` without a ready
preflight or policy/evaluation lineage. It now recomputes manifest/context identities,
then emits only a machine-readable blocked receipt and returns 2. It never reads doses,
assembles labels, or runs Gate A/B until the shared-control, H_l policy/capsule,
dev-suite, physics-seed, per-evaluation receipt, and latent-calibration links exist.

Final verification: 1,047 practice-utility tests pass; targeted Black, isort, Ruff, and
`git diff --check` pass. The objective is intentionally incomplete: origin generation is
ready, while the claim screen is correctly blocked instead of falsely runnable.
