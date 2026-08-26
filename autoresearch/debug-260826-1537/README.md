# Track B horizon-scaling launch hardening

- Started: 2026-08-26 15:37 America/New_York
- Mode: `autoresearch debug`
- Iteration budget: 4 CPU-only iterations
- Objective: make the exact Track-B horizon matrix safely preregistrable and resumable
  without launching a simulator job or changing the legacy single-budget interface.
- Keep metric: one immutable preregistration must freeze all 51 new branches and every
  scientific input before any GPU query; focused tests and style checks must pass.
- Discard condition: any design that silently merges incompatible historical lineage,
  overwrites an existing campaign, skips unverified artifacts on resume, or permits a busy
  shared GPU.

## Iteration 0 — launch audit

Status: baseline rejected for claim launch.

The single-budget driver supports seeds and modes but only one iteration budget. It writes
its only receipt after every arm, has no immutable preregistration, resume, collision
exclusion, or code/input hashes, and gates only on six GiB of free memory. The existing
evaluator accepts one training receipt, so a separate two-seed step-32 receipt cannot by
itself extend the historical three-seed cell. The historical launcher SHA also differs
from the current bytes. An active foreign Isaac process demonstrated that the old
free-memory gate would permit a contended launch.

## Iteration 1 — deterministic preregistration and index

Status: kept.

Added `run_curriculum_horizon_scaling.py`. Its frozen matrix is 32 iterations for seeds
8603–8604 and 64/128/256 for seeds 8600–8604, across all three modes: 51 branches and
6,912 iterations. A clean-tree dry run exclusively reserves its paths, hash-binds the
checkpoint, encoder, resolved source config, launch code, and six-channel configs, audits
the sealed historical commands/config/artifacts, then writes the immutable
preregistration before any GPU query. Per-budget receipts retain the legacy evaluator
interface, while one combined index selects the five-seed step-32 receipt.

## Iteration 2 — incremental failure and resume safety

Status: kept.

Each branch attempt now has a fresh exclusive directory. Status, budget receipts, and the
combined index are atomically rewritten at every state transition. Blocked, failed, and
interrupted attempts remain visible. Resume re-hashes every completed artifact and checks
capsule branch identity/global step before skipping work; a stale `running` attempt becomes
`interrupted` and retries without appending into its old outputs.

## Iteration 3 — idle-GPU and audit boundary

Status: kept.

Execution is impossible in the preregistration invocation: `--execute` requires a later
`--resume`. The resume arguments cannot weaken the frozen GPU gate. Before every branch,
three samples must each show at least 28,000 MiB free, at most 5% utilization, and zero
compute processes; a cooperative campaign lock prevents two instances of this launcher.
The training preregistration records the qualitative hypothesis but deliberately leaves
the numerical non-inferiority margin for the separate deployment-evaluator
preregistration, before any deployment outcome is opened.

Verification: 14 focused tests pass; targeted Ruff and `git diff --check` pass. No GPU
query or experiment launch occurred. The tree remains intentionally uncommitted for the
parent agent to review. A concurrent full-suite run reached 1,017 passes but was not a
valid Track-B regression verdict: 16 failures and 45 setup errors came from in-progress
Track-A sampler edits (`FakeMotionLib` lacked their new global frame-to-bin registry).
That owning agent was notified; rerun the full suite after the shared edits settle. The
combined step-32 receipt now exposes separate Git/launcher strata plus historical command
and artifact hashes. Automated disk-headroom enforcement remains a launch blocker.
