# Practice Utility

Measures whether extra practice on a training context causally improves later
deployment, rather than whether that context is currently hard. Implements
`lucid-design-implementation-plan.md`.

Everything here is **additive**: with no override armed, SONIC's code path is
unchanged, and `test_sampler_identity` enforces that.

## The chain

```
scan_pool ──► build_split ──► build_probe_manifest ──► [ paired GPU branches ] ──►
              (performer /      (stratified, frozen)     control vs intervention
               content)                                  from one capsule
                                                                │
   build_utility_record ◄── quality_metrics + latent_gap_probe ◄─┘
            │
            ├──► assess_identifiability   Gate A: is utility measurable at all?
            └──► assess_sufficiency       Gate B: is an estimator even warranted?
```

## Modules

| Module | Purpose |
|---|---|
| `schema.py` | Data contracts, keyed on stable motion hashes rather than batch-local ids |
| `motion_pool.py` | Parse BONES-SEED keys into performer / action / trajectory identities; family taxonomy |
| `split.py` | Group-disjoint, family-stratified splits over the leakage graph |
| `probe_manifest.py` | Stratified context selection, frozen before any branch runs |
| `intervention.py` | Local kernels, `(1-ε)ρ + εκ`, identity-preserving residual reweighting |
| `sampler_adapter.py` | Read/override SONIC's live bin distribution; realized-dose accounting |
| `rng_capsule.py` | Full RNG capture plus counter-stream primitives; production channels are not yet wired to those primitives |
| `branch_capsule.py` | Save/load/fork paired branches with provenance guards |
| `quality_metrics.py` | Physical-quality outcomes and quality-qualified success |
| `latent_gap_probe.py` | LUCID's temporal-VAE gap, as an audited predictor |
| `utility_label.py` | Paired evaluations → labels; **Gate A** identifiability |
| `proxy_audit.py` | Proxy scoring; **Gate B** sufficiency decision |
| `proxy_features.py` | Offline motion-structure descriptors (no fabricated contact) |
| `quality_telemetry.py` | Live quality collection from the simulator's contact sensor |
| `callbacks.py` | Trainer/env wiring; sampler snapshot; capsule saving |
| `run_log.py` | Parse training logs for parity, resume, and throughput checks |

## Decisions worth knowing

**Two split regimes, never one number.** BONES-SEED cannot support a split
closing both leakage channels at once — transitive closure puts 91% of the
4950-clip pool in one component. `performer` (unseen performers) and `content`
(unseen actions) are built and reported separately; `build_split` refuses a
degenerate combined split rather than returning one.

**Dose is counted in executed steps.** An episode terminating after three steps
must not score as full practice. `build_utility_record` refuses to emit a label
when the intervention branch received no more kernel exposure than the control.

**Efficacy cannot outvote a harm gate.** A breached gate makes a context
`harmful` outright, not a smaller positive.

**`J_eff` is a macro-mean over families.** A method that helps common motions
and harms rare ones must not read as an improvement.

**Correlations are computed within group, then averaged.** A proxy held constant
within each policy stage scores pooled Spearman 0.87 purely because both it and
utility rise with training; grouped it correctly scores 0.0.

**The latent gap is a predictor, not the scheduler.** Scoring a curriculum with
the quantity that drives it makes any improvement partly definitional.

**Gate B is hard to pass toward more machinery.** Any sufficient simple proxy
blocks the estimator — and that outcome is a publishable result.

**The current randomness contract is stochastic, not channel-wise CRN.** The
symmetric-restart identity test proves that two unchanged fresh restarts from one
full capsule agree. Once an intervention changes the trajectory, however, the live
friction, push, action-noise, and minibatch channels do not call the counter-stream
primitives. Claim-bearing probe receipts must therefore say
`stochastic_potential_outcomes_no_channelwise_crn` and estimate a same-estimand
noise floor; they must not infer matched random channels from capsule metadata.

## Verified live

Paired control/intervention branches ran inside real SONIC training. The
intervention boosted its target bin to sampling probability 0.146 against 0.0068
for the highest non-kernel context (21×), both radius-1 neighbours boosted, the
distribution still summing to 1.000000; the control drew 0 episodes on the target
motion and the intervention drew 5, for a realized extra kernel dose of 1.736.

Four defects surfaced only in live runs, each now covered by a test that pins the
upstream reality:

1. `_motion_fps` is a per-motion **tensor** upstream, not a scalar — a scalar
   fake made a fatal error invisible to the whole CPU suite.
2. SONIC loads resident motions **with replacement**, so one global bin occupies
   several active positions (18 of 535 observed).
3. A snapshot at install captures the sampler's **prior** — all failure rates
   read exactly 1.0, leaving a campaign with no difficulty axis. Hence
   `snapshot_at_step`, and a hard refusal of degenerate candidate sets.
4. A capsule storing the **combined** `model.state_dict()` cannot seed a branch,
   only archive one. Hence the split policy/value layout and
   `export_sonic_checkpoint`.

## Measured

| | |
|---|---|
| throughput | 70.7 s/iteration at `num_envs=256` (87 env-steps/s) |
| scaling | iteration time nearly flat in `num_envs` — horizon, not env count, drives cost |
| adaptation transient | reward 0.48 → 17.83, episode length 13 → 223 over 24 iterations |
| frozen campaign | 24 contexts, all 4 failure quartiles, 10 families, **31.4 GPU-hours** |

## Not yet claim-ready

The historical probe manifest has no matched settled origins. The hardened path
now creates hash-bound origin maps, rebuilds a manifest only from the origins'
common resident contexts, and audits a separate preregistration bundle before any
branch can be called claim-grade. Still required are frozen per-context latent-gap
features, same-estimand noise, deployment-side `J_eff`, calibrated directional
proxy testing, and a receipt-producing campaign launcher. The utility estimator
and residual allocator remain **gated**: inverse estimator authorization permits
them or nothing does.

## Running

```bash
source $LUCID_ROOT/lucid_env.sh
python -m pytest tests/practice_utility/ -q

python scripts/practice_utility/build_motion_pool.py \
    --pool-dir  $LUCID_ROOT/pools/debug512/robot_filtered \
    --pool-id   debug512 \
    --output-dir $LUCID_ROOT/manifests
```

### Horizon-scaling training

`run_curriculum_comparison.py` remains the compatible single-budget mechanism
driver. Paper-facing horizon scaling uses
`run_curriculum_horizon_scaling.py`, whose matrix is fixed to 51 new branches:
seeds 8603–8604 at 32 iterations and seeds 8600–8604 at 64, 128, and 256
iterations, with `lucid`, `fixed`, and `off` at every cell.

The first command is deliberately a dry run. It requires a clean Git tree,
exclusively reserves the campaign paths, audits the sealed historical 32-step
receipt, and writes the immutable preregistration plus an initially incomplete
training index. It does not query the GPU.

```bash
python scripts/practice_utility/run_curriculum_horizon_scaling.py \
    --campaign-id curriculum_horizon_scaling_v1_20260826
```

After reviewing that preregistration, the same arguments can execute only via
the frozen resume path:

```bash
python scripts/practice_utility/run_curriculum_horizon_scaling.py \
    --campaign-id curriculum_horizon_scaling_v1_20260826 \
    --resume --execute
```

Execution refuses a GPU with any compute process, more than 5% utilization, or
less than 28,000 MiB free in any of three samples. Every retry gets a fresh,
collision-exclusive attempt directory. Mutable status and budget receipts are
atomically updated after every transition; resume re-hashes all completed
artifacts before skipping them. The combined index points the existing frozen
evaluator to one legacy-shaped receipt per budget. The 32-step receipt retains
the historical and new launcher lineages explicitly: its commands and configs
are audited as equivalent, but their launcher byte hashes differ.

The launcher does not yet enforce disk headroom. Before execution, verify at
least 90 GiB free on the artifact filesystem for the approximately 74 GiB
51-branch matrix; adding an immutable storage gate remains a launch blocker.
