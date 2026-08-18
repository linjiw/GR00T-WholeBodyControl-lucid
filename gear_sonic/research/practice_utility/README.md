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
| `rng_capsule.py` | Counter-based common random numbers; full RNG capture |
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

## Not yet built

Dev-suite `J_eff` evaluation (one pass per branch per horizon), the utility
estimator, and the residual allocator. The last two are **gated**: Gate B
authorizes them or nothing does.

## Running

```bash
source /data/robotixx/lucid-sonic/lucid_env.sh
python -m pytest tests/practice_utility/ -q          # 530 tests

python scripts/practice_utility/build_motion_pool.py \
    --pool-dir  /data/robotixx/lucid-sonic/pools/debug512/robot_filtered \
    --pool-id   debug512 \
    --output-dir /data/robotixx/lucid-sonic/manifests
```
