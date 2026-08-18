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

## Not yet built

Trainer/env callbacks that install the adapter into a live SONIC run, the
branch runner, proxy-feature extraction from rollouts, the utility estimator,
and the residual allocator. Those need GPU time; everything above is validated
on CPU first, deliberately.

## Running

```bash
source /data/robotixx/lucid-sonic/lucid_env.sh
python -m pytest tests/practice_utility/ -q          # 530 tests

python scripts/practice_utility/build_motion_pool.py \
    --pool-dir  /data/robotixx/lucid-sonic/pools/debug512/robot_filtered \
    --pool-id   debug512 \
    --output-dir /data/robotixx/lucid-sonic/manifests
```
