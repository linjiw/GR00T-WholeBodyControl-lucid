# Practice Utility — T0 infrastructure

Implements the measurement layer for `lucid-design-implementation-plan.md`
(Part I §4–5, amended by Part II). Everything here is **additive**: with no
override armed, SONIC behaves exactly as upstream.

## Modules

| Module | Purpose | Plan ref |
|---|---|---|
| `schema.py` | Data contracts: `ContextKey`, `MotionPoolManifest`, `SamplingSnapshot`, `DoseReport`, `RngReceipt`, `BranchCapsule`, `HarmVector`, `UtilityRecord` | §4.1, §4.3, §4.4 |
| `intervention.py` | Local kernels, `(1-ε)ρ + εκ` mixing, identity-preserving residual reweighting with KL radius, coverage floors, probability-ratio caps | §4.2, §4.8 |
| `sampler_adapter.py` | Reads and optionally overrides SONIC's live bin distribution; realized-dose accounting | §5.2, §4.3 |
| `rng_capsule.py` | Counter-based common random numbers + full RNG capture/restore | §5.4 |
| `branch_capsule.py` | Save/load/fork paired branches with provenance guards | §5.3 |

## Invariants the tests enforce (203 tests, CPU only)

- **Pass-through identity.** No override armed ⇒ `apply()` returns the *same tensor object*; dtype and device preserved.
- **ε = 0 identity.** Arms the override path but leaves the distribution unchanged — this is what makes the ε=0 branch a valid noise-floor measurement rather than a different code path.
- **α = 0 / constant-score identity.** The residual curriculum returns the base distribution exactly.
- **Kernels never cross a motion boundary.** Neighbouring bins of a different clip are not a neighbourhood.
- **KL radius, coverage floor, and `max_prob_ratio` bounds actually hold** (asserted on `q/base`, not merely "smaller than uncapped").
- **Dose is counted in executed steps**, so an episode that terminates after 3 steps does not count as full practice.
- **Stale kernels are detected** after a motion-library reload rather than silently mis-attributing dose.
- **Forked capsules are identical** in model, optimizer, sampler, env, RNG, and provenance; only `branch_id`/`role` differ.
- **Resume against a different motion pool or config is refused.**
- **Upstream contract test** re-reads `MotionLibBase`'s real source and fails if an attribute the adapter depends on disappears.

## Not yet built (next)

Quality-metric evaluator, LUCID latent-gap/VAE probe, branch runner + trainer
hooks, utility-label builder, proxy audit, estimator, residual allocator.

## Running

```bash
source /data/robotixx/lucid-sonic/lucid_env.sh
python -m pytest tests/practice_utility/ -q
```
