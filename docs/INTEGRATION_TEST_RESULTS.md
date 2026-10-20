# Integration verification

Checked October 19, 2026 in the `hf-agents` Conda environment.

| Check | Result |
| --- | --- |
| Root beta/application tests | 18 passed |
| Controlled experiment tests | 150 passed, 1 skipped |
| Expected skip | Historical raw-workspace preparation test; private run inputs are not distributed |
| Optional local validation importer | Reuse without generation; fingerprinting; reject changed sets and empty references |
| Local validation-guided curriculum | Exact 100-example fixture run, independent validation access and complete epochs |
| Training settings and round events | Validated settings, accumulation applied, summaries reconciled with committed state |
| Existing generated data | Three banks exported locally: 130 accepted candidates each; no new generation or verification calls |
| Published snapshot | Five completed arms; 100 unique examples each; identical 64-row evaluation contracts; scores and costs reconcile |
| Graphs | Four PNG figures and four SVG exports, rebuildable from numeric snapshot |
| Publication checks | No credentials, model weights, raw training examples or runtime caches staged; new documentation links resolve |
| Application changes | Optional validation-file import; no automatic replacement of the active external checkout |

Commands were run in separate processes to isolate the two packages named `TEFLD`:

```powershell
conda run -n hf-agents python -m pytest tests -q
cd experiments/controlled
conda run -n hf-agents python -m pytest tests -q
```

The controlled suite reported third-party SWIG deprecation warnings. Tests use
fixture models; no real Gemma training, new paid API calls, model downloads or
final-holdout scoring were performed during integration. The active external
beta checkout and its uncommitted work remain outside this commit.

## October 20: opt-in bank replenishment

The controlled suite now passes **165 tests, with the same one historical-input
skip**. Fifteen new offline tests cover quota/RNG parity with the frozen engine,
generation only for category deficits, durable admission reuse, interrupted
replenishment without state advancement, verification rejection, explicit budget
requirements and separate continuation identities. The root application was not
changed or retested in this update.

The recovery PowerShell launcher passed syntax checking and a fully mocked
six-result workflow. Actual preparation (no training, weights or paid requests)
created a separate 94-example GPT-6.1 loss continuation and a fresh Luna recovery
plan. The source loss state fingerprint still matches its pre-fork fingerprint.
The original engine and `run_controlled.py` were not changed, so existing
fixed-bank plans retain their original execution identities. The recovery
subclass pins its additional code in a new identity and records source lineage.

The user authorized a new shared ceiling of **900,000 toman** for these recovery
arms. Actual API availability, credit, real-model continuation and resulting
quality remain untested; paid calls and training are left to the user's command.

The credential bootstrap has seven additional offline tests covering explicit
`.env` loading, BOM/quoted values, process precedence, missing/empty keys,
suppression of key output, and forwarding only to the recovery entry point.
It is separate from the fingerprinted training code so prepared recovery plans
and saved adapters retain their existing identities.

## October 20: matched selection and diagnostics

The controlled suite passes **184 tests, with the same one historical-input
skip**. Twelve new tests cover exact full-answer token/update matching across all
nine selection/replay combinations, identical warmup batches, random-feedback
isolation, guided replay ranking, parity of instrumented updates with the
original optimizer, nonfinite rejection, fixed probes and final-test isolation,
completed-run reuse, bank tamper rejection, resource-report integrity and
historical reporting including Windows UTF-8 BOM input.

Both reset-per-round and continuous optimizer/scheduler lifecycles pass exact
CPU-fixture interruption/restart checks. After the full-suite run, the new
12-test suite was rerun for the fixed shared-warmup probe. Its end-to-end test
was additionally rerun with figure generation enabled; learning curves and
gradient/LR plots were visually inspected. These are fixture checks, not new
Gemma results. The root application and original controlled engine were unchanged.

The real local-data **preparation only** produced a fresh Dolly bank of 12,641
usable, distinct candidates, 25 full-answer updates and 100 unique examples per
replay-off arm. Seeds 42/43/44 have frozen token budgets 18,422/15,978/16,309,
respectively; all methods within each seed share the same budget. The plan
fingerprint matches the finalized code. It starts no model training or API calls.
Exact-length constraints leave 13/8/10 post-warmup slots with only one eligible
candidate out of 92 slots per seed; report choice capacity before interpreting
selection differences.

A separate report was reconstructed from all six completed historical runs,
including category/replay token exposure and available curriculum curves.
Missing initial scores, fixed-probe curves and optimizer-health traces are
explicitly unavailable. Graph colors were checked for consistent category
mapping. Runtime banks, prepared plans, local paths and graphs remain under
ignored `outputs/`; only code, tests and design documentation are published.

The new diagnostic is a one-pass, length-conditional development ablation,
with optional history-replay contrasts. It does not claim equivalence to the
historical three-epoch recipe or its threshold/cooldown replay policy. Actual
quantized-Gemma training and diagnostic quality remain to be measured by the user.
