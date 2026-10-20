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
