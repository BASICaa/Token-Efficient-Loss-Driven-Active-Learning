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
