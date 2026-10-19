# Integration review: controlled validation experiments

This branch starts from committed `beta/adaptive-validation-difficulty`
(`cc417d1`). It preserves existing default application behavior and imports the
comparison project's experiment engine under `experiments/controlled/`.
The active external beta checkout and its uncommitted files are not rewritten,
committed or included implicitly.

## Imported and adapted now

| Feature | Source | Integration |
| --- | --- | --- |
| Dataset-only loss/validation curricula | Comparison engine | Original-reference retrieval, explicit method selection, no teacher network |
| API loss/validation curricula | Comparison engine | Verified live generation or explicit saved-bank selection |
| Split-role protections | Comparison engine | Independent curriculum/comparison validation; protected final test |
| Encoding and finite-update safety | Comparison engine | Complete answer targets, prompt parity, finite-loss/gradient guards |
| Recoverable runs | Comparison engine | Atomic state, writer locks, adapter weight hashes and identity checks |
| Token/example/replay/cost accounting | Comparison engine | Explicit controls and compact per-round event files |
| Configuration validation | Reviewed application training-config design | Adapted to protocol-owned settings, independent of mutable section state |
| Optional local validation-file import | Reviewed active beta checkout | Added to the root application with fingerprinting and replacement protection; default generation unchanged |
| Structured round logging | Reviewed application logging design | Adapted to regenerate summaries from authoritative committed run state |
| Existing generated examples | Comparison admission records | Export tool preserves passing verification and provenance; quota exhaustion stops without generation |
| Measured results | Completed comparison study | Sanitized numeric snapshot, charts and limitations |

The two packages both use the name `TEFLD`; their processes must be launched
from their own working directories. Their schemas and `validation.py` modules
have different responsibilities, so files are not copied over the application.

## Application features preserved for later controlled evaluation

The beta application retains its shared-validation generation, task-shape
weakness profiles and difficulty controller. The external active checkout also
contains uncommitted configuration, logging and adaptive replay work. Those
files remain in that checkout; selected configuration/logging ideas and the
local validation importer are adapted explicitly rather than publishing its
uncommitted files wholesale.

The application importer is exposed as `run_tefld.py --validation-file FILE`.
It stores a fingerprint, rejects missing reference answers, reuses the same
frozen set without generation, and refuses to replace an existing different
generated or imported validation set. It does not disable training generation
or turn the beta loop into a dataset-only trainer.

Adaptive difficulty, history-relative replay thresholds and checkpoint rollback
change the experiment itself. They are not enabled in the controlled comparison
by default or claimed as explanations of the measured results. Promotion should
use explicit configuration flags, separate ablations and independent validation.

## Follow-up before merging application and experiment internals

The reviewed external application still has paths needing regression fixes:
answer truncation can remove supervision; build-only can fall through with a
cached batch; adapter-completeness checks do not verify weight contents; finite
validation filtering can change the scored sample set; curriculum decisions and
rollback currently depend on shared validation. The controlled engine has
separate implementations and protections, but this import does not claim to
have repaired the external application's uncommitted code.

The external checkout passed 24 existing offline tests during review. Those tests
do not establish complete coverage of the issues above. Root beta tests and the
controlled engine's independent regression suite are run separately for this PR.

## Review criteria

- Preserve the beta application's committed behavior and external uncommitted work.
- Exercise local validation-guided training and cache-only synthetic selection
  with fixtures; no real Gemma training or paid requests during integration.
- Keep old measured results immutable, with explicit code fingerprints and no
  fabricated local-validation result.
- Export readable, regenerable figures and verify numeric snapshot consistency.
- Exclude credentials, model weights, raw dataset content and runtime caches.
- Open a reviewable PR against the beta branch. Main/beta merges wait for user
  acceptance; this integration does not merge either branch automatically.
