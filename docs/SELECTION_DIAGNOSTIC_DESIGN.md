# Selection controls and training diagnostics

This adds a fresh development experiment. Historical 100-example runs and their
execution identities are unchanged. There is no teacher client, API generation,
final-test scoring, automatic early stopping, rollback or best-checkpoint search.

## Primary comparison

Random selection, training-loss category feedback, and curriculum-validation
category feedback choose from the **same frozen bank**. Preparation can build
that bank from the frozen Dolly training pool, retaining reference answers, or
use an existing verified synthetic export. Both inputs pass format, protected
split leakage, full-answer encoding and duplicate-prompt gates.

For each seed all three arms receive the same starting model/adapter seed,
identical warmup IDs in identical order, update count, microbatch count, complete
answer-token exposure at every update, optimizer lifecycle, learning-rate
schedule and evaluation milestones. The replay-off comparison has no repeated
candidate IDs. Its default is 25 updates of four examples: 100 unique examples,
including eight shared warmup examples. Each selected example is presented once;
this is deliberately a new diagnostic recipe, not the historical three epochs.

### Exact matching without cutting answers

Preparation freezes a schedule of **exact answer lengths**. At a slot requiring
70 answer tokens, all arms choose an unused candidate with exactly 70 supervised
tokens. This preserves complete answers and matches answer tokens and optimizer
updates simultaneously. Total tokens are determined at preparation and printed
per seed; they are not an arbitrary 20,000-token request. Budgets match within a
seed, while different seeds may have different schedules/totals. Report paired
differences within seeds rather than pretending every seed had the same budget.

This estimates selection quality **conditional on answer length**. It restricts
the original TEFLD selector. A small bank may leave few choices, or categories
may be confounded with length. Per-slot eligible candidate/category counts and
single-choice/category-choice fractions expose that limitation. Prefer the
Dolly pool for the first experiment; inspect these counts before interpreting
a synthetic-bank result. Never silently shorten an answer, substitute categories
or generate extra data to satisfy this control.

Within the eligible length group, random chooses uniformly over candidates.
Feedback methods choose a category proportional to its positive loss, then
choose uniformly within that category. Loss feedback scores only exposed
training examples; validation feedback reads only the frozen curriculum view.
The fixed training probe is observer-only. Random selection never consumes
any feedback score. All categories' observations are still logged for analysis.
This is a category-feedback ablation, not an implementation of gradient scoring.

## Replay ablation

`--replay-ablation` prepares a 3-by-3 factorial: each selection policy with replay
off, random replay, and highest-training-loss replay. All nine arms have the
same update/answer-token schedule. At predeclared first-slot opportunities, replay
arms replace a new example with a same-length previously exposed example. The
common warmup ensures availability; no repeated ID is allowed inside one update.

Replay ranking uses training-only loss, refreshed after each round; newer exposed
items use their observed training loss until that refresh. Every replay trace
records the actual candidate and exposure. Compare random versus off within a
selection policy for replay's effect, and guided versus random for replay
prioritization's effect. Reports also retain selection differences with replay
off. Unique-example counts can differ in replay arms while resources match.

This replay control does **not** reproduce the older 0.35 threshold, two-use cap
and cooldown rule. It is a separately identified history-replay experiment.
Preparation rejects replay studies with no available matching-length opportunity.

`prepare_replay_study.py` extends a completed selection-only study in a separate
directory. Before reusing no-replay controls it checks code/bank/protocol
fingerprints, each completed state's original plan identity, total and per-update
resources, no-repeat traces, warmup IDs, initial evaluations, executed LR/reset
traces, evaluation row IDs and adapter integrity. The copied states retain their
source-state fingerprint and original identity in `reused_control`; the new plan
records all control lineage. Scores are imported unchanged. Adapters remain at
their original paths, so retain the source directory. No new evaluations, model
loading or provider calls occur during preparation.

The first pilot uses seed 42, with three imported controls and six new runs,
each of 25 updates and 18,422 answer-token exposures. It is a development
investigation chosen after inspecting the earlier selection-only results.
Record that sequencing; do not call it an independent confirmatory study.

## Diagnostics

Before any training, and after the shared warmup and each fixed update milestone,
every arm scores the same fixed training probe, curriculum view and checkpoint
validation view. The default is eight examples per category (64 rows) in each
view, with disjoint frozen role membership. Probe exposure membership is recorded:
the training probe is not advertised as an unseen validation set. These evaluations
use teacher-forced answer NLL; they do not measure free-generation correctness.
The identical warmup examples also form a fixed exposed-training probe within
each seed. Its loss curve tracks memorization on a stable set even when a
synthetic candidate bank has no overlap with the original-Dolly training probe.

Logs contain macro and token-weighted losses, per-category and per-row scores,
actual tokens/updates, category token shares, source origin, new/replay IDs,
choice constraints, effective LR, gradient norm before clipping, clipping events,
optimizer resets and round timing. The executed step uses the same target-weighted
loss, AdamW settings and clipping threshold as the original engine. Nonfinite
losses or gradients abort the update; failed rounds never advance committed state.

`reset-per-round` preserves the original reset behavior as a shared control;
`continuous` is an explicit separate optimizer-lifecycle experiment. The latter
persists optimizer and scheduler state at every checkpoint. Both model and
optimizer checkpoints are fingerprinted. Changing lifecycle requires a fresh
study directory. This comparison changes optimizer history **and the LR schedule
lifecycle together**; it does not isolate Adam moments alone.

### Small optimizer pilot

`prepare_optimizer_study.py` prepares a **single seed-42 random-selection,
replay-off continuous run**, reusing the completed reset run only as a measured
comparison. Its schedule and bank are copied without retokenizing or generating
data. Compatibility gates verify source code, protocol, bank, control state,
warmup, per-update resources, evaluation IDs and adapter integrity. Control state
and plan fingerprints are retained. No trained weights or optimizer state from
the reset control initialize the new run: it starts with the same base and seed.

Use the dedicated `run_continuous_optimizer.ps1` launcher to avoid starting the
full three-selection sweep. It performs one 25-update, 100-example run with
18,422 answer tokens. Random selection fixes candidate IDs/order independently
of gradients and feedback. The report rejects any executed-data/order mismatch
and compares only the final matched-budget checkpoint. A one-seed result applies
to this optimizer/LR recipe and fixed curriculum, not automatically to adaptive
TEFLD or other seeds. This bounded pilot reflects the user's hardware constraint.

Only the final fixed-budget checkpoint is compared. Intermediate checkpoint
scores are diagnostic and cannot choose training examples, stop training, tune
the current run or select a favored checkpoint. The final test stays locked.
These existing development rows have already been examined, so they should not
be represented as a new untouched holdout. Three default seeds provide paired
differences and sample standard deviations, not a proof of generalization.

## Outputs and review

`plan.json` freezes source/bank/model/code fingerprints, seeds, budgets, initial
batches and evaluation rows. Each arm commits a `state.json`, adapters, optional
optimizer state and failure metadata. Raw bank content stays under ignored
`outputs/`; publish only reviewed numeric summaries.

The report includes fixed-probe and validation curves against tokens and updates,
gradient/LR plots, final scores, category/replay accounting, paired seed summaries
and replay contrasts. Inspect choice capacity and actual exposure before drawing
conclusions. Diagnostics suggest hypotheses; controlled contrasts test them.

`selection_report.py --existing-comparison ...` reconstructs available exposure,
replay and curriculum summaries from completed historical outputs. It explicitly
marks missing base-model scores, fixed probes, gradient norms and executed LR
traces as unavailable; it does not invent those measurements or score the model.
