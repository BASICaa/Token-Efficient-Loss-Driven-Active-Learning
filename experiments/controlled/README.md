# Controlled TEFLD experiments

This engine is isolated from the beta application's section-based `TEFLD`
package. Run the commands below **from this directory**. This integration
changes code; the published results describe the earlier measured version,
whose core hash is recorded in the [numeric snapshot](../../results/five-way-100/snapshot.json).

## Included methods

| Mode | Training source | Curriculum feedback | Teacher calls |
| --- | --- | --- | --- |
| `dolly` | Original Dolly references | Balanced seeded sampling | None |
| `local-loss` | Original Dolly references | Refreshed training losses | None |
| `local-validation` | Original Dolly references | Separate curriculum validation | None |
| `api-loss` | Verified synthetic examples or saved bank | Refreshed training losses | Explicit opt-in for live generation |
| `api-validation` | Verified synthetic examples or saved bank | Separate curriculum validation | Explicit opt-in for live generation |

`local-validation` is implemented and fixture-tested. It is **not** the local
method measured in the published five-method study; that study used local loss
feedback. No new real training has been performed for this integration.

The beta application's `--validation-file` option controls whether its validation
examples are supplied locally or generated through an API. It does not switch
its training source to dataset-only training. These controlled modes declare
training source and curriculum feedback separately.

## Environment and local inputs

Use the existing `hf-agents` Conda environment. The pinned runtime in
`requirements-experiments.lock` describes the measured Windows/CUDA setup.
Install the matching CUDA Torch wheel before the other pinned dependencies.
Plotting additionally requires `requirements-results.txt`.

Supply local Gemma-compatible model files and the frozen Dolly JSONL yourself.
Automatic model downloads are disabled. The frozen split manifest is included;
the dataset must match its declared checksum:

```text
9765d49979c37d4b96a51651c4cfb8c399169390b6f751ca43e198355eedeb2a
```

A differently serialized or updated dataset will fail closed. Do not change a
hash to make it pass and describe the result as the same study. The legacy
protocol templates retain historical path fields; `run_controlled.py prepare`
replaces those fields with the explicit local paths below.

## Prepare local validation-guided TEFLD

```powershell
conda activate hf-agents
python run_controlled.py prepare --mode local-validation --dataset "D:\path\to\dolly_train.jsonl" --model "D:\path\to\gemma-4-E2B-it" --examples 100 --full-validation --output outputs/local-validation
```

Preparation loads a local tokenizer and validates a training-only warmup. It
does not load model weights, train or make API requests. The common ten-reference
warmup counts within the 100-example training target.

`--full-validation` uses all 256 curriculum-validation and all 256 comparison-
validation examples in their separate partitions. Without it, the earlier
eight-per-category (64-row) views are used. The final-test partition remains
locked. Using more validation examples defines a new experiment; it does not
change the old published 64-row results.

The following command **starts local training** when the user runs it:

```powershell
python run_controlled.py run --output outputs/local-validation
```

Use `--mode dolly` or `--mode local-loss` and separate output directories to
prepare the corresponding controls. Settings, model identity, warmup and data
hashes are recorded; changed settings require fresh output. The preparation
tool uses seed 42 by default; `--seed` declares a different fresh run.

## Reuse already generated data

Export each teacher's saved accepted admissions as a separate private bank:

```powershell
python export_candidate_bank.py --admissions "D:\path\to\candidate_admissions.json" --output outputs/banks/sol61-validation.json
```

The exporter preserves reference answers and passing verification metadata,
records a source fingerprint and prints category coverage. It performs no
generation, inference or paid verification. Exported training text belongs in
ignored `outputs/`, not in the public result snapshot.

Prepare an API-derived validation selection experiment, then explicitly use
the saved bank:

```powershell
python run_controlled.py prepare --mode api-validation --dataset "D:\path\to\dolly_train.jsonl" --model "D:\path\to\gemma-4-E2B-it" --examples 100 --full-validation --output outputs/bank-validation
python run_controlled.py run --output outputs/bank-validation --candidate-bank outputs/banks/sol61-validation.json
```

Saved-bank execution disables the teacher network. Local format, encoding and
protected-split leakage checks run again before training. Missing category
coverage stops the round; there is no automatic generation fallback. This
tests **selection over fixed synthetic data**, not live adaptive generation.
The bank path starts from a bank-selected initial batch; the prepared shared
Dolly warmup is used by dataset and live-generation paths, not the fixed-bank
path. Compare like-for-like initial batches when designing a bank ablation.

## Explicit live generation

With `api-validation` or `api-loss`, omitting `--candidate-bank` requires both
`--allow-paid-api` and an explicit total `--teacher-budget-toman`. Those flags
start real paid requests and training. Local and saved-bank modes reject paid
enablement. The portable entry point uses the declared GPT-6.1 Sol Hibana
profile; Luna-generated saved banks can also be exported and reused locally.

Set `HIBANA_API_KEY` locally using `.env.hibana.example`. Keep the key out of
command arguments. Operational ceilings are bound to each teacher identity,
with a shared study cap, bounded requests and client pacing. A cap is not a
wallet balance. The identity-matched historical verifier audit is reused if
applicable; its 16 fixtures are a regression check, not a correctness guarantee.
Provider rates/options are recorded historical settings and should be reviewed
before future paid studies.

## Safeguards, outputs and tests

- Independent curriculum, comparison and final-test roles; no raw held-out text
  in teacher prompts.
- Complete answer supervision and finite loss/gradient checks.
- Original-reference provenance for local retrieval, verified admissions for
  live generation and fixed-bank provenance for reused synthetic data.
- Writer locks, atomic state commits, saved-adapter weight hashes and restart checks.
- Unique-example or token accounting, replay records and per-round summaries.
- Validated LoRA and training settings; the default recipe stays rank 4, alpha 8,
  dropout 0.05, learning rate 0.0002, three epochs and four-example accumulation.
- Difficulty changes, adaptive replay thresholds and rollback are not silently
  enabled in this measured-method comparison. They remain separate beta features
  requiring controlled ablations before promotion.

New run artifacts are under the selected output's `arms/<mode>/pilot/` directory.
`state.json` is authoritative; `round_events.json` provides a compact timeline.

```powershell
python run_controlled.py status --output outputs/local-validation
python -m pytest tests -q
python plot_results.py --snapshot ../../results/five-way-100/snapshot.json --output outputs/rebuilt-figures
```

The regression suite uses fixture models and no provider calls. One historical
raw-workspace integration test is skipped in the public checkout because its
private run inputs are intentionally not distributed. Published numeric results
have an independent integrity test. `run_five_way_training.py` remains the
historical workspace launcher; it requires those private historical inputs.
Use `run_controlled.py` for fresh portable preparation.
