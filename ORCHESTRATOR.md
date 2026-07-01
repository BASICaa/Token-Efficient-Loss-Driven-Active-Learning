# Orchestrator

The orchestrator is the TEFLD end-to-end runner. It is resumable by design: every major stage is saved to section state so an interrupted run can continue.

Implementation:

```text
TEFLD/src/orchestrator.py
```

## Main Loop

```text
bootstrap section files
load example.txt into pipeline state
ensure shared_validation.json exists for evaluated runs

if no current training batch exists:
    if an active recipe exists:
        reuse it
    elif this is fresh round 0:
        create a full generation recipe
    else:
        ask PolicyMaker for the next recipe

    ask Instructor to build the current training batch with adaptive difficulty budgets

if a current training batch exists:
    train the Student LoRA adapter
    evaluate fixed shared_validation.json for checkpoint decisions
    run Evaluator on the batch
    append results to the ledger
    rebuild the failure vault
    advance the round
    clear active recipe and current batch
```

## Resume Rules

If a run stops after recipe creation, `active_recipe` remains in `pipeline_state.json`.

If a run stops after batch creation, `current_training_batch` remains in `pipeline_state.json`.

If a run stops after training, the adapter may already exist under `TEFLD/Models/<section_id>/round_XXX/`. If `adapter_config.json` and `training_complete.json` exist, the next run skips retraining and evaluates.

If a run stops after evaluation, the ledger and vault have already been updated and the current round has advanced.

Best-round and rollback decisions use the fixed shared-validation loss when `shared_validation.json` is available. This avoids choosing checkpoints based only on the just-trained 10-sample batch, which can become memorized after repeated rounds.

## Result Statuses

Common statuses returned by `run_tefld.py`:

- `needs_example`: `example.txt` exists but needs user examples
- `ready`: section files and examples are ready
- `batch_ready`: a 10-sample training batch has been created
- `trained`: training finished
- `evaluated`: evaluation finished and the round advanced

## Recommended Entry Point

From the repository root:

```powershell
python run_tefld.py --rounds 1
```

For direct Python usage:

```python
from TEFLD.src.orchestrator import Orchestrator

result = Orchestrator().run(max_rounds=1)
print(result)
```
