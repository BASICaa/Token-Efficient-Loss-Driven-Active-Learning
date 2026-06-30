# TEFLD Comparison Data

This folder contains a compact comparison snapshot from run `20260629_215451` and section `section_026`.

![TEFLD comparison dashboard](dashboard.png)

## Key Takeaways

- Best TEFLD validation checkpoint: round `9` with shared validation loss `2.7027`.
- TEFLD best holdout avg loss: `3.1823`.
- Dataset baseline holdout avg loss: `2.2175`.
- The gap suggests TEFLD learned its generated curriculum, but the broader Dolly-style holdout distribution was still harder.

## Files

- `TEFLD_comparison_summary.xlsx`: formatted workbook with dashboard, charts, and source tables.
- `eval_summary.csv`: shared validation and final holdout loss summary.
- `tefld_round_summary.csv`: TEFLD training and validation trend by round.
- `ledger_round_summary.csv`: section ledger summary by round.
- `tag_summary.csv`: losses and counts by learning tag.
- `vault_items.csv`: current failure vault contents.
- `validation_weakness_categories.csv`: latest validation weakness categories.
- `shared_eval_losses.csv` and `holdout_eval_losses.csv`: per-example eval losses.

GitHub renders this Markdown file, CSV tables, and `dashboard.png` directly. Download the `.xlsx` file for the full workbook experience.
