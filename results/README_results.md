# Results Export Layout

This project saves notebook outputs in two places:

- `results/runs/run_XXX/`: permanent per-run exports
- `results/aggregated/`: latest report-ready outputs from the most recent successful notebook run

## Per-run folders

Each notebook execution creates the next sequential run folder:

- `run_001`
- `run_002`
- `run_003`

Each run folder contains:

- `figures/`: plots saved during that run
- `tables/`: exported tables saved during that run
- `logs/`: raw per-seed metrics and training-history exports
- `config.json`: notebook configuration and major hyperparameters
- `run_summary.json`: concise summary of the run outcome
- `manifest.json`: machine-readable inventory of saved files

## Aggregated outputs

`results/aggregated/` stores the latest polished outputs that are easiest to use for report writing:

- `figures/`
- `tables/`
- `logs/`
- `summary.json`

These aggregated files are overwritten by the most recent successful notebook execution.

## `latest/`

`results/runs/latest/` is intentionally not maintained automatically in this project.
The `aggregated/` directory serves as the latest easy-to-use output area instead.
