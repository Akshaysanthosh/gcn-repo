# Reproducibility Study of Graph Convolutional Networks for Semi-Supervised Learning

This repository contains a CS 679 course reproducibility project based on:

Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017 / arXiv:1609.02907.

The project focuses on a clean reimplementation of the original 2-layer GCN for semi-supervised node classification on the Cora citation network, along with a feature-only MLP baseline and one extension experiment on label sparsity.

## Quick start

If you want the shortest path from a fresh clone to a full notebook run, use:

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `notebooks/gcn_reproducibility.ipynb` and run all cells from a fresh kernel.

## Project goals

- Reproduce the main claim that a graph-aware GCN outperforms a feature-only MLP on Cora.
- Keep the implementation close to the original paper and code release.
- Make the workflow readable, well-structured, and runnable end to end from a single notebook.

## What is included

- Original-split Cora loader using the Planetoid-format files from the original GCN repository
- 2-layer GCN with explicit graph convolution layers
- Feature-only 2-layer MLP baseline
- Early stopping on validation loss
- Multi-seed evaluation with accuracy and macro-F1
- t-SNE visualization of learned embeddings
- Label sparsity study comparing GCN and MLP under fewer labeled nodes per class

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── gcn_reproducibility.ipynb
├── src/
│   ├── data.py
│   ├── eval.py
│   ├── export.py
│   ├── models.py
│   ├── plots.py
│   ├── train.py
│   └── utils.py
├── results/
│   ├── runs/
│   ├── aggregated/
│   └── README_results.md
└── data/
```

## Setup

### Prerequisites

- Python 3.10 or newer
- `pip`
- Internet access on the first run so the original Cora files can be downloaded

### Step-by-step setup

1. Clone the repository and enter the project directory:

```bash
git clone <repo-url>
cd <repo-folder>
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open `notebooks/gcn_reproducibility.ipynb`.

6. In the notebook UI, restart the kernel if needed and run the notebook from top to bottom.

The notebook downloads the original Cora files into `data/` if they are not already present.
Each fresh execution also creates a new run folder under `results/runs/run_XXX/`.

### What a successful run looks like

After the notebook finishes, you should have:

- a new folder such as `results/runs/run_001/`
- saved plots in `results/runs/run_001/figures/`
- saved tables in `results/runs/run_001/tables/`
- saved logs in `results/runs/run_001/logs/`
- metadata files: `config.json`, `run_summary.json`, and `manifest.json`
- a mirrored set of latest report-ready outputs in `results/aggregated/`

The final notebook cell also prints the main saved paths for that run.

## Experiments

### Main reproduction

- Dataset: Cora
- Models: MLP, GCN
- Metrics: test accuracy, macro-F1
- Reporting: mean ± standard deviation over multiple random seeds

### Extension: label sparsity

The extension study varies the number of labeled training nodes per class:

- 1
- 3
- 5
- 10
- 20

The original test set is kept fixed, and new class-balanced training splits are sampled from the remaining nodes.

## Expected outputs

Running the notebook should produce:

- training curves for representative MLP and GCN runs
- a table comparing MLP and GCN across multiple seeds
- t-SNE visualizations of learned embeddings
- a label sparsity table and plot

Per-run outputs are saved under `results/runs/run_XXX/`:

- `figures/` for plots
- `tables/` for report tables
- `logs/` for per-seed metrics and training histories
- `config.json`, `run_summary.json`, and `manifest.json` for run metadata

The latest report-ready copies are mirrored into `results/aggregated/`. See `results/README_results.md` for the export layout.

## Notes for first-time users

- The first notebook run takes longer because it downloads Cora and executes all experiments, including multi-seed evaluation and the label sparsity study.
- If `jupyter notebook` is not found, activate the virtual environment again and rerun `pip install -r requirements.txt`.
- If the notebook appears to have stale output, restart the kernel and run all cells again from the top.

## Reproducibility notes

- Random seeds are explicitly controlled for Python, NumPy, and PyTorch.
- The implementation uses the standard semi-supervised Cora split from the original GCN release.
- Dense normalized adjacency is used for simplicity and readability because Cora is small enough for this to remain practical in a course project setting.
