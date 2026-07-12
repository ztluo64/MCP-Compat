# Environment files

This folder contains the environment specifications used for the COinCO MCP-TCL experiments.

## Files

- `environment_coinco_full.yml`: full Conda environment export from the experiment machine, with local prefix removed.
- `environment_coinco_history.yml`: Conda environment export from explicitly installed packages only.
- `requirements_coinco_lock.txt`: pip freeze output for package-version reference.

## Recommended setup

We recommend creating the environment from the full Conda export:

```bash
conda env create -f envs/environment_coinco_full.yml
conda activate coinco

If dependency solving fails on another machine, create a clean Python 3.10 environment and install packages according to environment_coinco_history.yml and requirements_coinco_lock.txt.

Notes
The experiments used Python 3.10 and PyTorch with CUDA support.
Dataset files, checkpoints, pretrained CLIP weights, and COinCO released resources are not included in this repository.
Local machine paths are intentionally removed from the exported environment files.
