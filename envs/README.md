# Environment files

This folder contains the environment specifications used for the MCP-Compat experiments.

## Files

- `environment_coinco_full.yml`: full Conda environment export from the experiment machine, with the local prefix removed.
- `requirements_coinco_lock.txt`: pip package-version reference.

## Recommended setup

```bash
conda env create -f envs/environment_coinco_full.yml
conda activate coinco
```

Alternatively, create a clean Python 3.10 environment and install the locked Python packages:

```bash
conda create -n coinco python=3.10 -y
conda activate coinco
pip install -r envs/requirements_coinco_lock.txt
```

The experiments were developed with Python 3.10, PyTorch 2.5.1, and CUDA 12.1.

Dataset files, checkpoints, pretrained CLIP weights, and released COinCO resources are not included in this repository.
