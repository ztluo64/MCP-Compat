# MCP-Compat: Masked-Context Prior-Guided Object–Scene Compatibility Reasoning for Inpainted Images

This repository contains the implementation of **Masked-Context Prior-Guided Object--Scene Compatibility Reasoning for Inpainted Images**.

The project studies **target- and replacement-aware object--scene compatibility reasoning**. Given an inpainted image, a target-region mask, and the replacement-object category, the model predicts whether the inserted object is **in-context (IC)** or **out-of-context (OOC)** with respect to the surrounding scene.

## Overview

MCP-Compat decomposes each sample into three visual views:

1. **Full image** for global scene composition.
2. **Target crop** for inserted-object appearance.
3. **Masked context** for surrounding contextual evidence.

The three visual views and replacement-object text are encoded with frozen CLIP encoders. A masked-context prior branch predicts an expected-object distribution for the target location. The predicted prior is compared with the actual replacement through latent prior--text interactions and explicit probability-based compatibility cues.

The original-object category is used only to supervise the context-prior branch during training and is not required at inference.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── envs/
│   ├── README.md
│   ├── environment_coinco_full.yml
│   └── requirements_coinco_lock.txt
├── baselines/
│   └── coinco_specialists/
│       ├── run_specialist_scores.py
│       ├── evaluate_specialist_scores.py
│       └── evaluate_fusion_val_threshold.py
└── tri_view_compat/
    ├── datasets/
    │   ├── coinco_triview_dataset.py
    │   └── coinco_triview_dataset_v2.py
    ├── models/
    │   ├── clip_triview_baseline.py
    │   ├── clip_triview_text_prior.py
    │   ├── clip_triview_text_prior_aux.py
    │   ├── clip_triview_text_prior_explicit_only.py
    │   └── clip_triview_text_prior_v2.py
    ├── tools/
    │   ├── add_replacement_label.py
    │   ├── build_triview_splits.py
    │   ├── build_balanced_test_split.py
    │   ├── eval_threshold_calibrated.py
    │   ├── eval_threshold_calibrated_prior_aux.py
    │   ├── eval_threshold_calibrated_explicit_only.py
    │   ├── inspect_model_details.py
    │   └── train_context_prior_probe.py
    ├── train_triview.py
    ├── train_triview_prior.py
    ├── train_triview_prior_aux.py
    ├── train_triview_prior_explicit_only.py
    └── train_triview_prior_v2.py
```

## Data Preparation

This repository does **not** include COinCO images, masks, released specialist checkpoints, pretrained backbones, or our trained checkpoints.

Official COinCO resources:

- **COinCO dataset:** https://huggingface.co/datasets/COinCO/COinCO-dataset
- **COinCO resources / checkpoints:** https://huggingface.co/datasets/COinCO/COinCO-resources
- **Official COinCO repository:** https://github.com/YangTianze009/COinCO

Optional downloads with the Hugging Face CLI:

```bash
hf download COinCO/COinCO-dataset \
  --repo-type dataset \
  --local-dir COinCO-dataset

hf download COinCO/COinCO-resources \
  --repo-type dataset \
  --local-dir COinCO-resources
```

After preparing the official resources, place or symlink the processed data under the project root as:

```text
task_data/
checkpoints/
```

The scripts expect the relevant COinCO context-prediction, inpainting-information, objects-from-context, image, mask, and cache resources under `task_data/`.

## Backbones

### MCP-Base / MCP-Compat

Our models use **OpenAI CLIP ViT-B/32** with frozen image and text encoders.

- Model page: https://huggingface.co/openai/clip-vit-base-patch32

```bash
export CLIP_PATH=openai/clip-vit-base-patch32
```

or use a local snapshot:

```bash
export CLIP_PATH=/path/to/clip-vit-base-patch32
```

### Released COinCO specialists

The released specialist baselines use Qwen2.5-VL-3B-Instruct and the COinCO specialist checkpoints:

- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- https://huggingface.co/COinCO/Qwen2.5-VL-3B-Co_occurrence
- https://huggingface.co/COinCO/Qwen2.5-VL-3B-Location
- https://huggingface.co/COinCO/Qwen2.5-VL-3B-Size

## Environment

```bash
conda env create -f envs/environment_coinco_full.yml
conda activate coinco
```

Alternatively:

```bash
conda create -n coinco python=3.10 -y
conda activate coinco
pip install -r envs/requirements_coinco_lock.txt
```

The experiments were developed with Python 3.10, PyTorch 2.5.1, and CUDA 12.1.

## Building the Replacement-Aware Splits

```bash
python -m tri_view_compat.tools.build_triview_splits \
  --task_data_root task_data \
  --output_dir tri_view_compat/outputs/splits
```

If required, add replacement labels:

```bash
python -m tri_view_compat.tools.add_replacement_label \
  --input_dir tri_view_compat/outputs/splits \
  --output_dir tri_view_compat/outputs/splits
```

Build the balanced test subset:

```bash
python -m tri_view_compat.tools.build_balanced_test_split \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --output_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --seed 777
```

Expected split sizes:

```text
Train:       76,256
Validation:  19,064
Test:         2,402
Balanced:     1,112
```

The original test set contains 556 IC and 1,846 OOC samples. The balanced subset contains all 556 IC samples and 556 randomly sampled OOC samples with seed 777.

## Training

### MCP-Base

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_no_geo_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.2 \
  --no_geo \
  --seed 777
```

### Decoupled Prior

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior_aux \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --prior_loss_weight 0.2 \
  --no_geo \
  --seed 777
```

### Explicit Compatibility Only

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior_explicit_only \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_explicit_only_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --prior_loss_weight 0.2 \
  --no_geo \
  --seed 777
```

### Latent Prior Compatibility

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --prior_loss_weight 0.2 \
  --no_geo \
  --seed 777
```

### MCP-Compat

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior_v2 \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --prior_loss_weight 0.2 \
  --no_geo \
  --seed 777
```

## Evaluation

The evaluation protocol first selects the best checkpoint by validation Macro-F1 using the default argmax rule. The decision threshold is then calibrated on the validation set over `{0.01, 0.02, ..., 0.99}` by maximizing Macro-F1, with ties broken by Balanced Accuracy and then Accuracy. The selected threshold is fixed for both test sets. Test labels are not used for checkpoint selection or threshold calibration.

MCP-Compat on the original test set:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --batch_size 256 \
  --num_workers 8 \
  --seed 777
```

For the balanced test subset, replace `--test_csv` with:

```text
tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv
```

For Decoupled Prior and Explicit-only evaluation, use:

```text
tri_view_compat.tools.eval_threshold_calibrated_prior_aux
tri_view_compat.tools.eval_threshold_calibrated_explicit_only
```

## Context-Prior Diagnostic

The diagnostic experiment reported in the paper can be reproduced with:

```bash
python -m tri_view_compat.tools.train_context_prior_probe \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH"
```

## Released COinCO Specialist Baselines

We reevaluate the three released COinCO specialists under our replacement-aware protocol:

- Co-occurrence
- Location
- Size

`run_specialist_scores.py` extracts a continuous OOC score from the first discriminative decision token. `evaluate_specialist_scores.py` selects each specialist threshold on the full validation set using the same Macro-F1 / BAcc / Acc calibration rule used for our models.

### Score Fusion

Score Fusion is our calibrated fusion of the three released specialists, not an author-released fourth specialist. For specialist `k` with OOC score `s_k` and validation-selected threshold `tau_k`, we align scores in logit space:

```text
z_k = logit(s_k) - logit(tau_k)
```

and use the strongest aligned OOC evidence:

```text
s_fuse = sigmoid(max_k z_k)
```

The final fusion threshold is selected independently on the full validation set using the same calibration protocol.

## Main Results

### Original Test Set

| Method | Th. | Acc | BAcc | M-F1 | F1-IC | F1-OOC | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Co-occurrence† | 0.93 | 77.06 | 71.44 | 69.88 | 55.17 | 84.59 | 79.27 |
| Location† | 0.90 | 75.77 | 69.22 | 67.96 | 52.14 | 83.78 | 78.98 |
| Size† | 0.68 | 75.44 | 66.42 | 66.11 | 48.34 | 83.89 | 74.88 |
| Score Fusion† | 0.71 | 75.56 | 69.40 | 67.94 | 52.32 | 83.57 | 76.59 |
| MCP-Base | 0.38 | 81.43 | 75.73 | 74.80 | 61.88 | 87.73 | 85.18 |
| **MCP-Compat** | **0.37** | **81.93** | **76.93** | **75.71** | **63.41** | **88.00** | **85.45** |

### Balanced Test Subset

| Method | Acc | M-F1 | F1-IC | F1-OOC | AUC |
|---|---:|---:|---:|---:|---:|
| Co-occurrence† | 71.85 | 71.52 | 68.42 | 74.61 | 79.03 |
| Location† | 70.23 | 69.70 | 65.70 | 73.71 | 79.29 |
| Size† | 67.00 | 65.97 | 60.07 | 71.88 | 75.64 |
| Score Fusion† | 69.69 | 69.27 | 65.65 | 72.89 | 77.11 |
| MCP-Base | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| **MCP-Compat** | **75.99** | **75.82** | **73.80** | **77.84** | 84.05 |

† Score-based reevaluation of the released COinCO specialist checkpoints.

## Ablation Results

| Variant | Th. | Acc | M-F1 | F1-IC | F1-OOC | AUC |
|---|---:|---:|---:|---:|---:|---:|
| MCP-Base | 0.38 | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| Decoupled Prior | 0.33 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| Explicit | 0.33 | 73.56 | 73.41 | 71.40 | 75.42 | 81.62 |
| Latent | 0.34 | 75.27 | 74.97 | 72.25 | 77.70 | **84.15** |
| **MCP-Compat** | **0.37** | **75.99** | **75.82** | **73.80** | **77.84** | 84.05 |

All prior-based variants use `lambda_prior = 0.2`. The reported MCP-Base checkpoint uses dropout 0.2; prior-based variants use dropout 0.1.

## Reproducibility Notes

- Frozen CLIP ViT-B/32 is used for all internal variants.
- Seed: 777.
- Best checkpoints are selected by validation Macro-F1 before threshold calibration.
- Test labels are never used for checkpoint selection or threshold calibration.
- AUC is computed from continuous OOC scores.
- The original-object category is used only as training supervision for the prior branch.

## Acknowledgements

We sincerely thank the authors of [COinCO](https://github.com/YangTianze009/COinCO) for releasing the COinCO dataset, pretrained resources, and open-source code. Their public resources made this study possible and substantially facilitated reproducible research on object--scene compatibility reasoning in inpainted images.

## License

See `LICENSE`.

## Citation

If you find this work useful, please cite the manuscript:

**Masked-Context Prior-Guided Object--Scene Compatibility Reasoning for Inpainted Images**  
Zetong Luo$^{1}$, Chen Wan$^{1}$*, Wentao Zhang$^{1}$, Zichun Wu$^{1}$, Lifeng Huang$^{2}$

$^{1}$ Department of Computer Science and Technology, Shantou University, Shantou, China  
$^{2}$ College of Mathematics and Informatics, South China Agricultural University, Guangzhou, China  
*Corresponding author: Chen Wan.*

```bibtex
@misc{luo2026maskedcontext,
  title  = {Masked-Context Prior-Guided Object--Scene Compatibility Reasoning for Inpainted Images},
  author = {Luo, Zetong and Wan, Chen and Zhang, Wentao and Wu, Zichun and Huang, Lifeng},
  year   = {2026},
  note   = {Manuscript}
}
```

Please also cite the original COinCO and CLIP papers when using this code or data protocol.