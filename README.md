# Masked-Context Prior-Guided Object–Scene Compatibility Reasoning

This repository contains the implementation for **Masked-Context Prior-Guided Object–Scene Compatibility Reasoning for Inpainted Images**.

The project studies **target- and replacement-aware object–scene compatibility reasoning** on COinCO-style inpainted images. Given an inpainted image, a target-region mask, and the replacement-object category, the model predicts whether the inserted object is **in-context** or **out-of-context** with respect to the surrounding scene.

## Overview

Modern image inpainting can produce visually realistic local edits, making low-level artifact cues unreliable. However, the inserted object may still be semantically inconsistent with the surrounding scene. This project focuses on such high-level object–scene compatibility reasoning.

Our method decomposes each input into three complementary views:

1. **Full image**: captures global scene composition.
2. **Object crop**: captures the inserted object appearance.
3. **Masked context**: removes the target object region and captures what the surrounding context expects.

The three visual views and the replacement-object text are encoded with frozen CLIP image and text encoders. A masked-context prior branch predicts an expected-object distribution for the removed region. The predicted prior is then compared with the actual replacement object through latent prior–text interactions and explicit scalar compatibility cues.

## Method

The main model is referred to as:

**Masked-Context Prior-Guided Tri-view Compatibility Learning**

For each sample, the model receives:

```text
I   : inpainted image
M   : target-region / bounding-box mask
t_r : replacement-object category
```

and predicts:

```text
0: in-context
1: out-of-context
```

The original-object category is used only during training to supervise the masked-context prior branch. It is never used at inference.

### Main Components

- **Tri-view decomposition**
  - full image
  - object crop
  - masked context

- **Frozen CLIP encoders**
  - CLIP image encoder for visual views
  - CLIP text encoder for replacement-object prompt

- **Masked-context prior prediction**
  - predicts an 80-way COCO object distribution from the masked context

- **Prior–replacement compatibility**
  - latent prior–text interaction
  - explicit scalar cues:
    - replacement prior probability
    - log replacement probability
    - maximum prior probability
    - margin between maximum prior and replacement prior
    - normalized prior entropy
    - replacement-to-maximum prior ratio

- **Binary compatibility classifier**
  - predicts in-context versus out-of-context

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── envs/
│   ├── README.md
│   ├── environment_coinco_full.yml
│   ├── environment_coinco_history.yml
│   └── requirements_coinco_lock.txt
│
├── tri_view_compat/
│   ├── datasets/
│   │   ├── coinco_triview_dataset.py
│   │   └── coinco_triview_dataset_v2.py
│   │
│   ├── models/
│   │   ├── clip_triview_baseline.py
│   │   ├── clip_triview_text_prior.py
│   │   ├── clip_triview_text_prior_aux.py
│   │   └── clip_triview_text_prior_v2.py
│   │
│   ├── tools/
│   │   ├── build_triview_splits.py
│   │   ├── build_balanced_test_split.py
│   │   ├── add_replacement_label.py
│   │   ├── eval_threshold_calibrated.py
│   │   ├── eval_threshold_calibrated_prior_aux.py
│   │   ├── audit_paper_tables.py
│   │   ├── check_paper_table_values.py
│   │   ├── bootstrap_ci.py
│   │   ├── inspect_model_details.py
│   │   ├── select_qualitative_cases.py
│   │   └── render_qualitative_figure.py
│   │
│   ├── train_triview.py
│   ├── train_triview_prior.py
│   ├── train_triview_prior_aux.py
│   └── train_triview_prior_v2.py
│
├── coinco_v1_baselines/
│   └── train_cached_context_baseline.py
│
├── figures/
│   ├── fig_2.pdf
│   └── fig_2_preview.png
│
├── paper_audit/
│   └── final_results/
│
├── fake_localization/
└── objects_from_context_prediction/
```

## Data Preparation

This repository does **not** include COinCO images, masks, official cached features, or model checkpoints.

Prepare the official COinCO resources following the COinCO dataset instructions, then create symbolic links or place the data under the project root as follows:

```text
task_data/
checkpoints/
```

Expected data layout:

```text
task_data/
├── context_prediction/
├── inpainting_info/
├── objects_from_context_prediction/
├── images/
│   ├── training_val_images/
│   └── testing_images/
├── masks/
│   ├── bbox_masks_training_val/
│   └── bbox_masks_testing/
└── cache/
```

The code also requires a local CLIP ViT-B/32 checkpoint or a Hugging Face CLIP model path. In the commands below, replace:

```text
/path/to/clip-vit-base-patch32
```

with your actual CLIP path.

## Environment

A full environment file is provided under `envs/`.

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

The experiments were developed with PyTorch 2.5.1 and CUDA 12.1.

## Building Splits

Build the replacement-aware tri-view splits:

```bash
python -m tri_view_compat.tools.build_triview_splits \
  --task_data_root task_data \
  --output_dir tri_view_compat/outputs/splits
```

Add replacement labels if required:

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

The original test set contains 556 in-context and 1,846 out-of-context samples. The balanced subset contains all 556 in-context samples and 556 randomly sampled out-of-context samples using seed 777.

## Training

Set your CLIP path:

```bash
export CLIP_PATH=/path/to/clip-vit-base-patch32
```

### Tri-view + Text Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --clip_name "$CLIP_PATH" \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_no_geo_dropout01_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 4 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --no_geo \
  --seed 777
```

### Decoupled Prior

This variant trains the masked-context prior prediction branch but does not feed the predicted prior into the final compatibility classifier.

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

### Full Model

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

The evaluation protocol first selects the best checkpoint by validation Macro-F1 using the default argmax rule. Then it calibrates the decision threshold on the validation set over:

```text
{0.01, 0.02, ..., 0.99}
```

The threshold is selected by maximizing validation Macro-F1, breaking ties by Balanced Accuracy and then Accuracy. The selected threshold is fixed for both the original test set and the balanced test subset.

### Original Test Set

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

### Balanced Test Subset

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --seed 777
```

Available `model_type` values include:

```text
baseline
prior
prior_aux
prior_v2
```

## COinCO-style Baselines

The reimplemented COinCO-style baselines use the released cached features:

- `COinCO-VisualNet*`: VAE latent features
- `COinCO-SemanticNet*`: surrounding-object and replacement-object embeddings
- `COinCO-VisSemanticNet*`: concatenation of visual and semantic features

Run:

```bash
python coinco_v1_baselines/train_cached_context_baseline.py \
  --mode semantic \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --balanced_test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --output_dir coinco_v1_baselines/outputs/semantic
```

Replace `--mode semantic` with:

```text
visual
vissemantic
```

for the other two baselines.

The asterisk `*` in the paper denotes our reimplementation using released COinCO cached features under the unified replacement-aware protocol. Published COinCO numbers are not directly comparable because of different task definitions, sample construction, and evaluation protocols.

## Main Results

### Original Test Set

| Method | Th. | Acc | BAcc | M-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| COinCO-VisualNet* | 0.45 | 68.69 | 55.94 | 55.95 | 32.25 | 79.64 | 62.49 |
| COinCO-SemanticNet* | 0.37 | 76.94 | 68.22 | 67.99 | 51.06 | 84.91 | 78.60 |
| COinCO-VisSemanticNet* | 0.42 | 74.65 | 68.67 | 67.06 | 51.24 | 82.87 | 76.59 |
| Tri-view + Text | 0.38 | 80.93 | 75.59 | 74.39 | 61.45 | 87.33 | 84.94 |
| Ours | 0.37 | 81.93 | 76.93 | 75.71 | 63.41 | 88.00 | 85.45 |

### Balanced Test Subset

Balanced-set Balanced Accuracy equals Accuracy and is omitted.

| Method | Acc | M-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|
| COinCO-VisualNet* | 56.65 | 53.90 | 42.62 | 65.17 | 63.69 |
| COinCO-SemanticNet* | 68.35 | 67.47 | 62.15 | 72.80 | 77.86 |
| COinCO-VisSemanticNet* | 68.08 | 67.72 | 64.32 | 71.11 | 75.19 |
| Tri-view + Text | 74.37 | 74.17 | 71.92 | 76.43 | 83.48 |
| Ours | 75.99 | 75.82 | 73.80 | 77.84 | 84.05 |

## Ablation Results

Ablation on the balanced test subset:

| Variant | Th. | Acc | M-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tri-view + Text | 0.38 | 74.37 | 74.17 | 71.92 | 76.43 | 83.48 |
| Decoupled Prior | 0.33 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| Latent, λ = 0.1 | 0.38 | 73.47 | 73.21 | 70.59 | 75.84 | 83.50 |
| Latent, λ = 0.2 | 0.34 | 75.27 | 74.97 | 72.25 | 77.70 | 84.15 |
| Ours, λ = 0.2 | 0.37 | 75.99 | 75.82 | 73.80 | 77.84 | 84.05 |

## Qualitative Figure

The final qualitative figure is provided under:

```text
figures/fig_2.pdf
figures/fig_2_preview.png
```

Regenerate the qualitative figure:

```bash
python -m tri_view_compat.tools.render_qualitative_figure \
  --csv tri_view_compat/outputs/qualitative_selection/selected_cases.csv \
  --out_pdf figures/fig_2.pdf \
  --out_png figures/fig_2_preview.png \
  --case_rows 377 308 \
  --fig_h 3.05
```

The figure demonstrates two cases where the Tri-view + Text baseline fails but the full prior-guided model succeeds:

- an out-of-context replacement with negligible prior support,
- an in-context replacement that matches the top context expectation.

## Inspecting Model Size

Run:

```bash
python -m tri_view_compat.tools.inspect_model_details
```

Expected full-model size:

```text
Total parameters:     153.37M
Trainable parameters:   2.09M
Trainable ratio:        1.37%
```

## Reproducibility Notes

- All internal tri-view variants use frozen CLIP ViT-B/32.
- All internal tri-view variants use seed 777.
- The paper-facing Tri-view + Text baseline uses dropout 0.1.
- The full model uses `lambda_prior = 0.2`.
- Test labels are never used for checkpoint selection or threshold calibration.
- AUC is computed from predicted probabilities and is threshold-independent.
- The original-object category is used only for training the prior branch and is not used at inference.

## License

This repository is released under the license specified in `LICENSE`.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{luo2026maskedcontext,
  title     = {Masked-Context Prior-Guided Object--Scene Compatibility Reasoning for Inpainted Images},
  author    = {Luo, Zetong and Wan, Chen},
  booktitle = {ICASSP},
  year      = {2026}
}
```

Please also cite the original COinCO and CLIP papers when using this code or data protocol.
