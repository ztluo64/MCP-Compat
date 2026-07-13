# MCP-TCL: Masked Context Prior-guided Tri-view Compatibility Learning

This repository contains the code and result-audit files for our COinCO replacement-aware object-scene compatibility classification experiments.

We study the following question:

> Given an inpainted image, a target region, and the inserted replacement object category, is the inserted object compatible with the surrounding scene context?

Our method, **Masked Context Prior-guided Tri-view Compatibility Learning (MCP-TCL)**, decomposes each sample into three complementary views and explicitly compares the object expected by the masked context with the actual inserted replacement object.

---

## Overview

Modern diffusion-based image inpainting can generate visually realistic local content, making low-level artifact detection unreliable. COinCO shifts the focus from pixel-level artifacts to high-level contextual plausibility: an inserted object may look realistic, but still be semantically, spatially, or contextually implausible.

MCP-TCL addresses this setting with three components:

1. **Tri-view object-context decomposition**
   - Full image: global scene and layout
   - Object crop: inserted object appearance
   - Masked context: surrounding scene without the target object

2. **Replacement object semantic conditioning**
   - The replacement object category is encoded with a frozen CLIP text encoder.

3. **Masked-context prior compatibility**
   - The masked context predicts an expected-object distribution.
   - The predicted prior is compared with the actual replacement object through:
     - latent prior-text compatibility
     - explicit scalar compatibility cues such as \(p(\text{replacement}\mid\text{context})\), margin, entropy, and ratio.

---

## Method naming

The paper-facing method names differ slightly from some internal engineering names.

| Code / checkpoint name | Paper-facing name |
|---|---|
| `triview_text_no_geo_dropout01_bs256` | Tri-view + Text |
| `triview_text_prior_aux_lam02_bs256` | Decoupled Prior Prediction |
| `triview_text_prior_lam01_bs256` | Latent Prior Compatibility, \(\lambda_{\text{prior}}=0.10\) |
| `triview_text_prior_lam02_bs256` | Latent Prior Compatibility, \(\lambda_{\text{prior}}=0.20\) |
| `triview_text_prior_v2_lam02_bs256` | Ours / Latent + Explicit Scalar Prior Compatibility |
| `clip_triview_text_prior_aux.py` | Decoupled Prior Prediction implementation |
| `clip_triview_text_prior.py` | Latent Prior Compatibility implementation |
| `clip_triview_text_prior_v2.py` | Ours implementation |

Important terminology:

- `Decoupled Prior Prediction` trains an additional masked-context prior prediction head, but does **not** feed the predicted prior or any prior-replacement compatibility feature into the final classifier.
- `Ours` uses both latent prior-text compatibility and explicit scalar prior-replacement compatibility cues.

---

## Repository structure

```text
.
├── tri_view_compat/                 # Main MCP-TCL implementation
│   ├── datasets/                    # Tri-view COinCO datasets
│   ├── models/                      # CLIP tri-view and prior compatibility models
│   ├── tools/                       # Split construction, evaluation, visualization, audit
│   ├── train_triview.py             # Full / crop / masked / tri-view / tri-view+text baselines
│   ├── train_triview_prior.py       # Latent Prior Compatibility
│   ├── train_triview_prior_aux.py   # Decoupled Prior Prediction
│   └── train_triview_prior_v2.py    # Ours
├── coinco_v1_baselines/             # COinCO-style cached-feature baselines
├── MCP_TCL_core/                    # Compact code/results snapshot for paper review
├── paper_audit/final_results/       # Final table sources and manifest
├── envs/requirements_coinco_lock.txt
├── envs/environment_coinco_full.yml
└── README.md
```

The following large resources are intentionally **not** included in this repository:

```text
task_data/
checkpoints/
tri_view_compat/outputs/
coinco_v1_baselines/outputs/
qwen_lora/
*.pt / *.pth / *.bin / *.safetensors
```

---

## Data preparation

This code expects the released COinCO resources to be available locally.

A typical setup is:

```text
/path/to/COinCO/
├── task_data/
│   ├── images/
│   ├── masks/
│   ├── cache/
│   ├── context_prediction/
│   ├── inpainting_info/
│   └── objects_from_context_prediction/
└── checkpoints/
```

Inside the repository root, create symbolic links:

```bash
ln -s /path/to/COinCO/task_data task_data
ln -s /path/to/COinCO/checkpoints checkpoints
```

Expected important files include:

```text
task_data/context_prediction/balanced/training_val_data.csv
task_data/context_prediction/testing_data.csv
task_data/inpainting_info/training_inpainting_info.csv
task_data/inpainting_info/testing_inpainting_info.csv
task_data/objects_from_context_prediction/training_data.csv
task_data/objects_from_context_prediction/validation_data.csv
task_data/objects_from_context_prediction/testing_data.csv
task_data/cache/cached_embeddings_train_val_cp/
task_data/cache/cached_embeddings_testing_cp/
```

---

## Environment

We used Python 3.10 and PyTorch with CUDA support.

Create an environment from the provided file:

```bash
conda env create -f envs/environment_coinco_full.yml
conda activate coinco
```

Or install the minimal dependencies:

```bash
See envs/requirements_coinco_lock.txt for pip package versions.
```

You also need a local CLIP ViT-B/32 checkpoint or a HuggingFace-compatible CLIP directory. In our scripts this is passed through `--clip_name`.

Example:

```bash
--clip_name /path/to/clip-vit-base-patch32
```

---

## Split construction

Build the replacement-aware split:

```bash
python -m tri_view_compat.tools.build_triview_splits
```

Add replacement labels:

```bash
python -m tri_view_compat.tools.add_replacement_label
```

Build the balanced test subset:

```bash
python -m tri_view_compat.tools.build_balanced_test_split
```

Current split statistics:

| Split | Samples | Label 0 | Label 1 |
|---|---:|---:|---:|
| Train | 76,256 | 16,088 | 60,168 |
| Validation | 19,064 | 3,963 | 15,101 |
| Original test | 2,402 | 556 | 1,846 |
| Balanced test | 1,112 | 556 | 556 |

Label convention:

```text
0 = in-context
1 = out-of-context
```

Important protocol note:

- `prior_label` is used only during training as the supervision target for masked-context prior prediction.
- `prior_label` is not used as test-time input.
- `replacement_label` is deterministically mapped from `replacement_object` using the train split category mapping.
- The original test set is the full replacement-aware test set.
- The balanced test subset is used as a class-balance robustness check.

---

## Training

### Full / crop / masked / tri-view / tri-view+text baselines

```bash
python -m tri_view_compat.train_triview \
  --clip_name /path/to/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_no_geo_dropout01_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --dropout 0.1 \
  --no_geo
```

Example switches:

```text
--no_full      remove full image view
--no_crop      remove object crop view
--no_masked    remove masked context view
--no_text      remove replacement object text
--no_geo       remove geometry feature
```

### Decoupled Prior Prediction

```bash
python -m tri_view_compat.train_triview_prior_aux \
  --clip_name /path/to/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

### Latent Prior Compatibility

```bash
python -m tri_view_compat.train_triview_prior \
  --clip_name /path/to/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

### Ours

```bash
python -m tri_view_compat.train_triview_prior_v2 \
  --clip_name /path/to/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

---

## Evaluation protocol

For all binary classification results, the decision threshold is selected once on the validation set by maximizing Macro-F1:

```text
threshold ∈ {0.01, 0.02, ..., 0.99}
```

The selected threshold is then fixed for both:

```text
original replacement-aware test set
balanced test subset
```

Test labels are not used for threshold selection. AUC is computed from predicted probabilities and is threshold-independent.

Example evaluation on original test:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --batch_size 256 \
  --num_workers 8 \
  --metric macro_f1
```

Example evaluation on balanced test:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --metric macro_f1
```

---

## COinCO-style baselines

COinCO-style baselines are implemented in:

```text
coinco_v1_baselines/
```

They use released cached features:

```text
task_data/cache/cached_embeddings_train_val_cp/
task_data/cache/cached_embeddings_testing_cp/
```

Paper-facing baselines:

| Method | Features |
|---|---|
| COinCO-VisualNet* | `latent1 + latent2` |
| COinCO-SemanticNet* | `objects_embeddings + replacement_embedding` |
| COinCO-VisSemanticNet* | `latent1 + latent2 + objects_embeddings + replacement_embedding` |

Run example:

```bash
python coinco_v1_baselines/train_cached_context_baseline.py \
  --mode vissemantic \
  --output_dir coinco_v1_baselines/outputs/coinco_vissemanticnet_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 4 \
  --lr 1e-3
```

`*` denotes our reimplementation using released COinCO cached visual and semantic embeddings under the unified replacement-aware protocol. Published COinCO numbers are not directly comparable because of different data construction and evaluation settings.

---

## Final results

### Table 1: COinCO-style baselines and structured variants

This table reports both the original replacement-aware test set and the balanced test subset. The original test set is the main evaluation protocol, while the balanced subset is used as a class-balance robustness check.

| Type | Method | Th. | Original Acc | Original Bal Acc | Original Macro-F1 | Original AUC | Balanced Acc | Balanced Macro-F1 | Balanced AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COinCO-style | COinCO-VisualNet* | 0.45 | 68.69 | 55.94 | 55.95 | 62.49 | 56.65 | 53.90 | 63.69 |
| COinCO-style | COinCO-SemanticNet* | 0.37 | 76.94 | 68.22 | 67.99 | 78.60 | 68.35 | 67.47 | 77.86 |
| COinCO-style | COinCO-VisSemanticNet* | 0.42 | 74.65 | 68.67 | 67.06 | 76.59 | 68.08 | 67.72 | 75.19 |
| Ours variant | Tri-view + Text | 0.38 | 80.93 | 75.59 | 74.39 | 84.94 | 74.37 | 74.17 | 83.48 |
| Ours | Ours | 0.37 | 81.93 | 76.93 | 75.71 | 85.45 | 75.99 | 75.82 | 84.05 |

### Table 2: Input decomposition ablation on original test

| Method | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full only | 0.42 | 75.85 | 63.87 | 64.46 | 44.34 | 84.58 | 72.72 |
| Object crop only | 0.36 | 75.77 | 68.21 | 67.38 | 50.84 | 83.92 | 78.17 |
| Masked context only | 0.44 | 70.61 | 59.13 | 59.05 | 37.30 | 80.80 | 66.63 |
| Tri-view | 0.33 | 78.06 | 69.51 | 69.38 | 53.07 | 85.68 | 80.22 |
| Tri-view + Text | 0.38 | 80.93 | 75.59 | 74.39 | 61.45 | 87.33 | 84.94 |
| Ours | 0.37 | 81.93 | 76.93 | 75.71 | 63.41 | 88.00 | 85.45 |

### Table 3: Prior compatibility ablation on balanced test

| Method | Prior module | λ_prior | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tri-view + Text | none | - | 0.38 | 74.37 | 74.37 | 74.17 | 71.92 | 76.43 | 83.48 |
| + Decoupled Prior Prediction | prior prediction only, not used by classifier | 0.20 | 0.33 | 73.83 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.10 | 0.38 | 73.47 | 73.47 | 73.21 | 70.59 | 75.84 | 83.50 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.20 | 0.34 | 75.27 | 75.27 | 74.97 | 72.25 | 77.70 | 84.15 |
| Ours | latent + explicit scalar compatibility | 0.20 | 0.37 | 75.99 | 75.99 | 75.82 | 73.80 | 77.84 | 84.05 |

---

## Result audit

Final paper table sources are archived under:

```text
paper_audit/final_results/
```

Important files:

| File | Meaning |
|---|---|
| `final_tables.md` | final paper tables generated from saved metric JSON files |
| `manifest.csv` | mapping from each paper table row to its source JSON file |
| `table1_comparison_sources/` | source JSON files for Table 1 |
| `table2_input_ablation_sources/` | source JSON files for Table 2 |
| `table3_prior_ablation_sources/` | source JSON files for Table 3 |

Do not manually edit table numbers after audit unless experiments are rerun.

---

## Visualization

Render selected prior-compatibility cases:

```bash
python -m tri_view_compat.tools.render_prior_cases_paper_layout \
  --case_indices 281 168 879 \
  --out_dir tri_view_compat/outputs/paper_figures/case_vis_selected_v3 \
  --card_w 2600 \
  --card_h 900 \
  --gap 34
```

Selected cases:

| Case | idx | Meaning |
|---|---:|---|
| Case A | 281 | successful out-of-context prediction |
| Case B | 168 | successful in-context prediction |
| Case C | 879 | failure case |

---

## Reproducibility checklist

Before reporting results, check:

```bash
# Final checkpoint
ls -lh tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/best.pt

# Final audit files
cat paper_audit/final_results/final_tables.md
cat paper_audit/final_results/manifest.csv

# Table audit
python tri_view_compat/tools/audit_paper_tables.py
python tri_view_compat/tools/check_paper_table_values.py
```

Expected final audit result:

```text
FINAL: ALL OK
```

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{mcp_tcl_coinco,
  title        = {Masked Context Prior-guided Tri-view Compatibility Learning for COinCO Object-Scene Reasoning},
  author       = {Anonymous},
  year         = {2026},
  note         = {Manuscript in preparation}
}
```

Please also cite the original COinCO dataset paper.

---

## Acknowledgements

This project builds on the released COinCO dataset and resources. We thank the COinCO authors for making the dataset, cached features, and baseline resources available.
