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
│   │   ├── clip_triview_text_prior_explicit_only.py
│   │   └── clip_triview_text_prior_v2.py
│   │
│   ├── tools/
│   │   ├── build_triview_splits.py
│   │   ├── build_balanced_test_split.py
│   │   ├── add_replacement_label.py
│   │   ├── eval_threshold_calibrated.py
│   │   ├── eval_threshold_calibrated_prior_aux.py
│   │   ├── eval_threshold_calibrated_explicit_only.py
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
│   ├── train_triview_prior_explicit_only.py
│   └── train_triview_prior_v2.py
│
├── baselines/
│   └── coinco_specialists/
│       ├── run_specialist_scores.py
│       ├── evaluate_specialist_scores.py
│       └── evaluate_fusion_val_threshold.py
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

This repository does **not** include COinCO images, masks, official cached features, released specialist checkpoints, or our trained checkpoints.

### Official COinCO downloads

Please obtain the original data and released resources from the official COinCO project:

- **COinCO dataset:** https://huggingface.co/datasets/COinCO/COinCO-dataset
- **COinCO resources / checkpoints:** https://huggingface.co/datasets/COinCO/COinCO-resources
- **Official COinCO repository:** https://github.com/YangTianze009/COinCO

The dataset and resources are distributed separately. Follow the official COinCO instructions to download and prepare them before running the scripts in this repository.

Optional downloads with the Hugging Face CLI:

```bash
hf download COinCO/COinCO-dataset \
  --repo-type dataset \
  --local-dir COinCO-dataset

hf download COinCO/COinCO-resources \
  --repo-type dataset \
  --local-dir COinCO-resources
```

After preparing the official resources, create symbolic links or place the processed data under the project root as follows:

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

## Backbones and Model Downloads

### MCP-Base / MCP-Compat

Our models use **OpenAI CLIP ViT-B/32** as the shared image-text backbone. The CLIP image encoder is used for the three visual views, and the CLIP text encoder is used for the replacement-object prompt. Both encoders are frozen during training.

- **Backbone:** `openai/clip-vit-base-patch32`
- **Download / model page:** https://huggingface.co/openai/clip-vit-base-patch32

The training scripts accept either the Hugging Face model identifier or a local snapshot path, for example:

```bash
export CLIP_PATH=openai/clip-vit-base-patch32
```

or

```bash
export CLIP_PATH=/path/to/clip-vit-base-patch32
```

Optional local download with the Hugging Face CLI:

```bash
hf download openai/clip-vit-base-patch32 \
  --local-dir models/clip-vit-base-patch32
```

### Released COinCO Specialist Baselines

The released COinCO specialists used in our Table 2 comparison are based on **Qwen2.5-VL-3B-Instruct**. The base processor/backbone and the three released specialist checkpoints are available from Hugging Face:

- **Qwen2.5-VL-3B-Instruct:** https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- **Co-occurrence specialist:** https://huggingface.co/COinCO/Qwen2.5-VL-3B-Co_occurrence
- **Location specialist:** https://huggingface.co/COinCO/Qwen2.5-VL-3B-Location
- **Size specialist:** https://huggingface.co/COinCO/Qwen2.5-VL-3B-Size

Optional local downloads:

```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir models/Qwen2.5-VL-3B-Instruct

hf download COinCO/Qwen2.5-VL-3B-Co_occurrence \
  --local-dir models/Qwen2.5-VL-3B-Co_occurrence

hf download COinCO/Qwen2.5-VL-3B-Location \
  --local-dir models/Qwen2.5-VL-3B-Location

hf download COinCO/Qwen2.5-VL-3B-Size \
  --local-dir models/Qwen2.5-VL-3B-Size
```

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

### MCP-Base

The paper-facing MCP-Base corresponds to the tri-view + replacement-text baseline without geometry features.

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

### Explicit Compatibility Only

This ablation retains prior prediction and the explicit scalar compatibility cues while removing the latent prior-text interaction. Its training script reuses the full-model training protocol and only swaps the model class.

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

The evaluation protocol first selects the best checkpoint by validation Macro-F1 using the default argmax rule. It then calibrates the decision threshold on the validation set over:

```text
{0.01, 0.02, ..., 0.99}
```

The threshold is selected by maximizing validation Macro-F1, breaking ties by Balanced Accuracy and then Accuracy. The selected threshold is fixed for both the original test set and the balanced test subset. Test labels are never used for checkpoint selection or threshold calibration.

### MCP-Compat: Original Test Set

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

### MCP-Compat: Balanced Test Subset

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

Available `model_type` values for `eval_threshold_calibrated.py` include:

```text
baseline
prior_v1
prior_v2
```

For Decoupled Prior, use:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated_prior_aux \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --seed 777
```

For Explicit Compatibility Only, use:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated_explicit_only \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_explicit_only_lam02_bs256 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --seed 777
```

## Released COinCO Specialist Baselines

The current paper comparison includes a score-based reevaluation of the released COinCO Qwen2.5-VL-3B specialist checkpoints:

- **Co-occurrence**
- **Location**
- **Size**

This is a score-based reevaluation of the released specialist checkpoints under our replacement-aware evaluation protocol; it is not a claim that the original COinCO paper used this exact inference procedure.

`run_specialist_scores.py` draws the target box, applies the specialist-specific prompt, and extracts a continuous OOC score from the first discriminative decision token (`In-context` versus `Out-of-context`). The released specialist models are evaluated in BF16. `evaluate_specialist_scores.py` selects each specialist threshold on the full validation set using Macro-F1, with Balanced Accuracy and Accuracy as tie-breakers, then applies the fixed threshold to both test sets.

The input CSVs for specialist scoring must contain:

```text
coco_index
label
replacement_object
image_path
mask_path
```

Set the released model paths and an output directory:

```bash
export QWEN_BASE=/path/to/Qwen2.5-VL-3B-Instruct
export CO_MODEL=/path/to/Qwen2.5-VL-3B-Co_occurrence
export LOC_MODEL=/path/to/Qwen2.5-VL-3B-Location
export SIZE_MODEL=/path/to/Qwen2.5-VL-3B-Size
export SPECIALIST_RESULTS=baselines/coinco_specialists/results
mkdir -p "$SPECIALIST_RESULTS"
```

Generate validation and test scores for Co-occurrence:

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/coinco_specialists/run_specialist_scores.py \
  --specialist cooccurrence \
  --model_path "$CO_MODEL" \
  --processor_path "$QWEN_BASE" \
  --csv tri_view_compat/outputs/splits/val.csv \
  --output "$SPECIALIST_RESULTS/cooccurrence_val_scores.jsonl"

CUDA_VISIBLE_DEVICES=0 python baselines/coinco_specialists/run_specialist_scores.py \
  --specialist cooccurrence \
  --model_path "$CO_MODEL" \
  --processor_path "$QWEN_BASE" \
  --csv tri_view_compat/outputs/splits/test.csv \
  --output "$SPECIALIST_RESULTS/cooccurrence_test_scores.jsonl"
```

Repeat the same two commands for `location` and `size`, changing `--specialist`, `--model_path`, and the output filenames to:

```text
location_val_scores.jsonl
location_test_scores.jsonl
size_val_scores.jsonl
size_test_scores.jsonl
```

Evaluate each specialist with a validation-selected threshold:

```bash
for s in cooccurrence location size; do
  python baselines/coinco_specialists/evaluate_specialist_scores.py \
    --val "$SPECIALIST_RESULTS/${s}_val_scores.jsonl" \
    --test "$SPECIALIST_RESULTS/${s}_test_scores.jsonl" \
    --balanced_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
    --output "$SPECIALIST_RESULTS/${s}_score_metrics.json"
done
```

### Score Fusion

Our **Score Fusion** is not an author-released fourth specialist. It is our calibrated fusion of the three released specialist scores. Each specialist score is first aligned relative to its own validation-selected threshold in logit space; the maximum aligned specialist response is then converted back to a continuous fusion score. A final fusion threshold is selected on the full validation set using the same Macro-F1 / BAcc / Acc rule.

```bash
python baselines/coinco_specialists/evaluate_fusion_val_threshold.py \
  --results_dir "$SPECIALIST_RESULTS" \
  --balanced_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --output "$SPECIALIST_RESULTS/coinco_fusion_val_calibrated_metrics.json"
```

## Additional Cached-Feature Baselines

The repository also retains our earlier COinCO-style cached-feature baselines:

- `COinCO-VisualNet*`: VAE latent features
- `COinCO-SemanticNet*`: surrounding-object and replacement-object embeddings
- `COinCO-VisSemanticNet*`: concatenation of visual and semantic features

Run, for example:

```bash
python coinco_v1_baselines/train_cached_context_baseline.py \
  --mode semantic \
  --train_csv tri_view_compat/outputs/splits/train.csv \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --balanced_test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --output_dir coinco_v1_baselines/outputs/semantic
```

Replace `--mode semantic` with `visual` or `vissemantic` for the other two variants. These cached-feature reimplementations are retained for additional analysis and are separate from the released-specialist score reevaluation used in the current main comparison.

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

Balanced-set Balanced Accuracy equals Accuracy and is omitted.

| Method | Acc | M-F1 | F1-IC | F1-OOC | AUC |
|---|---:|---:|---:|---:|---:|
| Co-occurrence† | 71.85 | 71.52 | 68.42 | 74.61 | 79.03 |
| Location† | 70.23 | 69.70 | 65.70 | 73.71 | 79.29 |
| Size† | 67.00 | 65.97 | 60.07 | 71.88 | 75.64 |
| Score Fusion† | 69.69 | 69.27 | 65.65 | 72.89 | 77.11 |
| MCP-Base | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| **MCP-Compat** | **75.99** | **75.82** | **73.80** | **77.84** | 84.05 |

† Score-based reevaluation of the released COinCO specialist checkpoints. Score Fusion is our threshold-aligned calibrated fusion of the three specialist scores.

## Ablation Results

Ablation on the balanced test subset:

| Variant | Th. | Acc | M-F1 | F1-IC | F1-OOC | AUC |
|---|---:|---:|---:|---:|---:|---:|
| MCP-Base | 0.38 | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| Decoupled Prior | 0.33 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| Explicit | 0.33 | 73.56 | 73.41 | 71.40 | 75.42 | 81.62 |
| Latent | 0.34 | 75.27 | 74.97 | 72.25 | 77.70 | **84.15** |
| **MCP-Compat** | **0.37** | **75.99** | **75.82** | **73.80** | **77.84** | 84.05 |

All prior-based rows in this table use `lambda_prior = 0.2`. Decoupled Prior supervises the masked-context prior without feeding it to the classifier; Explicit uses only the explicit scalar compatibility embedding; Latent uses only latent prior-text compatibility; MCP-Compat combines latent and explicit compatibility.

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

The figure demonstrates two cases where MCP-Base fails but MCP-Compat succeeds:

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
- The reported MCP-Base checkpoint uses dropout 0.2; the prior-based variants use dropout 0.1.
- The reported prior-based variants use `lambda_prior = 0.2` unless explicitly stated otherwise.
- Best checkpoints are selected by validation Macro-F1 before threshold calibration.
- Test labels are never used for checkpoint selection or threshold calibration.
- AUC is computed from continuous predicted scores and is threshold-independent.
- The original-object category is used only for training the prior branch and is not used at inference.
- Released COinCO specialist checkpoints are evaluated with BF16 first-decision-token scoring in the specialist pipeline above.

## License

This repository is released under the license specified in `LICENSE`.

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
