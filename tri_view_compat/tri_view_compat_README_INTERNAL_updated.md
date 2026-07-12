# tri_view_compat Internal README

本 README 用于记录 `tri_view_compat/` 目录中各文件、模型、实验结果和复现实验流程。它面向项目内部整理、论文复现、给导师/合作者说明实验来源，不一定作为公开 GitHub README 的最终版本。

> 当前版本同步最终论文叙事：
> - **Aux-only Prior → Decoupled Prior Prediction**
> - **Table 1 从 balanced-only 更新为 Original test + Balanced test 双测试集主表**
> - **最终表格来源统一整理到 `paper_audit/final_results/`**

---

## 1. Project scope

`tri_view_compat/` 是 COinCO replacement-aware object-scene compatibility classification 的主实验目录。该目录实现论文主方法：

**Masked Context Prior-guided Tri-view Compatibility Learning**

论文中的最终方法为：

```text
Ours
= Tri-view + Text
+ Latent Prior Compatibility
+ Explicit Scalar Prior Compatibility
+ prior loss weight λ_prior = 0.2
```

最终 checkpoint：

```text
tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/best.pt
```

注意：代码文件和 checkpoint 中仍包含 `prior_v2`、`prior_aux` 等工程命名，但论文正文中不应使用 `Prior v1 / Prior v2 / Aux-only Prior` 作为方法名称。论文中的正式命名如下：

| Code / checkpoint name | Paper-facing name |
|---|---|
| `triview_text_no_geo_bs256` | Tri-view + Text |
| `triview_text_prior_aux_lam02_bs256` | Decoupled Prior Prediction |
| `triview_text_prior_lam01_bs256` | Latent Prior Compatibility, λ_prior=0.10 |
| `triview_text_prior_lam02_bs256` | Latent Prior Compatibility, λ_prior=0.20 |
| `triview_text_prior_v2_lam02_bs256` | Ours / Latent + Explicit Scalar Prior Compatibility |
| `clip_triview_text_prior_aux.py` | Decoupled Prior Prediction implementation |
| `clip_triview_text_prior.py` | Latent Prior Compatibility implementation |
| `clip_triview_text_prior_v2.py` | Ours implementation |

---

## 2. Current directory structure

Current structure snapshot:

```text
tri_view_compat/
├── configs/
├── datasets/
│   ├── coinco_triview_dataset.py
│   ├── coinco_triview_dataset_v2.py
│   └── __init__.py
├── models/
│   ├── clip_triview_baseline.py
│   ├── clip_triview_text_prior_aux.py
│   ├── clip_triview_text_prior.py
│   ├── clip_triview_text_prior_v2.py
│   └── __init__.py
├── tools/
│   ├── add_replacement_label.py
│   ├── audit_paper_tables.py
│   ├── bootstrap_ci.py
│   ├── build_balanced_test_split.py
│   ├── build_triview_splits.py
│   ├── check_paper_table_values.py
│   ├── eval_threshold_calibrated.py
│   ├── eval_threshold_calibrated_prior_aux.py
│   ├── inspect_task1_data.py
│   ├── render_prior_cases_paper_layout.py
│   ├── visualize_prior_cases.py
│   └── visualize_triview_samples.py
├── outputs/
│   ├── checkpoints/
│   ├── splits/
│   ├── splits_balanced/
│   ├── case_vis/
│   ├── paper_figures/
│   ├── bootstrap/
│   └── vis_triview_test/
├── logs/
├── train_triview.py
├── train_triview_prior.py
├── train_triview_prior_aux.py
├── train_triview_prior_v2.py
└── tri_view_compat_README_final.md
```

`__pycache__/` 和 `.pyc` 文件是 Python 自动生成缓存，不属于论文或复现实验的核心内容。整理公开代码时可以删除：

```bash
find tri_view_compat -type d -name "__pycache__" -prune -exec rm -rf {} +
find tri_view_compat -type f -name "*.pyc" -delete
```

---

## 3. Data dependencies

本目录依赖 COinCO released resources，并假设项目根目录为：

```text
/home/ubuntu/ztl/COinCO
```

关键软链接：

```text
task_data -> /home/ubuntu/ztl/dataset/COinCO/COinCO-resources/task_data
checkpoints -> /home/ubuntu/ztl/dataset/COinCO/COinCO-resources/checkpoints
```

主实验使用的 split 文件位于：

```text
tri_view_compat/outputs/splits/train.csv
tri_view_compat/outputs/splits/val.csv
tri_view_compat/outputs/splits/test.csv
tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv
```

当前 split 统计：

| Split | Samples | Label 0 | Label 1 |
|---|---:|---:|---:|
| Train | 76,256 | 16,088 | 60,168 |
| Val | 19,064 | 3,963 | 15,101 |
| Original test | 2,402 | 556 | 1,846 |
| Balanced test | 1,112 | 556 | 556 |

字段说明：

| Column | Meaning |
|---|---|
| `coco_index` | COinCO sample id / cache filename id |
| `label` | compatibility label, 0=in-context, 1=out-of-context |
| `class_name` | original / expected object category name |
| `object_index` | target object index in metadata |
| `replacement_object` | actual inserted object category name |
| `prior_label` | original object category index, used only as prior branch training target |
| `replacement_label` | replacement object category index, generated using train-only mapping |
| `image_path` | inpainted image path |
| `mask_path` | target region / bounding-box mask path |
| `split` | train / val / test |

Important no-leakage note:

- `prior_label` is used only during training as the supervision target for masked-context prior prediction.
- `prior_label` is not used as input during test.
- `replacement_label` is deterministically mapped from `replacement_object` using class-name/category-label correspondence built from the training split.
- The released metadata does not expose original COCO source-image identifiers. We follow the official split files and do not claim source-image-level disjointness beyond the released split.
- Original test is the complete replacement-aware test set. Balanced test is a class-balance robustness check built from the original test set by keeping all 556 in-context samples and sampling 556 out-of-context samples with seed 777.

---

## 4. Dataset implementations

### `datasets/coinco_triview_dataset.py`

Main dataset for CLIP tri-view baselines. It constructs:

```text
full_image
object_crop
masked_context
geometry
label
prior_label
replacement_object
class_name
text_prompt
```

Tri-view construction:

```text
I_f = full image
I_o = object crop from target region mask / bounding box
I_c = masked context, where target region is removed / filled with neutral value
```

### `datasets/coinco_triview_dataset_v2.py`

Extends the basic dataset with:

```text
replacement_label
```

This is required by Ours, because explicit scalar compatibility needs to read:

```text
p(replacement | context)
```

from the predicted expected-object distribution.

Before public release, the code should not silently default missing `replacement_label` to 0. It should raise an error instead:

```python
if replacement_label is None:
    raise ValueError(
        "replacement_label is required for explicit prior-replacement compatibility."
    )
```

---

## 5. Model implementations

### `models/clip_triview_baseline.py`

Used for:

| Experiment | Inputs |
|---|---|
| Full only | full image |
| Object crop only | object crop |
| Masked context only | masked context |
| Tri-view | full image + object crop + masked context |
| Tri-view + Text | full image + object crop + masked context + replacement text |

Backbone:

```text
Frozen CLIP ViT-B/32 image encoder
Frozen CLIP text encoder
Trainable MLP classifier
```

### `models/clip_triview_text_prior_aux.py`

Paper-facing name: **Decoupled Prior Prediction**.

It adds a masked-context prior prediction head:

```text
masked context feature -> 80-way expected object distribution
```

Training objective:

```text
L = L_cls + λ_prior L_prior
```

but the predicted prior is not fed into the final classifier, and no prior-replacement compatibility feature is constructed.

Purpose:

```text
This is a control setting to test whether prior prediction alone, when decoupled from the final decision branch, is sufficient.
```

### `models/clip_triview_text_prior.py`

Paper-facing name: **Latent Prior Compatibility**.

It predicts expected-object prior from masked context and compares it with replacement text in latent space:

```text
e_p = φ_p(p)
e_t = φ_t(f_t)
c_lat = [e_p; e_t; e_p ⊙ e_t; |e_p - e_t|]
```

### `models/clip_triview_text_prior_v2.py`

Paper-facing name: **Ours / Latent + Explicit Scalar Prior Compatibility**.

In addition to latent prior-text compatibility, it uses explicit scalar compatibility cues:

```text
p(replacement | context)
log p(replacement | context)
max prior probability
margin
normalized entropy
ratio
```

This is the final model used in the paper.

Final checkpoint:

```text
tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/best.pt
```

---

## 6. Training scripts

### 6.1 Baseline and tri-view variants

Script:

```text
tri_view_compat/train_triview.py
```

Example commands:

```bash
# Full only
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/full_only_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --no_crop --no_masked --no_geo --no_text

# Object crop only
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/crop_only_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --no_full --no_masked --no_geo --no_text

# Masked context only
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/masked_only_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --no_full --no_crop --no_geo --no_text

# Tri-view without text
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_no_geo_text_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --no_geo --no_text

# Tri-view + Text
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_no_geo_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --no_geo
```

### 6.2 Decoupled Prior Prediction

Script:

```text
tri_view_compat/train_triview_prior_aux.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior_aux \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

### 6.3 Latent Prior Compatibility

Script:

```text
tri_view_compat/train_triview_prior.py
```

Commands:

```bash
# λ_prior = 0.10
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_lam01_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.1 \
  --no_geo

# λ_prior = 0.20
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

### 6.4 Ours

Script:

```text
tri_view_compat/train_triview_prior_v2.py
```

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tri_view_compat.train_triview_prior_v2 \
  --clip_name /home/ubuntu/ztl/models/clip-vit-base-patch32 \
  --output_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --epochs 10 \
  --batch_size 256 \
  --num_workers 8 \
  --lr 1e-3 \
  --prior_loss_weight 0.2 \
  --no_geo
```

---

## 7. Evaluation protocol

All reported binary classification results use validation-calibrated threshold:

```text
threshold ∈ {0.01, 0.02, ..., 0.99}
selected once by maximizing Macro-F1 on the validation set
fixed for both the original replacement-aware test set and the balanced test subset
```

Test labels are not used for threshold selection. AUC is calculated from predicted probabilities and is threshold-independent.

Checkpoint selection is based on validation Macro-F1 during training. After checkpoint selection, validation-based threshold calibration is performed for final reporting.

### Standard evaluation

Script:

```text
tri_view_compat/tools/eval_threshold_calibrated.py
```

Example original test evaluation:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits/test.csv \
  --batch_size 256 \
  --num_workers 8 \
  --metric macro_f1

cp tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics.json \
   tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics_original_test.json
```

Example balanced test evaluation:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256 \
  --model_type prior_v2 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --metric macro_f1

cp tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics.json \
   tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json
```

### Decoupled Prior Prediction evaluation

Script:

```text
tri_view_compat/tools/eval_threshold_calibrated_prior_aux.py
```

Example:

```bash
python -m tri_view_compat.tools.eval_threshold_calibrated_prior_aux \
  --ckpt_dir tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256 \
  --val_csv tri_view_compat/outputs/splits/val.csv \
  --test_csv tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv \
  --batch_size 256 \
  --num_workers 8 \
  --metric macro_f1 \
  --output_json tri_view_compat/outputs/checkpoints/triview_text_prior_aux_lam02_bs256/threshold_metrics_balanced_test.json
```

---

## 8. Paper table audit

### Generate paper tables from saved JSON

```bash
python tri_view_compat/tools/audit_paper_tables.py | tee tri_view_compat/logs/audit_paper_tables.log
```

### Strict value consistency check

```bash
python tri_view_compat/tools/check_paper_table_values.py | tee tri_view_compat/logs/check_paper_table_values.log
```

Expected final line:

```text
FINAL: ALL OK
```

Current audit status:

```text
FINAL: ALL OK
```

This means all paper table values match saved metric JSON files up to rounding tolerance.

---

## 9. Final results audit folder

Final paper table sources are additionally archived under:

```text
paper_audit/final_results/
```

Recommended structure:

```text
paper_audit/final_results/
├── README.md
├── final_tables.md
├── manifest.csv
├── table1_comparison_sources/
├── table2_input_ablation_sources/
└── table3_prior_ablation_sources/
```

Important files:

| File / folder | Meaning |
|---|---|
| `final_tables.md` | final paper tables generated from saved metric JSON files |
| `manifest.csv` | mapping from each paper table row to its source JSON file |
| `table1_comparison_sources/` | source JSON files for Table 1 |
| `table2_input_ablation_sources/` | source JSON files for Table 2 |
| `table3_prior_ablation_sources/` | source JSON files for Table 3 |

All table values should be copied from:

```text
paper_audit/final_results/final_tables.md
```

Do not manually edit paper table values after this audit unless experiments are rerun.

---

## 10. Final paper results

### 10.1 Table 1: COinCO-style baselines and structured variants

This table reports both the original replacement-aware test set and the balanced test subset. The original test set is the main evaluation protocol, while the balanced subset is used as a class-balance robustness check.

| Type | Method | Th. | Original Acc | Original Bal Acc | Original Macro-F1 | Original AUC | Balanced Acc | Balanced Macro-F1 | Balanced AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COinCO-style | COinCO-VisualNet* | 0.45 | 68.69 | 55.94 | 55.95 | 62.49 | 56.65 | 53.90 | 63.69 |
| COinCO-style | COinCO-SemanticNet* | 0.37 | 76.94 | 68.22 | 67.99 | 78.60 | 68.35 | 67.47 | 77.86 |
| COinCO-style | COinCO-VisSemanticNet* | 0.42 | 74.65 | 68.67 | 67.06 | 76.59 | 68.08 | 67.72 | 75.19 |
| Ours variant | Tri-view + Text | 0.38 | 81.43 | 75.73 | 74.80 | 85.18 | 74.28 | 74.06 | 83.52 |
| Ours | Ours | 0.37 | 81.93 | 76.93 | 75.71 | 85.45 | 75.99 | 75.82 | 84.05 |

Notes:

- `*` denotes our reimplementation using released COinCO cached visual and semantic embeddings under the unified replacement-aware protocol.
- Published COinCO numbers are not directly comparable because of different data construction and evaluation settings.
- Tri-view + Text is our internal strong baseline without masked-context prior compatibility.
- Ours is the final proposed method.
- `Th.` is selected once on the validation set by maximizing Macro-F1 and then fixed for both test sets.

### 10.2 Table 2: Input decomposition ablation on original test

| Method | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full only | 0.42 | 75.85 | 63.87 | 64.46 | 44.34 | 84.58 | 72.72 |
| Object crop only | 0.36 | 75.77 | 68.21 | 67.38 | 50.84 | 83.92 | 78.17 |
| Masked context only | 0.44 | 70.61 | 59.13 | 59.05 | 37.30 | 80.80 | 66.63 |
| Tri-view | 0.33 | 78.06 | 69.51 | 69.38 | 53.07 | 85.68 | 80.22 |
| Tri-view + Text | 0.38 | 81.43 | 75.73 | 74.80 | 61.88 | 87.73 | 85.18 |
| Ours | 0.37 | 81.93 | 76.93 | 75.71 | 63.41 | 88.00 | 85.45 |

### 10.3 Table 3: Prior compatibility ablation on balanced test

| Method | Prior module | λ_prior | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tri-view + Text | none | - | 0.38 | 74.28 | 74.28 | 74.06 | 71.68 | 76.44 | 83.52 |
| + Decoupled Prior Prediction | prior prediction only, not used by classifier | 0.20 | 0.33 | 73.83 | 73.83 | 73.60 | 71.16 | 76.05 | 82.65 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.10 | 0.38 | 73.47 | 73.47 | 73.21 | 70.59 | 75.84 | 83.50 |
| + Latent Prior Compat. | latent prior-text compatibility | 0.20 | 0.34 | 75.27 | 75.27 | 74.97 | 72.25 | 77.70 | 84.15 |
| Ours | latent + explicit scalar compatibility | 0.20 | 0.37 | 75.99 | 75.99 | 75.82 | 73.80 | 77.84 | 84.05 |

Notes:

- `triview_text_prior_aux_lam02_bs256` is paper-facing **Decoupled Prior Prediction**.
- Decoupled Prior Prediction trains an additional masked-context prior prediction head, but does not feed the predicted prior or any prior-replacement compatibility feature into the final classifier.
- AUC of Latent Prior Compatibility with λ_prior=0.20 is slightly higher than Ours, while Ours achieves the best threshold-dependent balanced classification metrics.

---

## 11. Visualization

### 11.1 Tri-view sample visualization

Script:

```text
tri_view_compat/tools/visualize_triview_samples.py
```

Output:

```text
tri_view_compat/outputs/vis_triview_test/
```

This is used to inspect whether full image, object crop, masked context and mask processing are correct.

### 11.2 Prior case visualization

Script:

```text
tri_view_compat/tools/visualize_prior_cases.py
```

Output:

```text
tri_view_compat/outputs/case_vis/prior_v2_original/
```

This generates raw case visualizations and `case_summary.csv`.

### 11.3 Paper layout rendering

Script:

```text
tri_view_compat/tools/render_prior_cases_paper_layout.py
```

Recommended command:

```bash
python -m tri_view_compat.tools.render_prior_cases_paper_layout \
  --case_indices 281 168 879 \
  --out_dir tri_view_compat/outputs/paper_figures/case_vis_selected_v3 \
  --card_w 2600 \
  --card_h 900 \
  --gap 34
```

Recommended selected cases:

| Case | idx | Meaning |
|---|---:|---|
| Case A | 281 | successful out-of-context prediction |
| Case B | 168 | successful in-context prediction |
| Case C | 879 | failure case |

Important visualization logic:

```text
correct in-context + high p(replacement|context)
    -> High p supports a context match

correct out-of-context + low p(replacement|context)
    -> Low p indicates a context mismatch

failure
    -> incorrect or ambiguous context prior may mislead final decision
```

---

## 12. COinCO-style baselines

COinCO-style baselines are implemented outside `tri_view_compat/` in:

```text
coinco_v1_baselines/
```

They use released cached embeddings:

```text
task_data/cache/cached_embeddings_train_val_cp
task_data/cache/cached_embeddings_testing_cp
```

Each cached `.pt` contains:

```text
latent1
latent2
objects_embeddings
replacement_embedding
label
```

Paper-facing baselines:

| Method | Features |
|---|---|
| COinCO-VisualNet* | latent1 + latent2 |
| COinCO-SemanticNet* | objects_embeddings + replacement_embedding |
| COinCO-VisSemanticNet* | latent1 + latent2 + objects_embeddings + replacement_embedding |

Current original and balanced test results are included in Table 1.

Use this paper note:

```text
* denotes our reimplementation using the released COinCO cached visual and semantic embeddings under the unified replacement-aware protocol. Published COinCO numbers are not directly comparable because of different data construction and evaluation settings.
```

---

## 13. Bootstrap analysis

Bootstrap scripts are available but not used as main paper evidence because 95% confidence intervals overlap zero.

Scripts / outputs:

```text
tri_view_compat/tools/bootstrap_ci.py
tri_view_compat/outputs/bootstrap/bootstrap_original_triview_text_vs_ours.json
tri_view_compat/outputs/bootstrap/bootstrap_balanced_triview_text_vs_ours.json
```

Recommended paper wording if mentioned:

```text
Bootstrap analysis shows consistently positive point-estimate improvements, especially on the balanced subset, although the 95% intervals overlap zero.
```

Avoid claiming statistical significance.

---

## 14. Recommended paper-facing terminology

Use these terms in paper:

```text
COinCO-style baselines
replacement-aware protocol
target-aware / replacement-aware object-scene compatibility classification
Tri-view object-context decomposition
replacement object semantic conditioning
masked-context object prior
Decoupled Prior Prediction
Latent Prior Compatibility
Explicit Scalar Prior Compatibility
expected-vs-inserted object comparison
decoupled prior prediction alone is insufficient
```

Avoid these terms in paper body:

```text
Prior v1
Prior v2
v1 / v2
Aux-only Prior
auxiliary supervision alone
auxiliary prior supervision
significantly outperforms
statistically significant
```

Use cautious wording:

```text
consistently improves
achieves higher point estimates
obtains better balanced classification performance
```

---

## 15. Reproducibility checklist

Before final paper submission:

### 15.1 Confirm final checkpoint

```bash
ls -lh tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/best.pt
```

### 15.2 Confirm metric JSON files

```bash
# COinCO-style baselines, original and balanced
ls coinco_v1_baselines/outputs/coinco_visualnet_bs256/threshold_metrics_original_test.json
ls coinco_v1_baselines/outputs/coinco_visualnet_bs256/threshold_metrics_balanced_test.json

ls coinco_v1_baselines/outputs/coinco_semanticnet_bs256/threshold_metrics_original_test.json
ls coinco_v1_baselines/outputs/coinco_semanticnet_bs256/threshold_metrics_balanced_test.json

ls coinco_v1_baselines/outputs/coinco_vissemanticnet_bs256/threshold_metrics_original_test.json
ls coinco_v1_baselines/outputs/coinco_vissemanticnet_bs256/threshold_metrics_balanced_test.json

# Main structured variants
ls tri_view_compat/outputs/checkpoints/triview_text_no_geo_bs256/threshold_metrics_original_test.json
ls tri_view_compat/outputs/checkpoints/triview_text_no_geo_bs256/threshold_metrics_balanced_test.json
ls tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics_original_test.json
ls tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json
```

### 15.3 Confirm final result audit folder

```bash
cat paper_audit/final_results/final_tables.md
cat paper_audit/final_results/manifest.csv
```

### 15.4 Confirm table audit

```bash
python tri_view_compat/tools/audit_paper_tables.py
python tri_view_compat/tools/check_paper_table_values.py
```

Expected final result:

```text
FINAL: ALL OK
```

### 15.5 Confirm selected visualization figure

```bash
ls tri_view_compat/outputs/paper_figures/case_vis_selected_v3/selected_cases_paper_layout.png
```

### 15.6 Rule

Do not manually edit table numbers after audit unless experiments are rerun.
