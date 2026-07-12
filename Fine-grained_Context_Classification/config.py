"""
Centralized path configuration for the SFT VLM project.

Before running any script, update the paths below to match your environment.
All other scripts import paths from this file — no hardcoded paths elsewhere.
"""

import os

# ============================================================================
# ROOT DIRECTORIES — Update these to match your environment
# ============================================================================

# Root directory for all data (COCO images, CSVs, inpainted images, etc.)
DATA_ROOT = "/path/to/your/data"

# Root directory for model weights
WEIGHTS_ROOT = "/path/to/your/weights"

# Root directory for SFT training data (webdatasets) — included in this repo
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SFT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data_generation", "context_reasoning-res", "SFT_data")

# Fine-tuned model outputs live inside each bias type's train/ directory.
# After training, final_model/ is produced automatically.
# FINETUNED_MODELS_ROOT is no longer used — see per-model paths below.

# Root directory for context reasoning outputs
CONTEXT_REASONING_OUTPUT_ROOT = os.path.join(DATA_ROOT, "context_reasoning_outputs")

# ============================================================================
# BASE MODEL WEIGHTS
# ============================================================================

QWEN_3B_MODEL = os.path.join(WEIGHTS_ROOT, "Qwen2.5-VL-3B-Instruct")
QWEN_72B_MODEL = os.path.join(WEIGHTS_ROOT, "Qwen2.5-VL-72B-Instruct")
INTERNVL_38B_MODEL = os.path.join(WEIGHTS_ROOT, "InternVL3_5-38B")
MOLMO_72B_MODEL = os.path.join(WEIGHTS_ROOT, "Molmo-72B-0924")

# ============================================================================
# CSV FILES
# ============================================================================

# CSV files — included in this repo under csv_mappings/
TRAINING_CSV = os.path.join(PROJECT_ROOT, "csv_mappings", "training_inpainting_info.csv")
TESTING_CSV = os.path.join(PROJECT_ROOT, "csv_mappings", "testing_inpainting_info.csv")

# ============================================================================
# COCO DATASET
# ============================================================================

COCO_ROOT = os.path.join(DATA_ROOT, "coco")
COCO_TRAIN_IMAGES = os.path.join(COCO_ROOT, "images", "train2017")
COCO_VAL_IMAGES = os.path.join(COCO_ROOT, "images", "val2017")
COCO_TRAIN_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
COCO_VAL_ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

# ============================================================================
# SFT TRAINING DATA (WebDatasets)
# ============================================================================

LOCATION_WEBDATASET_DIR = os.path.join(SFT_DATA_ROOT, "location_webdataset")
SIZE_WEBDATASET_DIR = os.path.join(SFT_DATA_ROOT, "size_webdataset")
CO_OCCURRENCE_WEBDATASET_DIR = os.path.join(SFT_DATA_ROOT, "co_occurrence_webdataset")


# ============================================================================
# FINE-TUNED MODELS
# ============================================================================

LOCATION_FINETUNED_MODEL = os.path.join(PROJECT_ROOT, "location", "train", "qwen25_finetuned_location_specialist_multi_gpu", "final_model")


SIZE_FINETUNED_MODEL = os.path.join(PROJECT_ROOT, "size", "train", "qwen25_finetuned_size_specialist_multi_gpu", "final_model")
CO_OCCURRENCE_FINETUNED_MODEL = os.path.join(PROJECT_ROOT, "co_occurrence", "train", "qwen25_finetuned_co_occurrence_specialist_multi_gpu", "final_model")

# ============================================================================
# ROUND-2: CONTEXT REASONING
# ============================================================================

TRAINING_IMAGES_WITH_BBOX = os.path.join(DATA_ROOT, "training_images_with_bbox")
TESTING_IMAGES_WITH_BBOX = os.path.join(DATA_ROOT, "testing_images_with_bbox")

# Directory containing context reasoning result CSVs (internvl/, molmo/, qwen/ subdirs)
CONTEXT_REASONING_RES_DIR = os.path.join(DATA_ROOT, "context_reasoning_res")

# External coco_utils module path (for ConvertCocoPolysToMask)
COCO_UTILS_DIR = os.path.join(DATA_ROOT, "coco_utils")

# ============================================================================
# FOCUS ANALYSIS (location only)
# ============================================================================

FOCUS_DATA_DIR = os.path.join(DATA_ROOT, "focus_data")
