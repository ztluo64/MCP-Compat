import os
import pandas as pd

ROOT = "/home/ubuntu/ztl/COinCO"
OUT_DIR = os.path.join(ROOT, "tri_view_compat/outputs/splits")
os.makedirs(OUT_DIR, exist_ok=True)

# Main Task 1 labels
context_train_path = os.path.join(ROOT, "task_data/context_prediction/inpainting_only/training_val_data.csv")
context_test_path = os.path.join(ROOT, "task_data/context_prediction/testing_data.csv")

# Replacement metadata
inpaint_train_path = os.path.join(ROOT, "task_data/inpainting_info/training_inpainting_info.csv")
inpaint_test_path = os.path.join(ROOT, "task_data/inpainting_info/testing_inpainting_info.csv")

# Task 2 prior labels, used here mainly for splitting and future prior branch
prior_train_path = os.path.join(ROOT, "task_data/objects_from_context_prediction/training_data.csv")
prior_val_path = os.path.join(ROOT, "task_data/objects_from_context_prediction/validation_data.csv")
prior_test_path = os.path.join(ROOT, "task_data/objects_from_context_prediction/testing_data.csv")

def read_csv(path):
    return pd.read_csv(path, dtype={"coco_index": str})

def build_split(context_df, inpaint_df, prior_df, split_name, image_dir, mask_dir):
    context_df = context_df.copy()
    inpaint_df = inpaint_df.copy()
    prior_df = prior_df.copy()

    context_df["coco_index"] = context_df["coco_index"].astype(str)
    inpaint_df["coco_index"] = inpaint_df["coco_index"].astype(str)
    prior_df["coco_index"] = prior_df["coco_index"].astype(str)

    # Rename Task 2 label as prior_label
    prior_df = prior_df.rename(columns={"label": "prior_label"})

    # Keep only samples that have context label + replacement metadata + prior label
    df = context_df.merge(inpaint_df, on="coco_index", how="inner")
    df = df.merge(prior_df[["coco_index", "prior_label"]], on="coco_index", how="inner")

    df["image_path"] = df["coco_index"].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    df["mask_path"] = df["coco_index"].apply(lambda x: os.path.join(mask_dir, f"{x}.png"))
    df["split"] = split_name

    # Check image and mask existence
    df["image_exists"] = df["image_path"].apply(os.path.exists)
    df["mask_exists"] = df["mask_path"].apply(os.path.exists)

    missing_img = (~df["image_exists"]).sum()
    missing_mask = (~df["mask_exists"]).sum()

    if missing_img > 0 or missing_mask > 0:
        print(f"[WARN] {split_name}: missing_img={missing_img}, missing_mask={missing_mask}")
        df = df[df["image_exists"] & df["mask_exists"]].copy()

    keep_cols = [
        "coco_index",
        "label",
        "class_name",
        "object_index",
        "replacement_object",
        "prior_label",
        "image_path",
        "mask_path",
        "split",
    ]
    df = df[keep_cols].reset_index(drop=True)

    return df

context_train = read_csv(context_train_path)
context_test = read_csv(context_test_path)

inpaint_train = read_csv(inpaint_train_path)
inpaint_test = read_csv(inpaint_test_path)

prior_train = read_csv(prior_train_path)
prior_val = read_csv(prior_val_path)
prior_test = read_csv(prior_test_path)

train_df = build_split(
    context_train,
    inpaint_train,
    prior_train,
    "train",
    os.path.join(ROOT, "task_data/images/training_val_images"),
    os.path.join(ROOT, "task_data/masks/bbox_masks_training_val"),
)

val_df = build_split(
    context_train,
    inpaint_train,
    prior_val,
    "val",
    os.path.join(ROOT, "task_data/images/training_val_images"),
    os.path.join(ROOT, "task_data/masks/bbox_masks_training_val"),
)

test_df = build_split(
    context_test,
    inpaint_test,
    prior_test,
    "test",
    os.path.join(ROOT, "task_data/images/testing_images"),
    os.path.join(ROOT, "task_data/masks/bbox_masks_testing"),
)

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(out_path, index=False)

    print(f"\n===== {name} =====")
    print("saved:", out_path)
    print("shape:", df.shape)
    print("label counts:")
    print(df["label"].value_counts().sort_index().to_string())
    print("prior_label unique:", df["prior_label"].nunique())
    print("head:")
    print(df.head(5).to_string(index=False))

print("\nDone.")
