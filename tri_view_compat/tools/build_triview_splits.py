from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def add_paths(df: pd.DataFrame, task_data_root: Path, split_name: str) -> pd.DataFrame:
    df = df.copy()

    if split_name in {"train", "val"}:
        image_dir = task_data_root / "images" / "training_val_images"
        mask_dir = task_data_root / "masks" / "bbox_masks_training_val"
    elif split_name == "test":
        image_dir = task_data_root / "images" / "testing_images"
        mask_dir = task_data_root / "masks" / "bbox_masks_testing"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    df["image_path"] = df["coco_index"].apply(lambda x: str((image_dir / f"{int(x)}.png").resolve()))
    df["mask_path"] = df["coco_index"].apply(lambda x: str((mask_dir / f"{int(x)}.png").resolve()))
    df["split"] = split_name
    return df


def build_split(
    context_df: pd.DataFrame,
    inpaint_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    task_data_root: Path,
    split_name: str,
) -> pd.DataFrame:
    prior_df = prior_df.rename(columns={"label": "prior_label"})

    merged = context_df.merge(
        inpaint_df[["coco_index", "class_name", "object_index", "replacement_object"]],
        on="coco_index",
        how="inner",
    )
    merged = merged.merge(
        prior_df[["coco_index", "prior_label"]],
        on="coco_index",
        how="inner",
    )

    merged = add_paths(merged, task_data_root, split_name)

    columns = [
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
    merged = merged[columns].sort_values("coco_index").reset_index(drop=True)
    return merged


def save_split(df: pd.DataFrame, output_dir: Path, split_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n===== {split_name} =====")
    print(f"saved: {out_path}")
    print(f"shape: {df.shape}")
    print("label counts:")
    print(df["label"].value_counts().sort_index())
    print(f"prior_label unique: {df['prior_label'].nunique()}")
    print("head:")
    print(df.head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build replacement-aware tri-view COinCO train/val/test splits."
    )
    parser.add_argument(
        "--task_data_root",
        type=Path,
        default=Path("task_data"),
        help="Path to COinCO task_data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tri_view_compat/outputs/splits"),
        help="Directory to save train.csv, val.csv, and test.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_data_root = args.task_data_root.resolve()
    output_dir = args.output_dir

    context_train_path = task_data_root / "context_prediction" / "balanced" / "training_val_data.csv"
    context_test_path = task_data_root / "context_prediction" / "testing_data.csv"

    inpaint_train_path = task_data_root / "inpainting_info" / "training_inpainting_info.csv"
    inpaint_test_path = task_data_root / "inpainting_info" / "testing_inpainting_info.csv"

    prior_train_path = task_data_root / "objects_from_context_prediction" / "training_data.csv"
    prior_val_path = task_data_root / "objects_from_context_prediction" / "validation_data.csv"
    prior_test_path = task_data_root / "objects_from_context_prediction" / "testing_data.csv"

    context_train = read_csv(context_train_path)
    context_test = read_csv(context_test_path)
    inpaint_train = read_csv(inpaint_train_path)
    inpaint_test = read_csv(inpaint_test_path)
    prior_train = read_csv(prior_train_path)
    prior_val = read_csv(prior_val_path)
    prior_test = read_csv(prior_test_path)

    train_df = build_split(context_train, inpaint_train, prior_train, task_data_root, "train")
    val_df = build_split(context_train, inpaint_train, prior_val, task_data_root, "val")
    test_df = build_split(context_test, inpaint_test, prior_test, task_data_root, "test")

    save_split(train_df, output_dir, "train")
    save_split(val_df, output_dir, "val")
    save_split(test_df, output_dir, "test")

    print("\nDone.")


if __name__ == "__main__":
    main()
