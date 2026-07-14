from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(COCO_CLASSES)}


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def add_replacement_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "replacement_object" not in df.columns:
        raise KeyError("CSV must contain a 'replacement_object' column.")

    names = df["replacement_object"].map(normalize_name)
    missing = sorted(set(names) - set(CLASS_TO_IDX))
    if missing:
        raise ValueError(f"Unknown replacement object names: {missing}")

    df["replacement_label"] = names.map(CLASS_TO_IDX).astype(int)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add replacement_label to tri-view split CSV files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("tri_view_compat/outputs/splits"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tri_view_compat/outputs/splits"),
        help="Directory to save CSV files with replacement_label.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        in_path = args.input_dir / f"{split}.csv"
        out_path = args.output_dir / f"{split}.csv"

        if not in_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {in_path}")

        df = pd.read_csv(in_path)
        df = add_replacement_label(df)
        df.to_csv(out_path, index=False)

        print(f"{split}: saved {out_path}, shape={df.shape}, replacement_label unique={df['replacement_label'].nunique()}")

    print("Done.")


if __name__ == "__main__":
    main()
