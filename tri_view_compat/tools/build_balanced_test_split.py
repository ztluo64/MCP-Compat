from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_balanced_test(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if "label" not in df.columns:
        raise KeyError("Input CSV must contain a 'label' column.")

    in_context = df[df["label"] == 0]
    out_context = df[df["label"] == 1]

    if len(in_context) == 0 or len(out_context) == 0:
        raise ValueError("Both label 0 and label 1 samples are required.")

    n = min(len(in_context), len(out_context))
    out_sampled = out_context.sample(n=n, random_state=seed)

    balanced = pd.concat([in_context, out_sampled], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced test subset from the replacement-aware test CSV."
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("tri_view_compat/outputs/splits/test.csv"),
        help="Input test CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv"),
        help="Output balanced test CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for sampling out-of-context samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.test_csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {args.test_csv}")

    df = pd.read_csv(args.test_csv)
    balanced = build_balanced_test(df, args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(args.output_csv, index=False)

    print(f"saved: {args.output_csv}")
    print(f"shape: {balanced.shape}")
    print("label counts:")
    print(balanced["label"].value_counts().sort_index())
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
