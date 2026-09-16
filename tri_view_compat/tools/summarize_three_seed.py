import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "MCP-Base": "mcp_base",
    "MCP-Compat": "mcp_compat",
}
SPLITS = ["original", "balanced"]
METRICS = [
    "acc",
    "balanced_acc",
    "macro_f1",
    "f1_label0",
    "f1_label1",
    "auc",
]


def load_result(path: Path):
    with path.open("r") as f:
        obj = json.load(f)
    return obj["test_calibrated"], obj["best_threshold_from_val"]


def fmt_pct(mean, std):
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="tri_view_compat/outputs/three_seed_stability",
        help="Root containing mcp_base/seed*/ and mcp_compat/seed*/ directories.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 777, 2025])
    parser.add_argument(
        "--output_dir",
        default="tri_view_compat/outputs/three_seed_stability/summary",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method_name, method_dir in METHODS.items():
        for seed in args.seeds:
            run_dir = root / method_dir / f"seed{seed}"
            for split in SPLITS:
                path = run_dir / f"threshold_metrics_{split}.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                metrics, threshold = load_result(path)
                row = {
                    "method": method_name,
                    "seed": seed,
                    "split": split,
                    "threshold": float(threshold),
                }
                for metric in METRICS:
                    row[metric] = metrics.get(metric)
                rows.append(row)

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    summary_rows = []
    for split in SPLITS:
        for method_name in METHODS:
            sub = per_seed[(per_seed.method == method_name) & (per_seed.split == split)]
            row = {"split": split, "method": method_name, "n_seeds": len(sub)}
            row["threshold_mean"] = sub.threshold.mean()
            row["threshold_std"] = sub.threshold.std(ddof=1)
            for metric in METRICS:
                vals = sub[metric].astype(float).to_numpy()
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1))
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "mean_std_metrics.csv", index=False)

    delta_rows = []
    for split in SPLITS:
        for seed in args.seeds:
            base = per_seed[
                (per_seed.method == "MCP-Base")
                & (per_seed.seed == seed)
                & (per_seed.split == split)
            ].iloc[0]
            compat = per_seed[
                (per_seed.method == "MCP-Compat")
                & (per_seed.seed == seed)
                & (per_seed.split == split)
            ].iloc[0]
            row = {"split": split, "seed": seed}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(compat[metric] - base[metric])
            delta_rows.append(row)

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(out_dir / "paired_seed_deltas.csv", index=False)

    delta_summary_rows = []
    for split in SPLITS:
        sub = deltas[deltas.split == split]
        row = {"split": split, "n_seeds": len(sub)}
        for metric in METRICS:
            vals = sub[f"delta_{metric}"].to_numpy(dtype=float)
            row[f"delta_{metric}_mean"] = float(np.mean(vals))
            row[f"delta_{metric}_std"] = float(np.std(vals, ddof=1))
            row[f"delta_{metric}_positive_seeds"] = int(np.sum(vals > 0))
        delta_summary_rows.append(row)

    delta_summary = pd.DataFrame(delta_summary_rows)
    delta_summary.to_csv(out_dir / "paired_delta_summary.csv", index=False)

    payload = {
        "seeds": args.seeds,
        "mean_std": summary.to_dict(orient="records"),
        "paired_delta_summary": delta_summary.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Three-seed mean ± std (test calibrated) ===")
    for split in SPLITS:
        print(f"\n[{split}]")
        for method_name in METHODS:
            row = summary[(summary.split == split) & (summary.method == method_name)].iloc[0]
            print(f"{method_name}")
            for metric in ["acc", "balanced_acc", "macro_f1", "auc"]:
                print(
                    f"  {metric:12s}: "
                    + fmt_pct(row[f"{metric}_mean"], row[f"{metric}_std"])
                )

    print("\n=== Paired MCP-Compat - MCP-Base deltas ===")
    for split in SPLITS:
        print(f"\n[{split}]")
        sub = deltas[deltas.split == split]
        for _, row in sub.iterrows():
            print(
                f"seed {int(row.seed):4d}: "
                f"ΔAcc={100*row.delta_acc:+.2f}, "
                f"ΔBAcc={100*row.delta_balanced_acc:+.2f}, "
                f"ΔM-F1={100*row.delta_macro_f1:+.2f}, "
                f"ΔAUC={100*row.delta_auc:+.2f}"
            )
        drow = delta_summary[delta_summary.split == split].iloc[0]
        print(
            "mean ΔM-F1: "
            f"{100*drow.delta_macro_f1_mean:+.2f} ± "
            f"{100*drow.delta_macro_f1_std:.2f}; "
            f"positive seeds: {int(drow.delta_macro_f1_positive_seeds)}/{len(args.seeds)}"
        )

    print(f"\nSaved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
