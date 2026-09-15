import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def load_score(path, name):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            rows.append({
                "coco_index": int(x["coco_index"]),
                "label": int(x["label"]),
                f"score_{name}": float(x["score_ooc"]),
            })

    df = pd.DataFrame(rows)

    if df["coco_index"].duplicated().any():
        raise RuntimeError(f"Duplicate coco_index in {path}")

    return df


def load_threshold(path):
    with open(path, encoding="utf-8") as f:
        x = json.load(f)

    return float(x["val_best"]["threshold"])


def logit(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 1e-8, 1 - 1e-8)

    return np.log(x / (1 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build(split, files, thresholds):
    dfs = [
        load_score(cfg[split], name)
        for name, cfg in files.items()
    ]

    out = dfs[0]

    for d in dfs[1:]:
        out = out.merge(
            d,
            on=["coco_index", "label"],
            validate="one_to_one",
        )

    z = []

    for name in ["co", "loc", "size"]:
        z.append(
            logit(out[f"score_{name}"])
            - logit(thresholds[name])
        )

    zmax = np.maximum.reduce(z)

    out["score_fusion"] = sigmoid(zmax)

    return out


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "acc": float(
            accuracy_score(y, pred)
        ),
        "balanced_acc": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_label0": float(
            f1_score(
                y,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_label1": float(
            f1_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "auc": float(
            roc_auc_score(y, score)
        ),
    }


def search_best_threshold(y, score):
    best = None
    eps = 1e-12

    for threshold in np.linspace(0.01, 0.99, 99):
        m = metrics(y, score, float(threshold))

        if best is None:
            best = m
            continue

        if m["macro_f1"] > best["macro_f1"] + eps:
            best = m

        elif abs(
            m["macro_f1"] - best["macro_f1"]
        ) <= eps:

            if (
                m["balanced_acc"]
                > best["balanced_acc"] + eps
            ):
                best = m

            elif (
                abs(
                    m["balanced_acc"]
                    - best["balanced_acc"]
                ) <= eps
                and
                m["acc"] > best["acc"] + eps
            ):
                best = m

    return best


def show(name, m):
    print("\n" + name)
    print(f"threshold: {m['threshold']:.2f}")

    for k in [
        "acc",
        "balanced_acc",
        "macro_f1",
        "f1_label0",
        "f1_label1",
        "auc",
    ]:
        print(f"{k}: {m[k] * 100:.2f}")


def main(args):
    results_dir = Path(args.results_dir)

    files = {
        "co": {
            "val": results_dir / "cooccurrence_val_scores.jsonl",
            "test": results_dir / "cooccurrence_test_scores.jsonl",
            "metric": results_dir / "cooccurrence_score_metrics.json",
        },
        "loc": {
            "val": results_dir / "location_val_scores.jsonl",
            "test": results_dir / "location_test_scores.jsonl",
            "metric": results_dir / "location_score_metrics.json",
        },
        "size": {
            "val": results_dir / "size_val_scores.jsonl",
            "test": results_dir / "size_test_scores.jsonl",
            "metric": results_dir / "size_score_metrics.json",
        },
    }

    thresholds = {
        k: load_threshold(v["metric"])
        for k, v in files.items()
    }

    print("Specialist thresholds:", thresholds)

    val = build(
        split="val",
        files=files,
        thresholds=thresholds,
    )

    test = build(
        split="test",
        files=files,
        thresholds=thresholds,
    )

    print("Val rows:", len(val))
    print("Test rows:", len(test))

    if len(val) != 19064:
        raise RuntimeError(
            f"Expected 19064 validation samples, got {len(val)}"
        )

    if len(test) != 2402:
        raise RuntimeError(
            f"Expected 2402 test samples, got {len(test)}"
        )

    best = search_best_threshold(
        val["label"].to_numpy(),
        val["score_fusion"].to_numpy(),
    )

    threshold = best["threshold"]

    original = metrics(
        test["label"].to_numpy(),
        test["score_fusion"].to_numpy(),
        threshold,
    )

    balanced_ids = pd.read_csv(
        args.balanced_csv
    )[["coco_index", "label"]]

    balanced_ids["coco_index"] = (
        balanced_ids["coco_index"].astype(int)
    )

    balanced = balanced_ids.merge(
        test[
            [
                "coco_index",
                "label",
                "score_fusion",
            ]
        ],
        on=["coco_index", "label"],
        how="left",
        validate="one_to_one",
    )

    if balanced["score_fusion"].isna().any():
        raise RuntimeError(
            "Missing fusion score for balanced samples"
        )

    balanced_metrics = metrics(
        balanced["label"].to_numpy(),
        balanced["score_fusion"].to_numpy(),
        threshold,
    )

    show("[Val best fusion threshold]", best)
    show("[Original test]", original)
    show("[Balanced test]", balanced_metrics)

    result = {
        "specialist_thresholds": thresholds,
        "val_best": best,
        "original_test": original,
        "balanced_test": balanced_metrics,
    }

    output = (
        Path(args.output)
        if args.output is not None
        else results_dir / "coinco_fusion_val_calibrated_metrics.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nsaved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_dir",
        required=True,
        help=(
            "Directory containing the three specialists' "
            "validation/test score JSONL files and metric JSON files."
        ),
    )

    parser.add_argument(
        "--balanced_csv",
        required=True,
        help="CSV defining the balanced test subset.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to "
            "<results_dir>/coinco_fusion_val_calibrated_metrics.json."
        ),
    )

    args = parser.parse_args()

    main(args)
