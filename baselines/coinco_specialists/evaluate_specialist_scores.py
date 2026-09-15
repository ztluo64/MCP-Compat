import argparse
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def load_scores(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            rows.append({
                "coco_index": int(x["coco_index"]),
                "label": int(x["label"]),
                "score_ooc": float(x["score_ooc"]),
                "margin": float(x["margin"]),
            })

    df = pd.DataFrame(rows)

    if df["coco_index"].duplicated().any():
        raise RuntimeError(
            f"Duplicate coco_index in {path}"
        )

    return df


def metrics(labels, scores, threshold):

    preds = (scores >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "acc": float(
            accuracy_score(labels, preds)
        ),
        "balanced_acc": float(
            balanced_accuracy_score(
                labels,
                preds,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                preds,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_label0": float(
            f1_score(
                labels,
                preds,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_label1": float(
            f1_score(
                labels,
                preds,
                pos_label=1,
                zero_division=0,
            )
        ),
        "auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),
    }


def search_best_threshold(labels, scores):

    # exactly the same grid as MCP
    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    best = None
    eps = 1e-12

    for t in thresholds:

        m = metrics(
            labels,
            scores,
            float(t),
        )

        if best is None:
            best = m
            continue

        if (
            m["macro_f1"]
            > best["macro_f1"] + eps
        ):
            best = m

        elif (
            abs(
                m["macro_f1"]
                - best["macro_f1"]
            )
            <= eps
        ):
            if (
                m["balanced_acc"]
                > best["balanced_acc"] + eps
            ):
                best = m

            elif (
                abs(
                    m["balanced_acc"]
                    - best["balanced_acc"]
                )
                <= eps
                and
                m["acc"]
                > best["acc"] + eps
            ):
                best = m

    return best


def print_metrics(title, m):

    print("\n" + title)

    for k in [
        "threshold",
        "acc",
        "balanced_acc",
        "macro_f1",
        "f1_label0",
        "f1_label1",
        "auc",
    ]:
        if k == "threshold":
            print(
                f"{k}: {m[k]:.2f}"
            )
        else:
            print(
                f"{k}: {m[k]*100:.2f}"
            )


def main(args):

    val = load_scores(args.val)

    test = load_scores(args.test)

    print("Val rows:", len(val))
    print("Test rows:", len(test))

    print(
        "Val label counts:\n",
        val["label"].value_counts().sort_index(),
    )

    print(
        "Test label counts:\n",
        test["label"].value_counts().sort_index(),
    )

    val_best = search_best_threshold(
        val["label"].to_numpy(),
        val["score_ooc"].to_numpy(),
    )

    threshold = val_best["threshold"]

    test_metrics = metrics(
        test["label"].to_numpy(),
        test["score_ooc"].to_numpy(),
        threshold,
    )

    print_metrics(
        "[Val best threshold]",
        val_best,
    )

    print_metrics(
        "[Original test]",
        test_metrics,
    )

    # balanced subset = exact old paper split
    balanced_ids = pd.read_csv(
        args.balanced_csv
    )[["coco_index", "label"]]

    balanced_ids["coco_index"] = (
        balanced_ids["coco_index"]
        .astype(int)
    )

    balanced = balanced_ids.merge(
        test[
            [
                "coco_index",
                "score_ooc",
            ]
        ],
        on="coco_index",
        how="left",
        validate="one_to_one",
    )

    if balanced["score_ooc"].isna().any():
        raise RuntimeError(
            "Missing score for balanced samples"
        )

    balanced_metrics = metrics(
        balanced["label"].to_numpy(),
        balanced["score_ooc"].to_numpy(),
        threshold,
    )

    print_metrics(
        "[Balanced test: same fixed threshold]",
        balanced_metrics,
    )

    result = {
        "val_best": val_best,
        "original_test": test_metrics,
        "balanced_test": balanced_metrics,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nsaved:", args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--val",
        required=True,
    )

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--balanced_csv",
        required=True,
        help="CSV defining the balanced test subset.",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    main(args)
