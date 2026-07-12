import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from transformers import CLIPTokenizer

from tri_view_compat.tools import eval_threshold_calibrated as ev
from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset

try:
    from tri_view_compat.datasets.coinco_triview_dataset_v2 import COinCOTriViewDatasetV2
except Exception:
    COinCOTriViewDatasetV2 = COinCOTriViewDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(model_type, csv_path):
    cls = COinCOTriViewDatasetV2 if model_type in ["prior_v1", "prior_v2"] else COinCOTriViewDataset

    # Robust constructor fallback.
    for kwargs in [
        {"csv_path": csv_path},
        {"data_csv": csv_path},
        {"csv_file": csv_path},
    ]:
        try:
            return cls(**kwargs)
        except TypeError:
            pass

    return cls(csv_path)


def load_state_dict_flexible(model, ckpt):
    state = None
    for key in ["model", "model_state_dict", "state_dict"]:
        if key in ckpt:
            state = ckpt[key]
            break

    if state is None:
        raise KeyError(f"Cannot find model state in checkpoint. Keys: {list(ckpt.keys())}")

    # Remove DataParallel / DDP prefix if needed.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  first missing:", missing[:5])
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:5])

    return model


def load_model_and_probs(ckpt_dir, model_type, val_csv, test_csv, batch_size, num_workers, device, metric):
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    print("\n== Loading model ==")
    print("ckpt_dir:", ckpt_dir)
    print("model_type:", model_type)
    print("ckpt_path:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    print("checkpoint epoch:", ckpt.get("epoch", "NA"))
    print("clip_name:", ckpt_args.get("clip_name"))

    model = ev.build_model(model_type, ckpt_args, device)
    model = load_state_dict_flexible(model, ckpt)
    model.to(device)
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained(ckpt_args["clip_name"])

    val_ds = make_dataset(model_type, val_csv)
    test_ds = make_dataset(model_type, test_csv)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_labels, val_probs = ev.collect_probs(
        model, model_type, val_loader, tokenizer, device, desc=f"Val probs {os.path.basename(ckpt_dir)}"
    )
    test_labels, test_probs = ev.collect_probs(
        model, model_type, test_loader, tokenizer, device, desc=f"Test probs {os.path.basename(ckpt_dir)}"
    )

    val_best = ev.search_best_threshold(val_labels, val_probs, metric=metric)
    threshold = float(val_best["threshold"])
    print("selected threshold:", threshold)
    print("val_best", {k: v for k, v in val_best.items() if k != "report"})

    return {
        "ckpt_dir": ckpt_dir,
        "model_type": model_type,
        "threshold": threshold,
        "labels": np.asarray(test_labels, dtype=np.int64),
        "probs": np.asarray(test_probs, dtype=np.float64),
    }


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(np.int64)

    out = {
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_label0": f1_score(labels, preds, pos_label=0, zero_division=0),
        "f1_label1": f1_score(labels, preds, pos_label=1, zero_division=0),
    }

    try:
        out["auc"] = roc_auc_score(labels, probs)
    except Exception:
        out["auc"] = np.nan

    return out


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def bootstrap_pair(a, b, n_boot, seed):
    labels_a = a["labels"]
    labels_b = b["labels"]

    if len(labels_a) != len(labels_b):
        raise ValueError(f"Different test lengths: {len(labels_a)} vs {len(labels_b)}")
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Labels are not aligned between model A and B. Use the same test CSV/order.")

    labels = labels_a
    n = len(labels)
    rng = np.random.default_rng(seed)

    metrics = ["acc", "balanced_acc", "macro_f1", "f1_label0", "f1_label1", "auc"]

    base_a = compute_metrics(labels, a["probs"], a["threshold"])
    base_b = compute_metrics(labels, b["probs"], b["threshold"])
    base_diff = {k: base_b[k] - base_a[k] for k in metrics}

    dist_a = {k: [] for k in metrics}
    dist_b = {k: [] for k in metrics}
    dist_diff = {k: [] for k in metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]

        # Very rare, but skip AUC if bootstrap sample contains one class only.
        ma = compute_metrics(y, a["probs"][idx], a["threshold"])
        mb = compute_metrics(y, b["probs"][idx], b["threshold"])

        for k in metrics:
            dist_a[k].append(ma[k])
            dist_b[k].append(mb[k])
            dist_diff[k].append(mb[k] - ma[k])

    out = {
        "n": int(n),
        "n_boot": int(n_boot),
        "base_a": {k: float(v) for k, v in base_a.items()},
        "base_b": {k: float(v) for k, v in base_b.items()},
        "base_diff_b_minus_a": {k: float(v) for k, v in base_diff.items()},
        "bootstrap_a": {k: summarize(v) for k, v in dist_a.items()},
        "bootstrap_b": {k: summarize(v) for k, v in dist_b.items()},
        "bootstrap_diff_b_minus_a": {k: summarize(v) for k, v in dist_diff.items()},
    }

    return out


def pct(x):
    return 100.0 * x


def print_summary(name_a, name_b, result):
    print("\n================ Bootstrap Summary ================")
    print("Model A:", name_a)
    print("Model B:", name_b)
    print("B - A means improvement of Model B over Model A.")
    print("N:", result["n"], "n_boot:", result["n_boot"])

    print("\n[Point estimate on full test]")
    for k, v in result["base_a"].items():
        print(f"A {k:>12}: {pct(v):.2f}")
    for k, v in result["base_b"].items():
        print(f"B {k:>12}: {pct(v):.2f}")

    print("\n[Bootstrap 95% CI: Model A]")
    for k, s in result["bootstrap_a"].items():
        print(f"{k:>12}: {pct(s['mean']):.2f} ± {pct(s['std']):.2f} | [{pct(s['ci95_low']):.2f}, {pct(s['ci95_high']):.2f}]")

    print("\n[Bootstrap 95% CI: Model B]")
    for k, s in result["bootstrap_b"].items():
        print(f"{k:>12}: {pct(s['mean']):.2f} ± {pct(s['std']):.2f} | [{pct(s['ci95_low']):.2f}, {pct(s['ci95_high']):.2f}]")

    print("\n[Bootstrap 95% CI: B - A]")
    for k, s in result["bootstrap_diff_b_minus_a"].items():
        print(f"{k:>12}: {pct(s['mean']):+.2f} ± {pct(s['std']):.2f} | [{pct(s['ci95_low']):+.2f}, {pct(s['ci95_high']):+.2f}]")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_a_name", default="Tri-view + Text")
    parser.add_argument("--model_a_ckpt_dir", required=True)
    parser.add_argument("--model_a_type", choices=["baseline", "prior_v1", "prior_v2"], required=True)

    parser.add_argument("--model_b_name", default="Ours")
    parser.add_argument("--model_b_ckpt_dir", required=True)
    parser.add_argument("--model_b_type", choices=["baseline", "prior_v1", "prior_v2"], required=True)

    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_json", required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--metric", default="macro_f1")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    model_a = load_model_and_probs(
        ckpt_dir=args.model_a_ckpt_dir,
        model_type=args.model_a_type,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        metric=args.metric,
    )

    model_b = load_model_and_probs(
        ckpt_dir=args.model_b_ckpt_dir,
        model_type=args.model_b_type,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        metric=args.metric,
    )

    result = bootstrap_pair(model_a, model_b, n_boot=args.n_boot, seed=args.seed)

    result["model_a_name"] = args.model_a_name
    result["model_b_name"] = args.model_b_name
    result["model_a_ckpt_dir"] = args.model_a_ckpt_dir
    result["model_b_ckpt_dir"] = args.model_b_ckpt_dir
    result["val_csv"] = args.val_csv
    result["test_csv"] = args.test_csv

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print_summary(args.model_a_name, args.model_b_name, result)
    print("\nsaved to:", args.output_json)


if __name__ == "__main__":
    main()
