import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_feature(item, mode: str):
    latent1 = item["latent1"].float().flatten()
    latent2 = item["latent2"].float().flatten()
    visual = torch.cat([latent1, latent2], dim=0)

    obj = item["objects_embeddings"].float().flatten()
    rep = item["replacement_embedding"].float().flatten()
    semantic = torch.cat([obj, rep], dim=0)

    if mode == "visual":
        return visual
    if mode == "semantic":
        return semantic
    if mode == "vissemantic":
        return torch.cat([visual, semantic], dim=0)

    raise ValueError(f"Unknown mode: {mode}")


def load_split(csv_path, cache_dir, mode):
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)

    df = pd.read_csv(csv_path)
    xs, ys, missing = [], [], []

    for _, row in df.iterrows():
        coco_index = int(row["coco_index"])
        cache_path = cache_dir / f"{coco_index}.pt"

        if not cache_path.exists():
            missing.append(coco_index)
            continue

        item = safe_torch_load(cache_path)
        x = build_feature(item, mode)
        y = int(row["label"])

        xs.append(x)
        ys.append(y)

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} cache files under {cache_dir}. "
            f"First missing indices: {missing[:10]}"
        )

    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)

    return x, y


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu()
        probs.append(p)
        labels.append(y.cpu())

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return labels, probs


def metrics_from_probs(labels, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)

    out = {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "f1_label0": float(f1_score(labels, preds, pos_label=0)),
        "f1_label1": float(f1_score(labels, preds, pos_label=1)),
    }

    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except Exception:
        out["auc"] = None

    return out


def search_best_threshold(labels, probs):
    best_t, best_m = 0.5, None

    for t in np.arange(0.01, 1.00, 0.01):
        m = metrics_from_probs(labels, probs, t)
        if best_m is None or m["macro_f1"] > best_m["macro_f1"]:
            best_t, best_m = float(t), m

    return best_t, best_m


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["visual", "semantic", "vissemantic"])
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--balanced_test_csv", default="tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv")

    parser.add_argument("--cache_train_val", default="task_data/cache/cached_embeddings_train_val_cp")
    parser.add_argument("--cache_test", default="task_data/cache/cached_embeddings_testing_cp")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== Load features =====")
    print("mode:", args.mode)

    x_train, y_train = load_split(args.train_csv, args.cache_train_val, args.mode)
    x_val, y_val = load_split(args.val_csv, args.cache_train_val, args.mode)
    x_test, y_test = load_split(args.test_csv, args.cache_test, args.mode)
    x_bal, y_bal = load_split(args.balanced_test_csv, args.cache_test, args.mode)

    print("train:", tuple(x_train.shape), y_train.bincount().tolist())
    print("val:", tuple(x_val.shape), y_val.bincount().tolist())
    print("test:", tuple(x_test.shape), y_test.bincount().tolist())
    print("balanced:", tuple(x_bal.shape), y_bal.bincount().tolist())

    # Standardize with train statistics only.
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    x_bal = (x_bal - mean) / std

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    bal_loader = DataLoader(
        TensorDataset(x_bal, y_bal),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    input_dim = x_train.shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    counts = y_train.bincount(minlength=2).float()
    total = counts.sum()
    class_weight = total / (2.0 * counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_macro = -1.0
    best_record = None

    print("===== Train =====")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_seen += x.size(0)

        val_labels, val_probs = collect_probs(model, val_loader, device)
        best_t, val_metrics = search_best_threshold(val_labels, val_probs)

        avg_loss = total_loss / max(1, total_seen)

        print(
            f"Epoch {epoch:02d} "
            f"loss={avg_loss:.4f} "
            f"val_th={best_t:.2f} "
            f"val_acc={val_metrics['acc']*100:.2f} "
            f"val_bal={val_metrics['balanced_acc']*100:.2f} "
            f"val_macro={val_metrics['macro_f1']*100:.2f} "
            f"val_auc={val_metrics['auc']*100:.2f}"
        )

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_record = {
                "epoch": epoch,
                "best_threshold_from_val": best_t,
                "val_best": val_metrics,
            }

            torch.save(
                {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                    "input_dim": input_dim,
                    "best_record": best_record,
                },
                out_dir / "best.pt",
            )

    print("===== Final evaluation =====")
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    threshold = float(ckpt["best_record"]["best_threshold_from_val"])

    test_labels, test_probs = collect_probs(model, test_loader, device)
    bal_labels, bal_probs = collect_probs(model, bal_loader, device)

    test_metrics = metrics_from_probs(test_labels, test_probs, threshold)
    bal_metrics = metrics_from_probs(bal_labels, bal_probs, threshold)

    out_original = {
        "model": f"COinCO-{args.mode}",
        "mode": args.mode,
        "best_epoch": int(ckpt["best_record"]["epoch"]),
        "best_threshold_from_val": threshold,
        "val_best": ckpt["best_record"]["val_best"],
        "test_calibrated": test_metrics,
    }

    out_balanced = {
        "model": f"COinCO-{args.mode}",
        "mode": args.mode,
        "best_epoch": int(ckpt["best_record"]["epoch"]),
        "best_threshold_from_val": threshold,
        "val_best": ckpt["best_record"]["val_best"],
        "test_calibrated": bal_metrics,
    }

    with open(out_dir / "threshold_metrics_original_test.json", "w") as f:
        json.dump(out_original, f, indent=2)

    with open(out_dir / "threshold_metrics_balanced_test.json", "w") as f:
        json.dump(out_balanced, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "original": out_original,
                "balanced": out_balanced,
            },
            f,
            indent=2,
        )

    print("\nOriginal test:")
    for k, v in test_metrics.items():
        if k == "threshold":
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v * 100:.2f}")

    print("\nBalanced test:")
    for k, v in bal_metrics.items():
        if k == "threshold":
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v * 100:.2f}")

    print("\nsaved to:", out_dir)


if __name__ == "__main__":
    main()
