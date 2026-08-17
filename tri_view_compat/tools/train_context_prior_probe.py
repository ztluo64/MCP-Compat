from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import CLIPModel

from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset


VIEW_TO_KEY = {
    "full": "full_image",
    "crop": "object_crop",
    "masked": "masked_context",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_image_features(clip: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    feat = clip.get_image_features(pixel_values=pixel_values)

    # Compatibility with possible transformers output changes.
    if hasattr(feat, "pooler_output"):
        feat = feat.pooler_output
        if feat.shape[-1] != clip.config.projection_dim:
            feat = clip.visual_projection(feat)

    feat = F.normalize(feat, dim=-1)
    return feat


@torch.no_grad()
def extract_features(
    csv_path: Path,
    clip_name: str,
    view: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    cache_path: Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        return obj["features"], obj["labels"]

    if view not in VIEW_TO_KEY:
        raise ValueError(f"Unknown view: {view}. Choices: {sorted(VIEW_TO_KEY)}")

    ds = COinCOTriViewDataset(csv_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    clip = CLIPModel.from_pretrained(clip_name).to(device)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    key = VIEW_TO_KEY[view]
    all_feats = []
    all_labels = []

    for batch in tqdm(loader, desc=f"Extract {view} features: {csv_path.name}"):
        x = batch[key].to(device, non_blocking=True)
        y = batch["prior_label"]

        feat = encode_image_features(clip, x)

        all_feats.append(feat.detach().cpu())
        all_labels.append(y.detach().cpu())

    features = torch.cat(all_feats, dim=0).float()
    labels = torch.cat(all_labels, dim=0).long()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": features,
                "labels": labels,
                "csv_path": str(csv_path),
                "view": view,
                "clip_name": clip_name,
            },
            cache_path,
        )

    return features, labels


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    pred = logits.topk(k, dim=1).indices
    hit = pred.eq(labels.view(-1, 1)).any(dim=1).float()
    return float(hit.mean().item())


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    probe.eval()
    ds = TensorDataset(features, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    logits_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device)
        logits = probe(x)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y)

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    return {
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, 5),
    }


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)

    feat_dim = train_features.shape[1]
    probe = nn.Linear(feat_dim, num_classes).to(device)

    weights = class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_ds = TensorDataset(train_features, train_labels)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    best_state = None
    best_val_top1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        probe.train()

        total_loss = 0.0
        total_n = 0

        for x, y in tqdm(train_loader, desc=f"Train probe epoch {epoch}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = probe(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total_n += y.size(0)

        train_metrics = evaluate_probe(probe, train_features, train_labels, batch_size, device)
        val_metrics = evaluate_probe(probe, val_features, val_labels, batch_size, device)

        row = {
            "epoch": epoch,
            "loss": total_loss / max(total_n, 1),
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(row)

        print(
            f"epoch={epoch:02d} "
            f"loss={row['loss']:.4f} "
            f"train_top1={100*row['train_top1']:.2f} "
            f"train_top5={100*row['train_top5']:.2f} "
            f"val_top1={100*row['val_top1']:.2f} "
            f"val_top5={100*row['val_top5']:.2f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)

    final_train = evaluate_probe(probe, train_features, train_labels, batch_size, device)
    final_val = evaluate_probe(probe, val_features, val_labels, batch_size, device)
    final_test = evaluate_probe(probe, test_features, test_labels, batch_size, device)

    return {
        "best_val_top1": best_val_top1,
        "train": final_train,
        "val": final_val,
        "test": final_test,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic linear probe: predict original object category from frozen CLIP image features."
    )
    parser.add_argument("--train_csv", type=Path, default=Path("tri_view_compat/outputs/splits/train.csv"))
    parser.add_argument("--val_csv", type=Path, default=Path("tri_view_compat/outputs/splits/val.csv"))
    parser.add_argument("--test_csv", type=Path, default=Path("tri_view_compat/outputs/splits/test.csv"))

    parser.add_argument("--clip_name", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("tri_view_compat/outputs/context_prior_probe"))

    parser.add_argument("--views", nargs="+", default=["full", "crop", "masked"], choices=["full", "crop", "masked"])
    parser.add_argument("--num_classes", type=int, default=80)

    parser.add_argument("--extract_batch_size", type=int, default=256)
    parser.add_argument("--probe_batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=777)

    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = args.output_dir / "feature_cache"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    random_top1 = 1.0 / args.num_classes
    random_top5 = min(5, args.num_classes) / args.num_classes

    results = {
        "random": {
            "top1": random_top1,
            "top5": random_top5,
        },
        "views": {},
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
    }

    for view in args.views:
        print("\n" + "=" * 80)
        print(f"View: {view}")
        print("=" * 80)

        def cache(split: str) -> Path | None:
            if args.no_cache:
                return None
            return feature_cache_dir / f"{split}_{view}.pt"

        train_features, train_labels = extract_features(
            args.train_csv,
            args.clip_name,
            view,
            args.extract_batch_size,
            args.num_workers,
            device,
            cache("train"),
        )
        val_features, val_labels = extract_features(
            args.val_csv,
            args.clip_name,
            view,
            args.extract_batch_size,
            args.num_workers,
            device,
            cache("val"),
        )
        test_features, test_labels = extract_features(
            args.test_csv,
            args.clip_name,
            view,
            args.extract_batch_size,
            args.num_workers,
            device,
            cache("test"),
        )

        print("feature shapes:", train_features.shape, val_features.shape, test_features.shape)

        view_result = train_linear_probe(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            test_features=test_features,
            test_labels=test_labels,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.probe_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=device,
        )

        results["views"][view] = view_result

        print(f"\n[Final test: {view}]")
        print(f"Top-1: {100 * view_result['test']['top1']:.2f}%")
        print(f"Top-5: {100 * view_result['test']['top5']:.2f}%")

    out_json = args.output_dir / "context_prior_probe_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for view, res in results["views"].items():
        rows.append({
            "input": view,
            "test_top1": 100 * res["test"]["top1"],
            "test_top5": 100 * res["test"]["top5"],
            "val_top1": 100 * res["val"]["top1"],
            "val_top5": 100 * res["val"]["top5"],
        })
    rows.append({
        "input": "random",
        "test_top1": 100 * random_top1,
        "test_top5": 100 * random_top5,
        "val_top1": 100 * random_top1,
        "val_top5": 100 * random_top5,
    })

    out_csv = args.output_dir / "context_prior_probe_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nsaved json:", out_json)
    print("saved csv:", out_csv)


if __name__ == "__main__":
    main()
