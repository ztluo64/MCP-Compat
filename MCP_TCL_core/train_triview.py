import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from transformers import CLIPTokenizer

from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset
from tri_view_compat.models.clip_triview_baseline import CLIPTriViewBaseline


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_subset(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def evaluate(model, tokenizer, loader, device, criterion=None, desc="Eval"):
    model.eval()

    all_labels = []
    all_preds = []
    all_scores = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = move_batch_to_device(batch, device)

            text_inputs = None
            if model.use_text:
                text_inputs = tokenizer(
                    batch["text_prompt"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

            logits = model(batch, text_inputs=text_inputs)
            labels = batch["label"]

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                total_count += labels.size(0)

            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())

    metrics = {}
    metrics["loss"] = total_loss / max(total_count, 1) if criterion is not None else None
    metrics["acc"] = accuracy_score(all_labels, all_preds)
    metrics["balanced_acc"] = balanced_accuracy_score(all_labels, all_preds)
    metrics["macro_f1"] = f1_score(all_labels, all_preds, average="macro")
    metrics["f1_label0"] = f1_score(all_labels, all_preds, pos_label=0)
    metrics["f1_label1"] = f1_score(all_labels, all_preds, pos_label=1)

    try:
        metrics["auc"] = roc_auc_score(all_labels, all_scores)
    except Exception:
        metrics["auc"] = None

    metrics["report"] = classification_report(all_labels, all_preds, digits=4)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")

    parser.add_argument("--clip_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output_dir", default="tri_view_compat/outputs/checkpoints/clip_triview_full")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=777)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)

    parser.add_argument("--no_full", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_masked", action="store_true")
    parser.add_argument("--no_geo", action="store_true")
    parser.add_argument("--no_text", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    train_dataset = COinCOTriViewDataset(args.train_csv)
    val_dataset = COinCOTriViewDataset(args.val_csv)
    test_dataset = COinCOTriViewDataset(args.test_csv)

    train_dataset = make_subset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = make_subset(val_dataset, args.max_val_samples, args.seed + 1)
    test_dataset = make_subset(test_dataset, args.max_test_samples, args.seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_df = pd.read_csv(args.train_csv)
    counts = train_df["label"].value_counts().sort_index()
    total = counts.sum()
    weights = [total / (2.0 * counts.get(i, 1)) for i in range(2)]
    class_weight = torch.tensor(weights, dtype=torch.float32).to(device)
    print("class counts:", counts.to_dict())
    print("class weight:", weights)

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)

    model = CLIPTriViewBaseline(
        clip_name=args.clip_name,
        use_full=not args.no_full,
        use_crop=not args.no_crop,
        use_masked=not args.no_masked,
        use_geo=not args.no_geo,
        use_text=not args.no_text,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            text_inputs = None
            if model.use_text:
                text_inputs = tokenizer(
                    batch["text_prompt"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

            logits = model(batch, text_inputs=text_inputs)
            labels = batch["label"]
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)

            pbar.set_postfix(loss=total_loss / max(total_seen, 1))

        val_metrics = evaluate(model, tokenizer, val_loader, device, criterion, desc=f"Val epoch {epoch}")
        print(f"\nEpoch {epoch} val metrics:")
        for k, v in val_metrics.items():
            if k != "report":
                print(f"{k}: {v}")
        print(val_metrics["report"])

        record = {"epoch": epoch, "val": val_metrics}
        history.append(record)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            ckpt_path = os.path.join(args.output_dir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_macro_f1": best_macro_f1,
                "epoch": epoch,
            }, ckpt_path)
            print("saved best:", ckpt_path)

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    best_path = os.path.join(args.output_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("loaded best checkpoint:", best_path)

    test_metrics = evaluate(model, tokenizer, test_loader, device, criterion, desc="Test")
    print("\nFinal test metrics:")
    for k, v in test_metrics.items():
        if k != "report":
            print(f"{k}: {v}")
    print(test_metrics["report"])

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
