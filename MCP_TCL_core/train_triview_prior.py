import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import CLIPTokenizer

from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset
from tri_view_compat.models.clip_triview_text_prior import CLIPTriViewTextPrior


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_dataset(ds, max_samples):
    if max_samples is None or max_samples <= 0:
        return ds
    n = min(len(ds), max_samples)
    return Subset(ds, list(range(n)))


def compute_class_weight(csv_path):
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    weights = []
    for i in range(2):
        c = counts.get(i, 1)
        weights.append(float(total / (2.0 * c)))
    return weights, counts


def prior_topk(prior_logits, prior_labels, k):
    pred = torch.topk(prior_logits, k=k, dim=1).indices
    hit = pred.eq(prior_labels.view(-1, 1)).any(dim=1).float()
    return hit.mean().item()


def run_eval(model, loader, tokenizer, device, ce_ooc, ce_prior, prior_loss_weight, desc):
    model.eval()

    total_loss = 0.0
    total_ooc_loss = 0.0
    total_prior_loss = 0.0
    total_n = 0

    all_labels = []
    all_probs = []
    all_preds = []

    prior_top1_sum = 0.0
    prior_top3_sum = 0.0
    prior_top5_sum = 0.0
    prior_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            full_image = batch["full_image"].to(device, non_blocking=True)
            object_crop = batch["object_crop"].to(device, non_blocking=True)
            masked_context = batch["masked_context"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            prior_labels = batch["prior_label"].to(device, non_blocking=True)

            geometry = batch["geometry"].to(device, non_blocking=True)

            text_inputs = tokenizer(
                list(batch["text_prompt"]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs["input_ids"].to(device, non_blocking=True)
            attention_mask = text_inputs["attention_mask"].to(device, non_blocking=True)

            out = model(
                full_image=full_image,
                object_crop=object_crop,
                masked_context=masked_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                geometry=geometry,
            )

            logits = out["logits"]
            prior_logits = out["prior_logits"]

            ooc_loss = ce_ooc(logits, labels)
            prior_loss = ce_prior(prior_logits, prior_labels)
            loss = ooc_loss + prior_loss_weight * prior_loss

            bsz = labels.size(0)
            total_loss += loss.item() * bsz
            total_ooc_loss += ooc_loss.item() * bsz
            total_prior_loss += prior_loss.item() * bsz
            total_n += bsz

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

            prior_top1_sum += prior_topk(prior_logits, prior_labels, 1)
            prior_top3_sum += prior_topk(prior_logits, prior_labels, 3)
            prior_top5_sum += prior_topk(prior_logits, prior_labels, 5)
            prior_batches += 1

    metrics = {}
    metrics["loss"] = total_loss / max(total_n, 1)
    metrics["ooc_loss"] = total_ooc_loss / max(total_n, 1)
    metrics["prior_loss"] = total_prior_loss / max(total_n, 1)

    metrics["acc"] = float(accuracy_score(all_labels, all_preds))
    metrics["balanced_acc"] = float(balanced_accuracy_score(all_labels, all_preds))
    metrics["macro_f1"] = float(f1_score(all_labels, all_preds, average="macro"))
    metrics["f1_label0"] = float(f1_score(all_labels, all_preds, pos_label=0))
    metrics["f1_label1"] = float(f1_score(all_labels, all_preds, pos_label=1))

    try:
        metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        metrics["auc"] = None

    metrics["prior_top1"] = prior_top1_sum / max(prior_batches, 1)
    metrics["prior_top3"] = prior_top3_sum / max(prior_batches, 1)
    metrics["prior_top5"] = prior_top5_sum / max(prior_batches, 1)

    metrics["report"] = classification_report(
        all_labels,
        all_preds,
        digits=4,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")

    parser.add_argument("--clip_name", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=777)

    parser.add_argument("--prior_loss_weight", type=float, default=0.1)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)

    parser.add_argument("--no_full", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_masked", action="store_true")
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--no_geo", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = COinCOTriViewDataset(args.train_csv)
    val_ds = COinCOTriViewDataset(args.val_csv)
    test_ds = COinCOTriViewDataset(args.test_csv)

    train_ds = limit_dataset(train_ds, args.max_train_samples)
    val_ds = limit_dataset(val_ds, args.max_val_samples)
    test_ds = limit_dataset(test_ds, args.max_test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    class_weight, class_counts = compute_class_weight(args.train_csv)
    print("class counts:", class_counts)
    print("class weight:", class_weight)

    class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)

    ce_ooc = nn.CrossEntropyLoss(weight=class_weight)
    ce_prior = nn.CrossEntropyLoss()

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)

    model = CLIPTriViewTextPrior(
        clip_name=args.clip_name,
        use_full=not args.no_full,
        use_crop=not args.no_crop,
        use_masked=not args.no_masked,
        use_text=not args.no_text,
        use_geo=not args.no_geo,
        num_prior_classes=80,
        dropout=0.1,
        freeze_clip=True,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", trainable_params)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_macro_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_ooc_loss = 0.0
        total_prior_loss = 0.0
        total_n = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")

        for batch in pbar:
            full_image = batch["full_image"].to(device, non_blocking=True)
            object_crop = batch["object_crop"].to(device, non_blocking=True)
            masked_context = batch["masked_context"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            prior_labels = batch["prior_label"].to(device, non_blocking=True)

            geometry = batch["geometry"].to(device, non_blocking=True)

            text_inputs = tokenizer(
                list(batch["text_prompt"]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs["input_ids"].to(device, non_blocking=True)
            attention_mask = text_inputs["attention_mask"].to(device, non_blocking=True)

            out = model(
                full_image=full_image,
                object_crop=object_crop,
                masked_context=masked_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                geometry=geometry,
            )

            logits = out["logits"]
            prior_logits = out["prior_logits"]

            ooc_loss = ce_ooc(logits, labels)
            prior_loss = ce_prior(prior_logits, prior_labels)
            loss = ooc_loss + args.prior_loss_weight * prior_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = labels.size(0)
            total_loss += loss.item() * bsz
            total_ooc_loss += ooc_loss.item() * bsz
            total_prior_loss += prior_loss.item() * bsz
            total_n += bsz

            pbar.set_postfix({
                "loss": f"{total_loss / max(total_n, 1):.4f}",
                "ooc": f"{total_ooc_loss / max(total_n, 1):.4f}",
                "prior": f"{total_prior_loss / max(total_n, 1):.4f}",
            })

        val_metrics = run_eval(
            model,
            val_loader,
            tokenizer,
            device,
            ce_ooc,
            ce_prior,
            args.prior_loss_weight,
            desc=f"Val epoch {epoch}",
        )

        print(f"\nEpoch {epoch} val metrics:")
        for k, v in val_metrics.items():
            if k != "report":
                print(f"{k}: {v}")
        print(val_metrics["report"])

        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "train_ooc_loss": total_ooc_loss / max(total_n, 1),
            "train_prior_loss": total_prior_loss / max(total_n, 1),
            "val": val_metrics,
        }
        history.append(record)

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val_metrics,
                },
                best_path,
            )
            print("saved best:", best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print("loaded best checkpoint:", best_path)

    test_metrics = run_eval(
        model,
        test_loader,
        tokenizer,
        device,
        ce_ooc,
        ce_prior,
        args.prior_loss_weight,
        desc="Test",
    )

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
