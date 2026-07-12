import argparse
import inspect
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset
from tri_view_compat.datasets.coinco_triview_dataset_v2 import COinCOTriViewDatasetV2
from tri_view_compat.models.clip_triview_baseline import CLIPTriViewBaseline
from tri_view_compat.models.clip_triview_text_prior import CLIPTriViewTextPrior
from tri_view_compat.models.clip_triview_text_prior_v2 import CLIPTriViewTextPriorV2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filtered_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def build_model(model_type, ckpt_args, device):
    clip_name = ckpt_args["clip_name"]

    common_kwargs = dict(
        clip_name=clip_name,
        use_full=not ckpt_args.get("no_full", False),
        use_crop=not ckpt_args.get("no_crop", False),
        use_masked=not ckpt_args.get("no_masked", False),
        use_text=not ckpt_args.get("no_text", False),
        use_geo=not ckpt_args.get("no_geo", False),
        num_classes=2,
        dropout=0.1,
        freeze_clip=True,
    )

    if model_type == "baseline":
        cls = CLIPTriViewBaseline
    elif model_type == "prior_v1":
        cls = CLIPTriViewTextPrior
        common_kwargs["num_prior_classes"] = 80
    elif model_type == "prior_v2":
        cls = CLIPTriViewTextPriorV2
        common_kwargs["num_prior_classes"] = 80
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = cls(**filtered_kwargs(cls, common_kwargs))
    return model.to(device)


def metrics_from_probs(labels, probs, threshold):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
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

    out["report"] = classification_report(labels, preds, digits=4)
    return out


def search_best_threshold(labels, probs, metric="macro_f1"):
    best = None

    # Dense enough for stable macro-F1 selection.
    thresholds = np.linspace(0.01, 0.99, 99)

    for t in thresholds:
        m = metrics_from_probs(labels, probs, t)

        if best is None:
            best = m
            continue

        # Primary metric: macro-F1. Tie-breaker: balanced acc, then acc.
        if m[metric] > best[metric]:
            best = m
        elif abs(m[metric] - best[metric]) < 1e-12:
            if m["balanced_acc"] > best["balanced_acc"]:
                best = m
            elif abs(m["balanced_acc"] - best["balanced_acc"]) < 1e-12:
                if m["acc"] > best["acc"]:
                    best = m

    return best


@torch.no_grad()
def collect_probs(model, model_type, loader, tokenizer, device, desc):
    model.eval()

    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc=desc):
        full_image = batch["full_image"].to(device, non_blocking=True)
        object_crop = batch["object_crop"].to(device, non_blocking=True)
        masked_context = batch["masked_context"].to(device, non_blocking=True)
        geometry = batch["geometry"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        text_inputs = tokenizer(
            list(batch["text_prompt"]),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device, non_blocking=True)
        attention_mask = text_inputs["attention_mask"].to(device, non_blocking=True)

        kwargs = dict(
            full_image=full_image,
            object_crop=object_crop,
            masked_context=masked_context,
            input_ids=input_ids,
            attention_mask=attention_mask,
            geometry=geometry,
        )

        if model_type == "prior_v2":
            kwargs["replacement_label"] = batch["replacement_label"].to(device, non_blocking=True)

        try:
            out = model(**kwargs)
        except TypeError as e:
            # Compatibility fallback for the original CLIPTriViewBaseline:
            # forward(self, batch, text_inputs=None)
            if model_type == "baseline":
                batch_for_model = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch_for_model[k] = v.to(device, non_blocking=True)
                    else:
                        batch_for_model[k] = v

                text_inputs_for_model = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                out = model(batch_for_model, text_inputs_for_model)
            else:
                raise e

        if isinstance(out, dict):
            logits = out["logits"]
        else:
            logits = out

        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    return all_labels, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--model_type", required=True, choices=["baseline", "prior_v1", "prior_v2"])
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--metric", default="macro_f1")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    print("checkpoint epoch:", ckpt.get("epoch"))
    print("checkpoint args:", ckpt_args)

    tokenizer = CLIPTokenizer.from_pretrained(ckpt_args["clip_name"])

    dataset_cls = COinCOTriViewDatasetV2 if args.model_type == "prior_v2" else COinCOTriViewDataset

    val_ds = dataset_cls(args.val_csv)
    test_ds = dataset_cls(args.test_csv)

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

    model = build_model(args.model_type, ckpt_args, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    val_labels, val_probs = collect_probs(
        model, args.model_type, val_loader, tokenizer, device, desc="Collect val probs"
    )
    test_labels, test_probs = collect_probs(
        model, args.model_type, test_loader, tokenizer, device, desc="Collect test probs"
    )

    val_default = metrics_from_probs(val_labels, val_probs, 0.5)
    test_default = metrics_from_probs(test_labels, test_probs, 0.5)

    val_best = search_best_threshold(val_labels, val_probs, metric=args.metric)
    best_t = val_best["threshold"]

    test_calibrated = metrics_from_probs(test_labels, test_probs, best_t)

    result = {
        "ckpt_dir": args.ckpt_dir,
        "model_type": args.model_type,
        "selected_metric": args.metric,
        "best_threshold_from_val": best_t,
        "val_default": val_default,
        "val_best": val_best,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
    }

    out_path = os.path.join(args.ckpt_dir, "threshold_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n[Val default threshold=0.5]")
    for k, v in val_default.items():
        if k != "report":
            print(f"{k}: {v}")

    print("\n[Val best threshold]")
    for k, v in val_best.items():
        if k != "report":
            print(f"{k}: {v}")

    print("\n[Test default threshold=0.5]")
    for k, v in test_default.items():
        if k != "report":
            print(f"{k}: {v}")

    print("\n[Test calibrated by val threshold]")
    for k, v in test_calibrated.items():
        if k != "report":
            print(f"{k}: {v}")
    print(test_calibrated["report"])

    print("\nsaved to:", out_path)


if __name__ == "__main__":
    main()
