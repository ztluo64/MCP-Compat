import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from transformers import CLIPTokenizer

from tri_view_compat.datasets.coinco_triview_dataset import COinCOTriViewDataset
from tri_view_compat.models.clip_triview_text_prior_aux import CLIPTriViewTextPriorAux


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


def load_state_dict_strict(model, ckpt):
    state = ckpt.get("model", None)
    if state is None:
        state = ckpt.get("model_state_dict", None)
    if state is None:
        state = ckpt.get("state_dict", None)
    if state is None:
        raise KeyError(f"Cannot find model state in checkpoint keys: {list(ckpt.keys())}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)


def build_model(ckpt_args, device):
    model = CLIPTriViewTextPriorAux(
        clip_name=ckpt_args["clip_name"],
        use_full=not ckpt_args.get("no_full", False),
        use_crop=not ckpt_args.get("no_crop", False),
        use_masked=not ckpt_args.get("no_masked", False),
        use_text=not ckpt_args.get("no_text", False),
        use_geo=not ckpt_args.get("no_geo", False),
        num_classes=2,
        num_prior_classes=80,
        dropout=0.1,
        freeze_clip=True,
    ).to(device)
    return model


def collect_probs(model, tokenizer, loader, device, desc):
    model.eval()
    labels_all = []
    probs_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = move_batch_to_device(batch, device)

            text_inputs = tokenizer(
                list(batch["text_prompt"]),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            out = model(
                full_image=batch["full_image"],
                object_crop=batch["object_crop"],
                masked_context=batch["masked_context"],
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                geometry=batch.get("geometry", None),
            )

            logits = out["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            labels_all.extend(batch["label"].detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    return np.array(labels_all), np.array(probs_all)


def metrics_from_probs(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "f1_label0": float(f1_score(labels, preds, pos_label=0)),
        "f1_label1": float(f1_score(labels, preds, pos_label=1)),
        "auc": float(roc_auc_score(labels, probs)),
        "report": classification_report(labels, preds, digits=4),
    }
    return out


def search_best_threshold(labels, probs, metric="macro_f1"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    eps = 1e-12

    def get_bal_acc(m):
        if "balanced_acc" in m:
            return m["balanced_acc"]
        if "bal_acc" in m:
            return m["bal_acc"]
        return 0.0

    for t in thresholds:
        m = metrics_from_probs(labels, probs, float(t))

        if best is None:
            best = m
            continue

        # Primary criterion: target metric, usually Macro-F1
        if m[metric] > best[metric] + eps:
            best = m
            continue

        # Tie-break 1: Balanced Accuracy
        if abs(m[metric] - best[metric]) <= eps:
            if get_bal_acc(m) > get_bal_acc(best) + eps:
                best = m
                continue

            # Tie-break 2: Accuracy
            if abs(get_bal_acc(m) - get_bal_acc(best)) <= eps:
                if m.get("acc", 0.0) > best.get("acc", 0.0) + eps:
                    best = m
                    continue

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    print("checkpoint epoch:", ckpt.get("epoch", None))
    print("checkpoint args:", ckpt_args)

    tokenizer = CLIPTokenizer.from_pretrained(ckpt_args["clip_name"])

    model = build_model(ckpt_args, device)
    load_state_dict_strict(model, ckpt)
    model.eval()

    val_ds = COinCOTriViewDataset(args.val_csv)
    test_ds = COinCOTriViewDataset(args.test_csv)

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

    val_labels, val_probs = collect_probs(model, tokenizer, val_loader, device, "Collect val probs")
    test_labels, test_probs = collect_probs(model, tokenizer, test_loader, device, "Collect test probs")

    val_default = metrics_from_probs(val_labels, val_probs, 0.5)
    test_default = metrics_from_probs(test_labels, test_probs, 0.5)

    val_best = search_best_threshold(val_labels, val_probs, metric=args.metric)
    best_t = val_best["threshold"]
    test_calibrated = metrics_from_probs(test_labels, test_probs, best_t)

    result = {
        "ckpt_dir": args.ckpt_dir,
        "model_type": "prior_aux",
        "selected_metric": args.metric,
        "threshold_grid": "{0.01, 0.02, ..., 0.99}",
        "best_threshold_from_val": best_t,
        "val_default": val_default,
        "val_best": val_best,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
    }

    out_path = args.output_json
    if out_path is None:
        out_path = os.path.join(args.ckpt_dir, "threshold_metrics_prior_aux.json")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n[Val best threshold]")
    for k, v in val_best.items():
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
