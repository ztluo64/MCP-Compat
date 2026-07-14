import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer

from tri_view_compat.datasets.coinco_triview_dataset_v2 import COinCOTriViewDatasetV2
from tri_view_compat.tools.eval_threshold_calibrated import build_model


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_threshold(path):
    obj = load_json(path)
    if "test_calibrated" in obj and "threshold" in obj["test_calibrated"]:
        return float(obj["test_calibrated"]["threshold"])
    if "best_threshold_from_val" in obj:
        return float(obj["best_threshold_from_val"])
    if "threshold" in obj:
        return float(obj["threshold"])
    raise ValueError(f"Cannot find threshold in {path}")


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "model_state"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    raise ValueError("Unsupported checkpoint format.")


def extract_ckpt_args(ckpt):
    if isinstance(ckpt, dict) and "args" in ckpt:
        args = ckpt["args"]
        if isinstance(args, dict):
            return args
        if hasattr(args, "__dict__"):
            return vars(args)
    return {}


def load_model(ckpt_dir, model_type, device):
    ckpt_dir = Path(ckpt_dir)
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = extract_ckpt_args(ckpt)

    model = build_model(model_type, ckpt_args, device)
    state = extract_state_dict(ckpt)

    # Remove possible DDP prefix.
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    model.eval()

    print(f"[INFO] Loaded {model_type}: {ckpt_path}")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    return model, ckpt_args


def build_dataset(csv_path, clip_name):
    try:
        return COinCOTriViewDatasetV2(csv_file=csv_path, clip_name=clip_name)
    except TypeError:
        pass

    try:
        return COinCOTriViewDatasetV2(csv_path=csv_path, clip_name=clip_name)
    except TypeError:
        pass

    try:
        return COinCOTriViewDatasetV2(csv_path)
    except TypeError:
        pass

    return COinCOTriViewDatasetV2(csv_path)


def to_device_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def tokenize_text(batch, tokenizer, device):
    texts = batch["text_prompt"]
    if isinstance(texts, str):
        texts = [texts]
    else:
        texts = list(texts)

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def extract_logits(output):
    """
    Returns:
      cls_logits: [B, 2]
      prior_logits: [B, 80] or None
    """
    if torch.is_tensor(output):
        return output, None

    if isinstance(output, (list, tuple)):
        cls_logits = None
        prior_logits = None
        for x in output:
            if torch.is_tensor(x) and x.ndim == 2 and x.shape[-1] == 2 and cls_logits is None:
                cls_logits = x
            if torch.is_tensor(x) and x.ndim == 2 and x.shape[-1] == 80 and prior_logits is None:
                prior_logits = x
        if cls_logits is None:
            raise RuntimeError("Cannot find classification logits from tuple/list output.")
        return cls_logits, prior_logits

    if isinstance(output, dict):
        cls_logits = None
        prior_logits = None

        for key in ["logits", "cls_logits", "class_logits", "binary_logits", "pred_logits"]:
            if key in output and torch.is_tensor(output[key]):
                cls_logits = output[key]
                break

        for key in ["prior_logits", "object_logits", "expected_logits", "prior_pred_logits"]:
            if key in output and torch.is_tensor(output[key]):
                prior_logits = output[key]
                break

        if cls_logits is None:
            for v in output.values():
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == 2:
                    cls_logits = v
                    break

        if prior_logits is None:
            for v in output.values():
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == 80:
                    prior_logits = v
                    break

        if cls_logits is None:
            raise RuntimeError(f"Cannot find classification logits. Output keys: {list(output.keys())}")

        return cls_logits, prior_logits

    raise RuntimeError(f"Unsupported output type: {type(output)}")


@torch.inference_mode()
def forward_baseline(model, tokenizer, batch, device):
    """
    CLIPTriViewBaseline uses forward(batch).
    When use_text=True, it requires batch["text_inputs"].
    """
    device_batch = to_device_batch(batch, device)
    text_inputs = tokenize_text(batch, tokenizer, device)

    device_batch["text_inputs"] = text_inputs
    device_batch["input_ids"] = text_inputs["input_ids"]
    device_batch["attention_mask"] = text_inputs["attention_mask"]

    # CLIPTriViewBaseline.forward(batch, text_inputs=None)
    # requires text_inputs as the second argument, not as a key inside batch.
    output = model(device_batch, text_inputs)
    cls_logits, _ = extract_logits(output)
    prob_ooc = torch.softmax(cls_logits, dim=-1)[:, 1]

    return prob_ooc.detach().cpu().numpy()


@torch.inference_mode()
def forward_ours(model, tokenizer, batch, device):
    """
    CLIPTriViewTextPriorV2 uses keyword inputs:
    full_image, object_crop, masked_context, input_ids, attention_mask,
    geometry, replacement_label.
    """
    device_batch = to_device_batch(batch, device)
    text_inputs = tokenize_text(batch, tokenizer, device)

    kwargs = {
        "full_image": device_batch["full_image"],
        "object_crop": device_batch["object_crop"],
        "masked_context": device_batch["masked_context"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
    }

    if "geometry" in device_batch:
        kwargs["geometry"] = device_batch["geometry"]

    if "replacement_label" in device_batch:
        kwargs["replacement_label"] = device_batch["replacement_label"]

    try:
        output = model(**kwargs)
    except TypeError:
        # Fallback for any batch-style variant.
        device_batch["text_inputs"] = text_inputs
        device_batch["input_ids"] = text_inputs["input_ids"]
        device_batch["attention_mask"] = text_inputs["attention_mask"]
        output = model(device_batch)

    cls_logits, prior_logits = extract_logits(output)

    prob_ooc = torch.softmax(cls_logits, dim=-1)[:, 1].detach().cpu().numpy()

    prior_prob = None
    if prior_logits is not None:
        prior_prob = torch.softmax(prior_logits, dim=-1).detach().cpu().numpy()

    return prob_ooc, prior_prob


def build_prior_name_map(train_csv):
    df = pd.read_csv(train_csv)
    if "prior_label" not in df.columns or "class_name" not in df.columns:
        return {}

    mapping = {}
    for _, r in df[["prior_label", "class_name"]].drop_duplicates().iterrows():
        try:
            mapping[int(r["prior_label"])] = str(r["class_name"])
        except Exception:
            pass

    return mapping


def normalized_entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-(prob * np.log(prob)).sum() / np.log(len(prob)))


def topk_expected(prob, name_map, k=5):
    idxs = np.argsort(-prob)[:k]
    items = []
    for rank, idx in enumerate(idxs, start=1):
        items.append({
            "rank": int(rank),
            "index": int(idx),
            "name": name_map.get(int(idx), f"class_{idx}"),
            "prob": float(prob[idx]),
        })
    return items


def topk_to_str(items):
    return "; ".join([f"{x['name']}:{x['prob'] * 100:.2f}%" for x in items])


def item_from_batch(batch, key, i):
    v = batch[key]
    if torch.is_tensor(v):
        return v[i].item()
    return v[i]


def bool_str(x):
    return "True" if bool(x) else "False"


def build_record(
    row_index,
    batch_i,
    batch,
    csv_row,
    base_prob,
    ours_prob,
    base_th,
    ours_th,
    prior_prob_i,
    name_map,
    topk,
):
    label = int(item_from_batch(batch, "label", batch_i))
    base_pred = int(base_prob >= base_th)
    ours_pred = int(ours_prob >= ours_th)

    replacement_label = int(item_from_batch(batch, "replacement_label", batch_i)) if "replacement_label" in batch else int(csv_row.get("replacement_label", -1))
    prior_label = int(item_from_batch(batch, "prior_label", batch_i)) if "prior_label" in batch else int(csv_row.get("prior_label", -1))

    record = {
        "row_index": int(row_index),
        "coco_index": int(csv_row.get("coco_index", item_from_batch(batch, "coco_index", batch_i))),
        "label": label,
        "label_name": "out-of-context" if label == 1 else "in-context",
        "class_name": str(csv_row.get("class_name", item_from_batch(batch, "class_name", batch_i))),
        "replacement_object": str(csv_row.get("replacement_object", item_from_batch(batch, "replacement_object", batch_i))),
        "text_prompt": str(csv_row.get("text_prompt", item_from_batch(batch, "text_prompt", batch_i))),
        "prior_label": prior_label,
        "replacement_label": replacement_label,
        "image_path": str(csv_row.get("image_path", "")),
        "mask_path": str(csv_row.get("mask_path", "")),
        "baseline_prob_ooc": float(base_prob),
        "baseline_threshold": float(base_th),
        "baseline_pred": base_pred,
        "baseline_pred_name": "out-of-context" if base_pred == 1 else "in-context",
        "baseline_correct": bool(base_pred == label),
        "ours_prob_ooc": float(ours_prob),
        "ours_threshold": float(ours_th),
        "ours_pred": ours_pred,
        "ours_pred_name": "out-of-context" if ours_pred == 1 else "in-context",
        "ours_correct": bool(ours_pred == label),
    }

    if prior_prob_i is not None and replacement_label >= 0:
        top_items = topk_expected(prior_prob_i, name_map, k=topk)
        p_r = float(prior_prob_i[replacement_label])
        p_max = float(np.max(prior_prob_i))
        margin = float(p_max - p_r)
        ratio = float(p_r / max(p_max, 1e-12))

        record.update({
            "topk_expected_objects": json.dumps(top_items, ensure_ascii=False),
            "topk_expected_str": topk_to_str(top_items),
            "top1_expected_name": top_items[0]["name"],
            "top1_expected_prob": top_items[0]["prob"],
            "p_replacement": p_r,
            "log_p_replacement": float(np.log(max(p_r, 1e-12))),
            "p_max": p_max,
            "margin": margin,
            "normalized_entropy": normalized_entropy(prior_prob_i),
            "ratio": ratio,
        })
    else:
        record.update({
            "topk_expected_objects": "",
            "topk_expected_str": "",
            "top1_expected_name": "",
            "top1_expected_prob": np.nan,
            "p_replacement": np.nan,
            "log_p_replacement": np.nan,
            "p_max": np.nan,
            "margin": np.nan,
            "normalized_entropy": np.nan,
            "ratio": np.nan,
        })

    return record


def select_cases(df, num_ooc=2, num_ic=2):
    df = df.copy()
    df["hard_gain"] = (~df["baseline_correct"]) & (df["ours_correct"])

    selected = []

    def pick(label_value, n):
        used = set([x["row_index"] for x in selected])

        sub = df[(df["label"] == label_value) & (~df["row_index"].isin(used))].copy()

        # Priority group:
        # 0 = baseline wrong, ours correct
        # 1 = ours correct
        # 2 = fallback
        sub["priority"] = 2
        sub.loc[sub["ours_correct"], "priority"] = 1
        sub.loc[sub["hard_gain"], "priority"] = 0

        if label_value == 1:
            # Out-of-context: lower p_replacement and larger margin are more interpretable.
            sub = sub.sort_values(
                by=["priority", "p_replacement", "margin", "ours_prob_ooc"],
                ascending=[True, True, False, False],
            )
        else:
            # In-context: higher p_replacement and lower OOC probability are more interpretable.
            sub = sub.sort_values(
                by=["priority", "p_replacement", "ours_prob_ooc"],
                ascending=[True, False, True],
            )

        return sub.head(n).to_dict("records")

    selected.extend(pick(1, num_ooc))
    selected.extend(pick(0, num_ic))

    out = pd.DataFrame(selected)
    if len(out) == 0:
        return out

    # Final order: OOC first, then IC.
    out["order"] = out["label"].map({1: 0, 0: 1})
    out = out.sort_values(["order", "row_index"]).drop(columns=["order"])
    return out


def write_preview(selected_df, path):
    lines = []
    lines.append("===== Selected qualitative cases =====")
    lines.append("")

    for idx, r in selected_df.reset_index(drop=True).iterrows():
        lines.append(f"[Case {idx + 1}]")
        lines.append(f"row_index:          {r['row_index']}")
        lines.append(f"coco_index:         {r['coco_index']}")
        lines.append(f"GT:                 {r['label_name']}")
        lines.append(f"Expected/original:  {r['class_name']}")
        lines.append(f"Replacement:        {r['replacement_object']}")
        lines.append(f"Text prompt:        {r['text_prompt']}")
        lines.append(
            f"Baseline:           {r['baseline_pred_name']} | "
            f"prob_ooc={r['baseline_prob_ooc']:.6f} | "
            f"correct={bool_str(r['baseline_correct'])}"
        )
        lines.append(
            f"Ours:               {r['ours_pred_name']} | "
            f"prob_ooc={r['ours_prob_ooc']:.6f} | "
            f"correct={bool_str(r['ours_correct'])}"
        )
        lines.append(f"p_replacement:      {r['p_replacement']:.8f}")
        lines.append(f"p_max:              {r['p_max']:.8f}")
        lines.append(f"margin:             {r['margin']:.8f}")
        lines.append(f"ratio:              {r['ratio']:.8f}")
        lines.append(f"entropy:            {r['normalized_entropy']:.8f}")
        lines.append(f"Top-K expected:     {r['topk_expected_str']}")
        lines.append(f"image_path:         {r['image_path']}")
        lines.append(f"mask_path:          {r['mask_path']}")
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--clip_name", default="/home/ubuntu/ztl/models/clip-vit-base-patch32")

    parser.add_argument("--baseline_ckpt_dir", required=True)
    parser.add_argument("--baseline_threshold_json", required=True)

    parser.add_argument("--ours_ckpt_dir", required=True)
    parser.add_argument("--ours_threshold_json", required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_ooc", type=int, default=2)
    parser.add_argument("--num_ic", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output_dir", default="tri_view_compat/outputs/qualitative_selection")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    base_th = load_threshold(args.baseline_threshold_json)
    ours_th = load_threshold(args.ours_threshold_json)
    print(f"[INFO] baseline threshold = {base_th}")
    print(f"[INFO] ours threshold     = {ours_th}")

    baseline_model, _ = load_model(args.baseline_ckpt_dir, "baseline", device)
    ours_model, _ = load_model(args.ours_ckpt_dir, "prior_v2", device)

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)

    dataset = build_dataset(args.csv, args.clip_name)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    csv_df = pd.read_csv(args.csv)
    name_map = build_prior_name_map(args.train_csv)

    rows = []
    global_start = 0

    for batch in tqdm(loader, desc="Evaluating test set"):
        base_probs = forward_baseline(baseline_model, tokenizer, batch, device)
        ours_probs, prior_probs = forward_ours(ours_model, tokenizer, batch, device)

        bs = len(base_probs)
        for i in range(bs):
            row_index = global_start + i
            csv_row = csv_df.iloc[row_index]

            prior_prob_i = None
            if prior_probs is not None:
                prior_prob_i = prior_probs[i]

            rec = build_record(
                row_index=row_index,
                batch_i=i,
                batch=batch,
                csv_row=csv_row,
                base_prob=base_probs[i],
                ours_prob=ours_probs[i],
                base_th=base_th,
                ours_th=ours_th,
                prior_prob_i=prior_prob_i,
                name_map=name_map,
                topk=args.topk,
            )
            rows.append(rec)

        global_start += bs

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.DataFrame(rows)
    all_csv = out_dir / "all_cases_scored.csv"
    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    selected_df = select_cases(all_df, num_ooc=args.num_ooc, num_ic=args.num_ic)
    selected_csv = out_dir / "selected_cases.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    preview_txt = out_dir / "selected_cases_preview.txt"
    write_preview(selected_df, preview_txt)

    print()
    print(f"[Saved] {all_csv}")
    print(f"[Saved] {selected_csv}")
    print(f"[Saved] {preview_txt}")
    print()

    show_cols = [
        "row_index",
        "coco_index",
        "label_name",
        "class_name",
        "replacement_object",
        "baseline_pred_name",
        "baseline_correct",
        "ours_pred_name",
        "ours_correct",
        "p_replacement",
        "p_max",
        "margin",
        "topk_expected_str",
    ]

    print("===== Selected cases =====")
    print(selected_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
