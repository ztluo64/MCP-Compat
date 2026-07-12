import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

from tri_view_compat.tools import eval_threshold_calibrated as ev
from tri_view_compat.datasets.coinco_triview_dataset_v2 import COinCOTriViewDatasetV2


def load_state_dict_flexible(model, ckpt):
    state = None
    for key in ["model", "model_state_dict", "state_dict"]:
        if key in ckpt:
            state = ckpt[key]
            break

    if state is None:
        raise KeyError(f"Cannot find model weights. Checkpoint keys: {list(ckpt.keys())}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("first missing:", missing[:5])
    if unexpected:
        print("first unexpected:", unexpected[:5])
    return model


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def extract_logits(output):
    """
    Compatible with dict / tuple / tensor outputs.
    Need final binary logits [B,2] and prior logits [B,80].
    """
    logits = None
    prior_logits = None

    if isinstance(output, dict):
        for k in ["logits", "cls_logits", "context_logits", "binary_logits"]:
            if k in output:
                logits = output[k]
                break

        for k in ["prior_logits", "prior", "object_prior_logits", "expected_object_logits"]:
            if k in output:
                prior_logits = output[k]
                break

    elif isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                if item.ndim == 2 and item.shape[-1] == 2 and logits is None:
                    logits = item
                elif item.ndim == 2 and item.shape[-1] >= 70 and prior_logits is None:
                    prior_logits = item

    elif torch.is_tensor(output):
        logits = output

    if logits is None:
        raise RuntimeError(
            "Cannot find binary logits from model output. "
            f"Output type={type(output)}"
        )

    if prior_logits is None:
        raise RuntimeError(
            "Cannot find prior_logits from model output. "
            "Please inspect model output keys."
        )

    return logits, prior_logits


def forward_with_prior(model, batch, tokenizer, device):
    batch = move_batch_to_device(batch, device)

    prompts = batch.get("text_prompt", None)
    if prompts is None:
        repl = batch.get("replacement_object", None)
        if repl is not None:
            prompts = [f"a photo of a {x}" for x in repl]

    if prompts is None:
        raise RuntimeError("Cannot find text_prompt or replacement_object in batch.")

    if isinstance(prompts, str):
        prompts = [prompts]

    text_inputs = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Your prior_v2 model signature:
    # forward(full_image=None, object_crop=None, masked_context=None,
    #         input_ids=None, attention_mask=None, geometry=None,
    #         replacement_label=None)
    output = model(
        full_image=batch["full_image"],
        object_crop=batch["object_crop"],
        masked_context=batch["masked_context"],
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        geometry=batch.get("geometry", None),
        replacement_label=batch.get("replacement_label", None),
    )

    return extract_logits(output)

def get_bbox_from_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 10)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def crop_with_padding(img, bbox, pad_ratio=0.10):
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    pad = int(max(bw, bh) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    return img.crop((x1, y1, x2 + 1, y2 + 1))


def make_masked_context(img, mask_path, fill=(127, 127, 127)):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(img.size, Image.NEAREST)
    arr = np.array(mask)

    img_arr = np.array(img).copy()
    img_arr[arr > 10] = np.array(fill, dtype=np.uint8)

    return Image.fromarray(img_arr)


def draw_bbox(img, bbox):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # red bbox
        for t in range(4):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))
    return img


def build_label_maps(train_csv, val_csv, test_csv):
    dfs = []
    for p in [train_csv, val_csv, test_csv]:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    prior_to_name = {}
    if "prior_label" in all_df.columns and "class_name" in all_df.columns:
        for lab, sub in all_df.groupby("prior_label"):
            name = sub["class_name"].value_counts().idxmax()
            prior_to_name[int(lab)] = str(name)

    name_to_prior = {}
    if "class_name" in all_df.columns and "prior_label" in all_df.columns:
        for _, row in all_df[["class_name", "prior_label"]].drop_duplicates().iterrows():
            name_to_prior[str(row["class_name"])] = int(row["prior_label"])

    # fallback for replacement labels if replacement_label exists
    if "replacement_label" in all_df.columns and "replacement_object" in all_df.columns:
        for _, row in all_df[["replacement_object", "replacement_label"]].dropna().drop_duplicates().iterrows():
            name_to_prior[str(row["replacement_object"])] = int(row["replacement_label"])

    return prior_to_name, name_to_prior


def collect_predictions(args, model, tokenizer, device):
    ds = COinCOTriViewDatasetV2(args.test_csv)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_probs = []
    all_labels = []
    all_prior_probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, prior_logits = forward_with_prior(model, batch, tokenizer, device)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            prior_probs = torch.softmax(prior_logits, dim=-1)

            labels = batch["label"]
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy().tolist()

            all_probs.extend(probs.detach().cpu().float().numpy().tolist())
            all_labels.extend([int(x) for x in labels])
            all_prior_probs.extend(prior_probs.detach().cpu().float().numpy())

    return np.asarray(all_labels), np.asarray(all_probs), np.asarray(all_prior_probs)


def select_cases(df, labels, probs, prior_probs, prior_to_name, name_to_prior, threshold, max_cases):
    preds = (probs >= threshold).astype(np.int64)

    records = []

    for i, row in df.iterrows():
        label = int(labels[i])
        pred = int(preds[i])
        prob_ooc = float(probs[i])

        repl = str(row["replacement_object"])
        repl_label = None

        if "replacement_label" in row and not pd.isna(row["replacement_label"]):
            repl_label = int(row["replacement_label"])
        elif repl in name_to_prior:
            repl_label = int(name_to_prior[repl])

        p_repl = None
        repl_rank = None

        prior = prior_probs[i]
        order = np.argsort(-prior)

        if repl_label is not None and 0 <= repl_label < len(prior):
            p_repl = float(prior[repl_label])
            repl_rank = int(np.where(order == repl_label)[0][0]) + 1

        top5 = []
        for lab in order[:5]:
            lab = int(lab)
            top5.append((prior_to_name.get(lab, f"class_{lab}"), float(prior[lab])))

        correct = int(label == pred)
        confidence = abs(prob_ooc - threshold)

        # Prefer correct examples with interpretable prior.
        score = confidence
        if p_repl is not None:
            if label == 1:
                score += max(0.0, 0.25 - p_repl)
            else:
                score += p_repl

        records.append({
            "idx": i,
            "coco_index": row["coco_index"],
            "label": label,
            "pred": pred,
            "correct": correct,
            "prob_ooc": prob_ooc,
            "confidence": confidence,
            "replacement_object": repl,
            "p_replacement": p_repl,
            "replacement_rank": repl_rank,
            "top5": top5,
            "score": score,
        })

    # Select a balanced set: correct OOC, correct IC, and some failure cases.
    correct_ooc = [r for r in records if r["label"] == 1 and r["pred"] == 1]
    correct_ic = [r for r in records if r["label"] == 0 and r["pred"] == 0]
    errors = [r for r in records if r["label"] != r["pred"]]

    correct_ooc = sorted(correct_ooc, key=lambda x: -x["score"])
    correct_ic = sorted(correct_ic, key=lambda x: -x["score"])
    errors = sorted(errors, key=lambda x: -x["confidence"])

    selected = []
    k = max_cases // 3
    selected.extend(correct_ooc[:k])
    selected.extend(correct_ic[:k])
    selected.extend(errors[:max_cases - len(selected)])

    return selected


def render_case(row, rec, prior_to_name, out_path):
    img = Image.open(row["image_path"]).convert("RGB")
    bbox = get_bbox_from_mask(row["mask_path"])

    full_with_bbox = draw_bbox(img, bbox)
    crop = crop_with_padding(img, bbox) if bbox is not None else img.copy()
    masked = make_masked_context(img, row["mask_path"])

    gt = "out-of-context" if rec["label"] == 1 else "in-context"
    pred = "out-of-context" if rec["pred"] == 1 else "in-context"

    top5_text = "\n".join([
        f"{j+1}. {name}: {prob*100:.1f}%"
        for j, (name, prob) in enumerate(rec["top5"])
    ])

    p_repl = rec["p_replacement"]
    if p_repl is None:
        p_repl_text = "N/A"
    else:
        p_repl_text = f"{p_repl*100:.2f}%"

    repl_rank = rec["replacement_rank"]
    repl_rank_text = "N/A" if repl_rank is None else str(repl_rank)

    title_color = "green" if rec["correct"] else "red"

    fig = plt.figure(figsize=(15, 7))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(full_with_bbox)
    ax1.set_title("Full image + target box")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(crop)
    ax2.set_title("Object crop")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(masked)
    ax3.set_title("Masked context")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.axis("off")
    ax4.text(
        0.0, 1.0,
        f"Replacement object:\n{rec['replacement_object']}\n\n"
        f"GT: {gt}\n"
        f"Prediction: {pred}\n"
        f"P(out-of-context): {rec['prob_ooc']*100:.1f}%\n"
        f"Correct: {bool(rec['correct'])}",
        va="top",
        fontsize=13,
        color=title_color,
    )

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")
    ax5.text(
        0.0, 1.0,
        "Top-5 expected objects\nfrom masked context:\n\n" + top5_text,
        va="top",
        fontsize=13,
    )

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.0, 1.0,
        f"Prior compatibility:\n\n"
        f"p(replacement | context): {p_repl_text}\n"
        f"replacement rank: {repl_rank_text}\n\n"
        f"Interpretation:\n"
        f"Low p(replacement | context)\n"
        f"usually indicates a context\n"
        f"mismatch for the inserted object.",
        va="top",
        fontsize=13,
    )

    fig.suptitle(
        f"Case idx={rec['idx']} | coco_index={rec['coco_index']}",
        fontsize=15,
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256")
    parser.add_argument("--model_type", default="prior_v2")
    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--val_csv", default="tri_view_compat/outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--out_dir", default="tri_view_compat/outputs/case_vis/prior_v2_original")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("args:", vars(args))

    ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = ev.build_model(args.model_type, ckpt_args, device)
    model = load_state_dict_flexible(model, ckpt)
    model.to(device)
    model.eval()

    tokenizer = CLIPTokenizer.from_pretrained(ckpt_args["clip_name"])

    prior_to_name, name_to_prior = build_label_maps(args.train_csv, args.val_csv, args.test_csv)

    labels, probs, prior_probs = collect_predictions(args, model, tokenizer, device)

    if args.threshold is None:
        metric_path = os.path.join(args.ckpt_dir, "threshold_metrics.json")
        if os.path.exists(metric_path):
            with open(metric_path, "r") as f:
                m = json.load(f)
            threshold = float(m["best_threshold_from_val"])
        else:
            threshold = 0.37
    else:
        threshold = float(args.threshold)

    print("using threshold:", threshold)

    df = pd.read_csv(args.test_csv)
    cases = select_cases(df, labels, probs, prior_probs, prior_to_name, name_to_prior, threshold, args.max_cases)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for j, rec in enumerate(cases):
        row = df.iloc[rec["idx"]]
        out_path = os.path.join(args.out_dir, f"case_{j:02d}_idx{rec['idx']}_label{rec['label']}_pred{rec['pred']}.png")
        render_case(row, rec, prior_to_name, out_path)

        rows.append({
            "case_id": j,
            "idx": rec["idx"],
            "coco_index": rec["coco_index"],
            "label": rec["label"],
            "pred": rec["pred"],
            "correct": rec["correct"],
            "prob_ooc": rec["prob_ooc"],
            "replacement_object": rec["replacement_object"],
            "p_replacement": rec["p_replacement"],
            "replacement_rank": rec["replacement_rank"],
            "top5": "; ".join([f"{n}:{p:.4f}" for n, p in rec["top5"]]),
            "figure_path": out_path,
        })

    summary_csv = os.path.join(args.out_dir, "case_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print("saved figures to:", args.out_dir)
    print("saved summary to:", summary_csv)


if __name__ == "__main__":
    main()
