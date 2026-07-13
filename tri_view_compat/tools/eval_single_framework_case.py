import argparse
import inspect
import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_threshold(ckpt_dir, fallback=0.5):
    candidates = [
        Path(ckpt_dir) / "threshold_metrics_original_test.json",
        Path(ckpt_dir) / "threshold_metrics_balanced_test.json",
        Path(ckpt_dir) / "threshold_metrics.json",
    ]
    for p in candidates:
        if p.exists():
            obj = load_json(p)
            if "test_calibrated" in obj and "threshold" in obj["test_calibrated"]:
                return float(obj["test_calibrated"]["threshold"]), str(p)
            if "threshold" in obj:
                return float(obj["threshold"]), str(p)
    return float(fallback), "fallback"


def import_dataset_class():
    import tri_view_compat.datasets.coinco_triview_dataset_v2 as ds_mod

    candidates = []
    for name, obj in vars(ds_mod).items():
        if inspect.isclass(obj) and issubclass(obj, Dataset) and obj is not Dataset:
            score = 0
            lname = name.lower()
            if "v2" in lname:
                score += 3
            if "triview" in lname:
                score += 2
            if "coinco" in lname:
                score += 2
            candidates.append((score, name, obj))

    if not candidates:
        raise RuntimeError("No Dataset subclass found in tri_view_compat.datasets.coinco_triview_dataset_v2")

    candidates.sort(reverse=True, key=lambda x: x[0])
    print(f"[INFO] Using dataset class: {candidates[0][1]}")
    return candidates[0][2]


def import_model_class():
    import tri_view_compat.models.clip_triview_text_prior_v2 as model_mod

    candidates = []
    for name, obj in vars(model_mod).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
            score = 0
            lname = name.lower()
            if "v2" in lname:
                score += 4
            if "prior" in lname:
                score += 3
            if "triview" in lname:
                score += 2
            if "clip" in lname:
                score += 1
            candidates.append((score, name, obj))

    if not candidates:
        raise RuntimeError("No nn.Module subclass found in tri_view_compat.models.clip_triview_text_prior_v2")

    candidates.sort(reverse=True, key=lambda x: x[0])
    print(f"[INFO] Using model class: {candidates[0][1]}")
    return candidates[0][2]


def instantiate_dataset(cls, csv_path, clip_name, root):
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    for k, p in sig.parameters.items():
        if k == "self":
            continue

        lk = k.lower()

        if lk in {"csv", "csv_path", "csv_file", "data_csv", "split_csv", "anno_path", "annotation_path"}:
            kwargs[k] = str(csv_path)
        elif lk in {"root", "root_dir", "data_root", "project_root"}:
            kwargs[k] = str(root)
        elif lk in {"clip_name", "clip_model", "clip_model_name", "processor_name", "pretrained_model_name_or_path"}:
            kwargs[k] = str(clip_name)
        elif lk in {"image_size", "img_size", "size"}:
            kwargs[k] = 224
        elif lk in {"max_length", "text_max_length"}:
            kwargs[k] = 77
        elif lk in {"train", "is_train", "training"}:
            kwargs[k] = False
        elif lk in {"augment", "use_augment", "use_aug"}:
            kwargs[k] = False
        elif p.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Cannot instantiate dataset. Required argument not recognized: {k}. "
                f"Dataset signature: {sig}"
            )

    return cls(**kwargs)


def instantiate_model(cls, clip_name, device):
    sig = inspect.signature(cls.__init__)
    kwargs = {}

    for k, p in sig.parameters.items():
        if k == "self":
            continue

        lk = k.lower()

        if lk in {"clip_name", "clip_model", "clip_model_name", "pretrained_model_name_or_path"}:
            kwargs[k] = str(clip_name)
        elif lk in {"num_classes", "num_labels", "n_classes"}:
            kwargs[k] = 2
        elif lk in {"prior_num_classes", "num_prior_classes", "num_object_classes", "num_categories", "object_num_classes"}:
            kwargs[k] = 80
        elif lk in {"use_geo", "with_geo"}:
            kwargs[k] = False
        elif lk in {"freeze_clip", "freeze_backbone"}:
            kwargs[k] = True
        elif lk in {"hidden_dim", "dim", "proj_dim"} and p.default is inspect.Parameter.empty:
            kwargs[k] = 512
        elif p.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Cannot instantiate model. Required argument not recognized: {k}. "
                f"Model signature: {sig}"
            )

    model = cls(**kwargs)
    return model.to(device)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict", "model_state"]:
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    # strip module. prefix if needed
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[WARN] First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("[WARN] First unexpected keys:", unexpected[:10])


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def call_model(model, batch):
    errors = []

    # Try model(**batch), filtered by forward signature if needed.
    sig = inspect.signature(model.forward)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if has_kwargs:
        try:
            return model(**batch)
        except Exception as e:
            errors.append(f"model(**batch) failed: {repr(e)}")
    else:
        allowed = {k: v for k, v in batch.items() if k in params}
        try:
            return model(**allowed)
        except Exception as e:
            errors.append(f"model(**allowed) failed: {repr(e)}")

    # Try model(batch).
    try:
        return model(batch)
    except Exception as e:
        errors.append(f"model(batch) failed: {repr(e)}")

    print("[ERROR] Batch keys:", list(batch.keys()))
    print("[ERROR] Forward signature:", sig)
    raise RuntimeError("\n".join(errors))


def find_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, (list, tuple)):
        for x in outputs:
            if torch.is_tensor(x) and x.ndim >= 2 and x.shape[-1] in {1, 2}:
                return x

    if isinstance(outputs, dict):
        preferred = [
            "logits", "cls_logits", "class_logits", "binary_logits",
            "out_logits", "y_logits", "pred_logits"
        ]
        for k in preferred:
            if k in outputs and torch.is_tensor(outputs[k]):
                return outputs[k]

        for k, v in outputs.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[-1] in {1, 2}:
                print(f"[INFO] Using logits key by shape: {k}")
                return v

    raise RuntimeError(f"Cannot find classification logits in model output. Output type={type(outputs)}")


def find_prior_logits(outputs):
    preferred = [
        "prior_logits", "object_logits", "q", "prior_q",
        "expected_logits", "prior_pred_logits"
    ]

    if isinstance(outputs, dict):
        for k in preferred:
            if k in outputs and torch.is_tensor(outputs[k]) and outputs[k].ndim >= 2 and outputs[k].shape[-1] >= 10:
                return k, outputs[k]

        for k, v in outputs.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[-1] == 80:
                return k, v

    if isinstance(outputs, (list, tuple)):
        for i, v in enumerate(outputs):
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[-1] == 80:
                return f"tuple[{i}]", v

    return None, None


def tensor_to_scalar(x):
    if torch.is_tensor(x):
        return x.detach().cpu().view(-1)[0].item()
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def get_class_names_from_train(train_csv):
    df = pd.read_csv(train_csv)
    if "prior_label" not in df.columns or "class_name" not in df.columns:
        return None

    mapping = {}
    for _, r in df.iterrows():
        try:
            mapping[int(r["prior_label"])] = str(r["class_name"])
        except Exception:
            pass

    if not mapping:
        return None

    return [mapping.get(i, f"class_{i}") for i in range(max(mapping.keys()) + 1)]


def normalized_entropy(p):
    p = p.clamp_min(1e-12)
    h = -(p * p.log()).sum().item()
    return h / math.log(p.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", default="tri_view_compat/outputs/framework_case_search/ooc_for_fig1/ooc_framework_candidates.csv")
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--ckpt_dir", default="tri_view_compat/outputs/checkpoints/triview_text_prior_v2_lam02_bs256")
    parser.add_argument("--ckpt_name", default="best.pt")
    parser.add_argument("--clip_name", default="/home/ubuntu/ztl/models/clip-vit-base-patch32")
    parser.add_argument("--train_csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates_csv = Path(args.candidates_csv)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / args.ckpt_name

    if not candidates_csv.exists():
        raise FileNotFoundError(candidates_csv)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cand = pd.read_csv(candidates_csv)
    row_df = cand[cand["rank"] == args.rank].copy()
    if len(row_df) != 1:
        raise RuntimeError(f"Expected exactly one row with rank={args.rank}, got {len(row_df)}")

    row = row_df.iloc[0]

    # Create one-row CSV compatible with dataset.
    keep_cols = [
        "coco_index", "label", "class_name", "replacement_object",
        "prior_label", "replacement_label", "image_path", "mask_path"
    ]
    one = row_df[[c for c in keep_cols if c in row_df.columns]].copy()

    tmp_dir = Path(tempfile.mkdtemp(prefix="single_case_"))
    single_csv = tmp_dir / f"rank_{args.rank}_single.csv"
    one.to_csv(single_csv, index=False)

    threshold, th_source = (
        (float(args.threshold), "manual")
        if args.threshold is not None
        else get_threshold(ckpt_dir, fallback=0.5)
    )

    print("===== Selected case =====")
    print(f"rank:               {args.rank}")
    print(f"row_index:          {row.get('row_index', '')}")
    print(f"coco_index:         {row.get('coco_index', '')}")
    print(f"label:              {row.get('label', '')} (1=out-of-context)")
    print(f"expected/original:  {row.get('class_name', '')}")
    print(f"replacement:        {row.get('replacement_object', '')}")
    print(f"prior_label:        {row.get('prior_label', '')}")
    print(f"replacement_label:  {row.get('replacement_label', '')}")
    print(f"image_path:         {row.get('image_path', '')}")
    print(f"mask_path:          {row.get('mask_path', '')}")
    print(f"threshold:          {threshold:.4f} ({th_source})")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    DatasetCls = import_dataset_class()
    ModelCls = import_model_class()

    dataset = instantiate_dataset(DatasetCls, single_csv, args.clip_name, root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = move_to_device(batch, device)

    # The dataset stores replacement text as raw text_prompt.
    # The model forward expects CLIP text inputs: input_ids and attention_mask.
    if ("input_ids" not in batch or "attention_mask" not in batch) and "text_prompt" in batch:
        tokenizer = AutoTokenizer.from_pretrained(args.clip_name)
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
        batch["input_ids"] = enc["input_ids"].to(device)
        batch["attention_mask"] = enc["attention_mask"].to(device)
        print(f"[INFO] Tokenized text_prompt: {texts}")

    model = instantiate_model(ModelCls, args.clip_name, device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        outputs = call_model(model, batch)
        logits = find_logits(outputs)

        if logits.ndim == 2 and logits.shape[-1] == 2:
            prob_ooc = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            prob_ooc = torch.sigmoid(logits.view(-1)[0]).item()

        pred = int(prob_ooc >= threshold)
        gt = int(row["label"])
        correct = pred == gt

    print("===== Classification result =====")
    print(f"GT label:            {gt} ({'out-of-context' if gt == 1 else 'in-context'})")
    print(f"P(out-of-context):   {prob_ooc:.6f}")
    print(f"Threshold:           {threshold:.4f}")
    print(f"Prediction:          {pred} ({'out-of-context' if pred == 1 else 'in-context'})")
    print(f"Correct:             {correct}")

    result = {
        "rank": int(args.rank),
        "row_index": int(row.get("row_index", -1)),
        "coco_index": int(row.get("coco_index", -1)),
        "gt_label": gt,
        "prob_out_of_context": prob_ooc,
        "threshold": threshold,
        "prediction": pred,
        "correct": correct,
        "class_name": str(row.get("class_name", "")),
        "replacement_object": str(row.get("replacement_object", "")),
        "prior_label": int(row.get("prior_label", -1)),
        "replacement_label": int(row.get("replacement_label", -1)),
    }

    key, prior_logits = find_prior_logits(outputs)
    if prior_logits is not None:
        p = torch.softmax(prior_logits[0].detach().float().cpu(), dim=-1)
        repl_idx = int(row.get("replacement_label", -1))

        class_names = get_class_names_from_train(args.train_csv)

        topk = torch.topk(p, k=min(10, p.numel()))
        top_items = []
        for rank_i, (idx, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
            name = class_names[idx] if class_names is not None and idx < len(class_names) else f"class_{idx}"
            top_items.append({
                "rank": rank_i,
                "index": idx,
                "name": name,
                "prob": float(val),
            })

        print()
        print("===== Prior prediction result =====")
        print(f"prior logits key:    {key}")
        print("Top expected objects:")
        for item in top_items:
            print(f"  {item['rank']:02d}. {item['name']} (idx={item['index']}): {item['prob'] * 100:.2f}%")

        if 0 <= repl_idx < p.numel():
            p_r = p[repl_idx].item()
            pmax = p.max().item()
            margin = pmax - p_r
            entropy = normalized_entropy(p)
            ratio = p_r / (pmax + 1e-12)

            repl_name = class_names[repl_idx] if class_names is not None and repl_idx < len(class_names) else f"class_{repl_idx}"

            print()
            print("Explicit compatibility cues:")
            print(f"  replacement class:      {repl_name} (idx={repl_idx})")
            print(f"  p_replacement:          {p_r:.8f}")
            print(f"  log_p_replacement:      {math.log(p_r + 1e-12):.6f}")
            print(f"  p_max:                  {pmax:.8f}")
            print(f"  margin = p_max - p_r:   {margin:.8f}")
            print(f"  normalized entropy:     {entropy:.6f}")
            print(f"  ratio = p_r / p_max:    {ratio:.8f}")

            result["prior_logits_key"] = key
            result["top_expected_objects"] = top_items
            result["compatibility_cues"] = {
                "replacement_class_name": repl_name,
                "replacement_index": repl_idx,
                "p_replacement": p_r,
                "log_p_replacement": math.log(p_r + 1e-12),
                "p_max": pmax,
                "margin": margin,
                "normalized_entropy": entropy,
                "ratio": ratio,
            }
    else:
        print()
        print("[WARN] Could not find prior logits in model output.")
        if isinstance(outputs, dict):
            print("[INFO] Output keys:", list(outputs.keys()))

    if args.out_json is None:
        out_json = Path(args.candidates_csv).parent / f"rank_{args.rank:03d}_model_eval.json"
    else:
        out_json = Path(args.out_json)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Saved result JSON: {out_json}")


if __name__ == "__main__":
    main()
