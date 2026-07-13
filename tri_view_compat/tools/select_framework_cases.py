import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def resolve_path(p, root):
    p = Path(str(p))
    if p.is_absolute():
        return p
    return root / p


def load_font(size=22, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def parse_csv_list(s):
    if not s:
        return set()
    return {x.strip().lower() for x in s.split(",") if x.strip()}


def parse_pairs(s):
    pairs = set()
    if not s:
        return pairs
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad pair format: {item}. Expected original:replacement")
        a, b = item.split(":", 1)
        pairs.add((a.strip().lower(), b.strip().lower()))
    return pairs


def mask_to_binary(mask_img, size):
    mask = mask_img.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    arr = np.array(mask)
    binary = arr > 127

    # If mask is mostly white, it may be inverted; choose smaller foreground.
    if binary.mean() > 0.5:
        binary = ~binary

    return binary


def bbox_from_binary(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(bbox, w, h, pad_ratio=0.12):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def draw_box(img, bbox, color=(255, 0, 0), width=4):
    out = img.copy()
    d = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    for i in range(width):
        d.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return out


def make_masked_context(img, binary, fill=(235, 235, 235)):
    arr = np.array(img.convert("RGB"))
    arr[binary] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(arr)


def resize_keep(img, size=(230, 180), bg=(255, 255, 255)):
    img = img.convert("RGB")
    canvas = Image.new("RGB", size, bg)
    img.thumbnail(size, Image.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def score_candidate(row, area_ratio, bbox, image_size, prefer_original, prefer_replacement):
    w, h = image_size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Prefer medium-sized target regions around 8%-18% of image area.
    target_area = 0.12
    score_area = math.exp(-abs(math.log((area_ratio + 1e-8) / target_area)))

    # Prefer not too thin / not too elongated.
    aspect = bw / max(bh, 1)
    score_aspect = math.exp(-abs(math.log(max(aspect, 1e-8))))

    # Prefer central-ish objects but not mandatory.
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    score_center = max(0.0, 1.0 - dist * 1.5)

    # Prefer replacement and original objects that are visually intuitive.
    orig = str(row.get("class_name", "")).lower()
    repl = str(row.get("replacement_object", "")).lower()

    score_pref = 0.0
    if prefer_original and orig in prefer_original:
        score_pref += 0.20
    if prefer_replacement and repl in prefer_replacement:
        score_pref += 0.25

    # Prefer mismatch for method illustration.
    score_mismatch = 0.15 if orig != repl else 0.0

    return (
        0.45 * score_area
        + 0.20 * score_aspect
        + 0.20 * score_center
        + score_pref
        + score_mismatch
    )


def make_card(rec, card_w=1180, card_h=300):
    font_title = load_font(22, bold=True)
    font = load_font(18, bold=False)
    font_bold = load_font(18, bold=True)

    canvas = Image.new("RGB", (card_w, card_h), (250, 250, 250))
    d = ImageDraw.Draw(canvas)

    x = 20
    y = 15
    d.text((x, y), f"rank={rec['rank']} | row={rec['row_index']} | coco_index={rec['coco_index']}", fill=(0, 0, 0), font=font_title)
    y += 32
    d.text((x, y), f"label={rec['label_name']} | original/expected={rec['class_name']} | replacement={rec['replacement_object']}", fill=(0, 0, 0), font=font)
    y += 26
    d.text((x, y), f"area={rec['area_ratio']:.3f} | score={rec['score']:.3f} | bbox={rec['bbox']}", fill=(70, 70, 70), font=font)

    view_y = 105
    thumb_size = (245, 175)
    gap = 25

    labels = [
        ("Full + target box", rec["full_box_img"]),
        ("Object crop", rec["crop_img"]),
        ("Masked context", rec["masked_img"]),
    ]

    for i, (lab, img) in enumerate(labels):
        xx = 20 + i * (thumb_size[0] + gap)
        d.text((xx, view_y - 28), lab, fill=(0, 0, 0), font=font_bold)
        canvas.paste(resize_keep(img, thumb_size), (xx, view_y))

    # right-side note
    tx = 20 + 3 * (thumb_size[0] + gap) + 10
    note = [
        "Use this if:",
        "- target is visually clear",
        "- crop is recognizable",
        "- masked context is interpretable",
        "- replacement is intuitive",
    ]
    yy = view_y
    for line in note:
        d.text((tx, yy), line, fill=(30, 30, 30), font=font)
        yy += 25

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="tri_view_compat/outputs/framework_case_search/train_ooc")
    parser.add_argument("--label", type=int, default=1, choices=[0, 1], help="0=in-context, 1=out-of-context")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max_samples", type=int, default=8000)
    parser.add_argument("--top_n", type=int, default=80)
    parser.add_argument("--min_area", type=float, default=0.025)
    parser.add_argument("--max_area", type=float, default=0.35)
    parser.add_argument("--prefer_original", default="chair,couch,bed,dining table,person,tennis racket,sink,toilet,oven,microwave,horse,elephant")
    parser.add_argument("--prefer_replacement", default="guitar,elephant,horse,chair,tennis racket,skateboard,surfboard,umbrella,bottle,dog,cat")
    parser.add_argument("--target_pairs", default="", help="Optional exact pairs, e.g. chair:guitar,couch:horse")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "individual_cases"
    cases_dir.mkdir(exist_ok=True)

    prefer_original = parse_csv_list(args.prefer_original)
    prefer_replacement = parse_csv_list(args.prefer_replacement)
    target_pairs = parse_pairs(args.target_pairs)

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a label column.")

    df = df[df["label"] == args.label].copy()

    if target_pairs:
        keep = []
        for _, r in df.iterrows():
            pair = (str(r.get("class_name", "")).lower(), str(r.get("replacement_object", "")).lower())
            keep.append(pair in target_pairs)
        df = df[np.array(keep)].copy()

    if len(df) == 0:
        raise RuntimeError("No samples after filtering.")

    rng = random.Random(args.seed)
    indices = list(df.index)
    rng.shuffle(indices)
    indices = indices[: min(args.max_samples, len(indices))]

    records = []
    for idx in indices:
        row = df.loc[idx]

        img_path = resolve_path(row["image_path"], root)
        mask_path = resolve_path(row["mask_path"], root)

        if not img_path.exists() or not mask_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask_img = Image.open(mask_path)
        except Exception:
            continue

        w, h = img.size
        binary = mask_to_binary(mask_img, (w, h))
        bbox = bbox_from_binary(binary)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        area_ratio = float(binary.mean())
        if area_ratio < args.min_area or area_ratio > args.max_area:
            continue

        bw, bh = x2 - x1, y2 - y1
        if bw < 30 or bh < 30:
            continue

        pb = pad_bbox(bbox, w, h)
        crop = img.crop(pb)
        masked = make_masked_context(img, binary)
        full_box = draw_box(img, bbox)

        score = score_candidate(row, area_ratio, bbox, (w, h), prefer_original, prefer_replacement)

        records.append({
            "row_index": int(idx),
            "coco_index": row.get("coco_index", ""),
            "label": int(row["label"]),
            "label_name": "out-of-context" if int(row["label"]) == 1 else "in-context",
            "class_name": row.get("class_name", ""),
            "replacement_object": row.get("replacement_object", ""),
            "prior_label": row.get("prior_label", ""),
            "replacement_label": row.get("replacement_label", ""),
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "area_ratio": area_ratio,
            "bbox": f"{bbox}",
            "score": score,
            "full_box_img": full_box,
            "crop_img": crop,
            "masked_img": masked,
        })

    if not records:
        raise RuntimeError("No valid candidate found. Try increasing --max_samples or relaxing area thresholds.")

    records = sorted(records, key=lambda x: x["score"], reverse=True)
    records = records[: args.top_n]

    # Save individual cases and CSV
    csv_records = []
    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank
        safe_id = str(rec["coco_index"]).replace("/", "_")
        prefix = f"{rank:03d}_row{rec['row_index']}_coco{safe_id}_{rec['class_name']}_to_{rec['replacement_object']}"
        prefix = prefix.replace(" ", "_").replace("/", "_")

        rec["full_box_img"].save(cases_dir / f"{prefix}_full_box.jpg", quality=95)
        rec["crop_img"].save(cases_dir / f"{prefix}_crop.jpg", quality=95)
        rec["masked_img"].save(cases_dir / f"{prefix}_masked_context.jpg", quality=95)

        csv_records.append({
            k: v for k, v in rec.items()
            if not k.endswith("_img")
        })

    pd.DataFrame(csv_records).to_csv(out_dir / "framework_candidates.csv", index=False)

    # Save contact sheet
    cards = [make_card(rec) for rec in records[: min(30, len(records))]]
    sheet_w = max(c.width for c in cards)
    sheet_h = sum(c.height for c in cards) + 12 * (len(cards) - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    y = 0
    for c in cards:
        sheet.paste(c, (0, y))
        y += c.height + 12

    sheet.save(out_dir / "framework_candidates_sheet.jpg", quality=95)

    print(f"Saved candidates CSV: {out_dir / 'framework_candidates.csv'}")
    print(f"Saved contact sheet:  {out_dir / 'framework_candidates_sheet.jpg'}")
    print(f"Saved individual cases under: {cases_dir}")
    print()
    print("Top candidates:")
    for rec in csv_records[:10]:
        print(
            f"rank={rec['rank']:03d} row={rec['row_index']} coco={rec['coco_index']} "
            f"label={rec['label_name']} original={rec['class_name']} replacement={rec['replacement_object']} "
            f"area={rec['area_ratio']:.3f} score={rec['score']:.3f}"
        )


if __name__ == "__main__":
    main()
