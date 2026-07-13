import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size=22, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def resolve_path(path, root):
    path = Path(str(path))
    if path.is_absolute():
        return path
    return root / path


def parse_words(s):
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
            raise ValueError(f"Bad pair: {item}. Expected format original:replacement")
        a, b = item.split(":", 1)
        pairs.add((a.strip().lower(), b.strip().lower()))
    return pairs


def mask_to_binary(mask_img, size):
    mask = mask_img.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)

    arr = np.array(mask)
    binary = arr > 127

    # If mask is mostly white, assume it is inverted.
    if binary.mean() > 0.50:
        binary = ~binary

    return binary


def bbox_from_binary(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(bbox, w, h, pad_ratio=0.15):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def draw_box(img, bbox, color=(230, 30, 30), width=5):
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


def resize_keep(img, size=(260, 190), bg=(255, 255, 255)):
    img = img.convert("RGB")
    canvas = Image.new("RGB", size, bg)
    temp = img.copy()
    temp.thumbnail(size, Image.LANCZOS)
    x = (size[0] - temp.width) // 2
    y = (size[1] - temp.height) // 2
    canvas.paste(temp, (x, y))
    return canvas


def object_pair_score(original, replacement, prefer_original, prefer_replacement, strong_pairs):
    original = original.lower()
    replacement = replacement.lower()

    score = 0.0

    # Strong manual pairs are best for framework figures.
    if (original, replacement) in strong_pairs:
        score += 1.00

    # Prefer intuitive original/expected objects.
    if original in prefer_original:
        score += 0.25

    # Prefer visually intuitive replacement objects.
    if replacement in prefer_replacement:
        score += 0.30

    # Prefer obvious mismatch.
    if original != replacement:
        score += 0.20

    # Penalize ambiguous / tiny / stuff-like categories a bit.
    weak_words = {
        "book", "remote", "cell phone", "keyboard", "mouse", "fork", "knife",
        "spoon", "tie", "handbag", "backpack"
    }
    if original in weak_words:
        score -= 0.10
    if replacement in weak_words:
        score -= 0.12

    return score


def visual_score(area_ratio, bbox, image_size):
    w, h = image_size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Method figure prefers medium-sized target region.
    target_area = 0.10
    area_score = math.exp(-abs(math.log((area_ratio + 1e-8) / target_area)))

    # Avoid very elongated boxes.
    aspect = bw / max(bh, 1)
    aspect_score = math.exp(-abs(math.log(max(aspect, 1e-8))))

    # Prefer target not too close to image boundary.
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    center_score = max(0.0, 1.0 - 1.35 * dist)

    # Prefer sufficiently large crop in pixels.
    pixel_score = min(1.0, min(bw, bh) / 80.0)

    return 0.35 * area_score + 0.25 * aspect_score + 0.25 * center_score + 0.15 * pixel_score


def make_card(rec, card_w=1280, card_h=360):
    title_font = load_font(24, bold=True)
    font = load_font(19, bold=False)
    bold = load_font(19, bold=True)
    small = load_font(17, bold=False)

    canvas = Image.new("RGB", (card_w, card_h), (250, 250, 250))
    d = ImageDraw.Draw(canvas)

    x0, y0 = 20, 15
    d.text(
        (x0, y0),
        f"Rank {rec['rank']:03d} | row={rec['row_index']} | coco_index={rec['coco_index']}",
        font=title_font,
        fill=(0, 0, 0),
    )
    y0 += 34
    d.text(
        (x0, y0),
        f"OOC case: expected/original = {rec['class_name']}   →   replacement = {rec['replacement_object']}",
        font=bold,
        fill=(160, 40, 40),
    )
    y0 += 28
    d.text(
        (x0, y0),
        f"score={rec['score']:.3f} | visual={rec['visual_score']:.3f} | pair={rec['pair_score']:.3f} | area={rec['area_ratio']:.3f} | bbox={rec['bbox']}",
        font=small,
        fill=(70, 70, 70),
    )

    view_y = 125
    thumb_size = (285, 205)
    gap = 25

    views = [
        ("Full image + target box", rec["full_box_img"]),
        ("Object crop", rec["crop_img"]),
        ("Masked context", rec["masked_img"]),
    ]

    for i, (name, img) in enumerate(views):
        xx = 20 + i * (thumb_size[0] + gap)
        d.text((xx, view_y - 28), name, font=bold, fill=(0, 0, 0))
        canvas.paste(resize_keep(img, thumb_size), (xx, view_y))

    tx = 20 + 3 * (thumb_size[0] + gap) + 10
    d.text((tx, view_y - 28), "Why useful for Fig.1", font=bold, fill=(0, 70, 0))
    notes = [
        "✓ out-of-context training sample",
        "✓ clear expected-vs-inserted mismatch",
        "✓ can illustrate low compatibility",
        "✓ good for prior-replacement reasoning",
        "",
        f"Prompt: a photo of a {rec['replacement_object']}",
    ]
    yy = view_y
    for line in notes:
        d.text((tx, yy), line, font=font, fill=(30, 30, 30))
        yy += 27

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tri_view_compat/outputs/splits/train.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="tri_view_compat/outputs/framework_case_search/ooc_for_fig1")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--min_area", type=float, default=0.025)
    parser.add_argument("--max_area", type=float, default=0.32)
    parser.add_argument("--min_short_side", type=int, default=35)
    parser.add_argument(
        "--prefer_original",
        default="chair,couch,bed,dining table,toilet,sink,oven,microwave,refrigerator,tennis racket,person,horse,car,bus,bicycle,motorcycle",
    )
    parser.add_argument(
        "--prefer_replacement",
        default="guitar,elephant,horse,dog,cat,tennis racket,skateboard,surfboard,umbrella,bottle,chair,person",
    )
    parser.add_argument(
        "--strong_pairs",
        default=(
            "chair:guitar,couch:guitar,chair:elephant,couch:elephant,"
            "toilet:elephant,sink:elephant,oven:horse,microwave:horse,"
            "tennis racket:elephant,bed:guitar,dining table:horse"
        ),
    )
    parser.add_argument("--save_top", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "individual_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    prefer_original = parse_words(args.prefer_original)
    prefer_replacement = parse_words(args.prefer_replacement)
    strong_pairs = parse_pairs(args.strong_pairs)

    df = pd.read_csv(args.csv)

    required = ["label", "image_path", "mask_path", "class_name", "replacement_object"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in split CSV: {missing}")

    # Only train OOC cases.
    df = df[df["label"] == 1].copy()
    if len(df) == 0:
        raise RuntimeError("No out-of-context samples found in CSV.")

    rng = random.Random(args.seed)
    all_indices = list(df.index)
    rng.shuffle(all_indices)
    all_indices = all_indices[: min(args.max_samples, len(all_indices))]

    records = []
    for idx in all_indices:
        row = df.loc[idx]
        original = str(row["class_name"]).strip().lower()
        replacement = str(row["replacement_object"]).strip().lower()

        # For Fig.1, avoid trivial same-class cases.
        if original == replacement:
            continue

        img_path = resolve_path(row["image_path"], root)
        mask_path = resolve_path(row["mask_path"], root)

        if not img_path.exists() or not mask_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
        except Exception:
            continue

        w, h = img.size
        binary = mask_to_binary(mask, (w, h))
        bbox = bbox_from_binary(binary)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        area_ratio = float(binary.mean())

        if area_ratio < args.min_area or area_ratio > args.max_area:
            continue

        if min(bw, bh) < args.min_short_side:
            continue

        aspect = bw / max(bh, 1)
        if aspect > 3.5 or aspect < 1 / 3.5:
            continue

        full_box = draw_box(img, bbox)
        crop = img.crop(pad_bbox(bbox, w, h))
        masked = make_masked_context(img, binary)

        vscore = visual_score(area_ratio, bbox, (w, h))
        pscore = object_pair_score(original, replacement, prefer_original, prefer_replacement, strong_pairs)
        total = vscore + pscore

        records.append({
            "row_index": int(idx),
            "coco_index": row.get("coco_index", ""),
            "label": int(row["label"]),
            "class_name": str(row["class_name"]),
            "replacement_object": str(row["replacement_object"]),
            "prior_label": row.get("prior_label", ""),
            "replacement_label": row.get("replacement_label", ""),
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "bbox": f"({x1}, {y1}, {x2}, {y2})",
            "area_ratio": area_ratio,
            "visual_score": vscore,
            "pair_score": pscore,
            "score": total,
            "full_box_img": full_box,
            "crop_img": crop,
            "masked_img": masked,
        })

    if not records:
        raise RuntimeError(
            "No valid OOC candidate found. Try increasing --max_samples or relaxing --min_area/--max_area."
        )

    records = sorted(records, key=lambda x: x["score"], reverse=True)
    records = records[: args.top_n]

    csv_rows = []
    for rank, rec in enumerate(records, 1):
        rec["rank"] = rank

        safe = f"{rank:03d}_row{rec['row_index']}_coco{rec['coco_index']}_{rec['class_name']}_to_{rec['replacement_object']}"
        safe = safe.replace(" ", "_").replace("/", "_")

        rec["full_box_img"].save(cases_dir / f"{safe}_full_box.jpg", quality=95)
        rec["crop_img"].save(cases_dir / f"{safe}_object_crop.jpg", quality=95)
        rec["masked_img"].save(cases_dir / f"{safe}_masked_context.jpg", quality=95)

        csv_rows.append({
            k: v for k, v in rec.items()
            if not k.endswith("_img")
        })

    pd.DataFrame(csv_rows).to_csv(out_dir / "ooc_framework_candidates.csv", index=False)

    # Contact sheet for quick manual inspection.
    cards = [make_card(rec) for rec in records[: args.save_top]]
    sheet_w = max(c.width for c in cards)
    sheet_h = sum(c.height for c in cards) + 12 * (len(cards) - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))

    y = 0
    for card in cards:
        sheet.paste(card, (0, y))
        y += card.height + 12

    sheet.save(out_dir / "ooc_framework_candidates_sheet.jpg", quality=95)

    print(f"Saved CSV:           {out_dir / 'ooc_framework_candidates.csv'}")
    print(f"Saved contact sheet: {out_dir / 'ooc_framework_candidates_sheet.jpg'}")
    print(f"Saved images under:  {cases_dir}")
    print()
    print("Top OOC candidates:")
    for r in csv_rows[:15]:
        print(
            f"rank={r['rank']:03d} "
            f"row={r['row_index']} "
            f"coco={r['coco_index']} "
            f"expected={r['class_name']} "
            f"replacement={r['replacement_object']} "
            f"area={r['area_ratio']:.3f} "
            f"score={r['score']:.3f}"
        )


if __name__ == "__main__":
    main()
