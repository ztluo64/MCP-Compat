import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size=28, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def resolve_path(p, root):
    p = Path(str(p))
    if p.is_absolute():
        return p
    return root / p


def mask_to_binary(mask_img, size):
    mask = mask_img.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)

    arr = np.array(mask)
    binary = arr > 127

    # COinCO masks can be foreground-white or inverted depending on export;
    # choose the smaller region as target if mask is mostly white.
    if binary.mean() > 0.5:
        binary = ~binary
    return binary


def bbox_from_binary(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def parse_bbox(s):
    try:
        obj = ast.literal_eval(str(s))
        if isinstance(obj, tuple) and len(obj) == 4:
            return tuple(int(x) for x in obj)
    except Exception:
        return None
    return None


def pad_bbox(bbox, w, h, pad_ratio=0.15):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def draw_box(img, bbox, color=(230, 35, 35), width=6):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    for i in range(width):
        d.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return out


def make_masked_context(img, binary, fill=(235, 235, 235)):
    arr = np.array(img.convert("RGB"))
    arr[binary] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(arr)


def make_mask_overlay(img, binary, color=(230, 35, 35), alpha=110):
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.array(overlay)
    arr[binary] = np.array([color[0], color[1], color[2], alpha], dtype=np.uint8)
    overlay = Image.fromarray(arr)
    return Image.alpha_composite(base, overlay).convert("RGB")


def resize_to_width(img, width):
    img = img.convert("RGB")
    if img.width == width:
        return img
    h = int(round(img.height * width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def fit_to_canvas(img, size, bg=(255, 255, 255)):
    img = img.convert("RGB")
    canvas = Image.new("RGB", size, bg)
    temp = img.copy()
    temp.thumbnail(size, Image.LANCZOS)
    x = (size[0] - temp.width) // 2
    y = (size[1] - temp.height) // 2
    canvas.paste(temp, (x, y))
    return canvas


def multiline_text(draw, xy, lines, font, fill=(0, 0, 0), line_gap=10):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def save_card(path, title, lines, size=(1100, 520), accent=(55, 90, 160)):
    title_font = load_font(38, bold=True)
    body_font = load_font(30, bold=False)
    small_font = load_font(26, bold=False)

    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)

    # top bar
    d.rectangle([0, 0, size[0], 72], fill=(245, 247, 250))
    d.rectangle([0, 0, 10, size[1]], fill=accent)

    d.text((34, 18), title, font=title_font, fill=(20, 20, 20))

    y = 110
    for item in lines:
        if isinstance(item, tuple):
            text, color, is_bold = item
        else:
            text, color, is_bold = item, (30, 30, 30), False
        font = load_font(30, bold=is_bold)
        d.text((42, y), text, font=font, fill=color)
        y += font.size + 18

    img.save(path, quality=95)


def save_prior_card(path, eval_obj, size=(1180, 760)):
    title_font = load_font(38, bold=True)
    body_font = load_font(28, bold=False)
    bold = load_font(28, bold=True)
    small = load_font(24, bold=False)

    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, size[0], 78], fill=(245, 247, 250))
    d.rectangle([0, 0, 10, size[1]], fill=(70, 125, 80))
    d.text((34, 19), "Masked-context object prior", font=title_font, fill=(20, 20, 20))

    top = eval_obj.get("top_expected_objects", [])
    cues = eval_obj.get("compatibility_cues", {})

    y = 115
    d.text((42, y), "Top expected objects from masked context:", font=bold, fill=(20, 20, 20))
    y += 48

    # Top-5 bars
    top5 = top[:5]
    max_prob = max([x["prob"] for x in top5] + [1e-8])
    bar_x = 260
    bar_w_max = 520
    for item in top5:
        name = item["name"]
        prob = float(item["prob"])
        rank = int(item["rank"])
        d.text((60, y), f"{rank}. {name}", font=body_font, fill=(30, 30, 30))
        d.rectangle([bar_x, y + 7, bar_x + bar_w_max, y + 32], outline=(200, 200, 200), width=2)
        d.rectangle([bar_x, y + 7, bar_x + int(bar_w_max * prob / max_prob), y + 32], fill=(120, 160, 120))
        d.text((bar_x + bar_w_max + 25, y), f"{prob * 100:.2f}%", font=body_font, fill=(30, 30, 30))
        y += 52

    y += 14
    d.line([42, y, size[0] - 42, y], fill=(220, 220, 220), width=2)
    y += 35

    repl_name = cues.get("replacement_class_name", eval_obj.get("replacement_object", "replacement"))
    p_repl = float(cues.get("p_replacement", 0.0))
    ratio = float(cues.get("ratio", 0.0))
    margin = float(cues.get("margin", 0.0))

    d.text((42, y), "Inserted replacement object:", font=bold, fill=(20, 20, 20))
    y += 45
    d.text((70, y), f"{repl_name}", font=load_font(34, bold=True), fill=(170, 45, 45))
    y += 58

    d.text((42, y), "Prior-replacement compatibility:", font=bold, fill=(20, 20, 20))
    y += 45
    d.text((70, y), f"p({repl_name} | context) = {p_repl:.8f}", font=body_font, fill=(170, 45, 45))
    y += 42
    d.text((70, y), f"margin = {margin:.6f}", font=body_font, fill=(60, 60, 60))
    y += 42
    d.text((70, y), f"ratio = {ratio:.6f}", font=body_font, fill=(60, 60, 60))

    img.save(path, quality=95)


def save_triptych(path, full_box, crop, masked, size_each=(460, 330)):
    title_font = load_font(30, bold=True)

    labels = ["Full image + target box", "Object crop", "Masked context"]
    imgs = [full_box, crop, masked]

    gap = 36
    top_h = 58
    w = size_each[0] * 3 + gap * 2
    h = top_h + size_each[1]
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    for i, (label, im) in enumerate(zip(labels, imgs)):
        x = i * (size_each[0] + gap)
        d.text((x + 8, 12), label, font=title_font, fill=(20, 20, 20))
        canvas.paste(fit_to_canvas(im, size_each), (x, top_h))

    canvas.save(path, quality=95)


def save_metadata_md(path, row, eval_obj):
    cues = eval_obj.get("compatibility_cues", {})
    lines = []
    lines.append("# Fig.1 rank-009 component metadata")
    lines.append("")
    lines.append("Recommended narrative:")
    lines.append("")
    lines.append("> The masked context suggests indoor/common objects, while the inserted replacement object is an elephant. The prior probability of elephant under the masked context is extremely low, leading to low prior-replacement compatibility and an out-of-context prediction.")
    lines.append("")
    lines.append("## Case information")
    lines.append("")
    lines.append(f"- rank: {row.get('rank')}")
    lines.append(f"- row_index: {row.get('row_index')}")
    lines.append(f"- coco_index: {row.get('coco_index')}")
    lines.append(f"- label: {row.get('label')} / out-of-context")
    lines.append(f"- expected/original object: {row.get('class_name')}")
    lines.append(f"- replacement object: {row.get('replacement_object')}")
    lines.append(f"- image_path: `{row.get('image_path')}`")
    lines.append(f"- mask_path: `{row.get('mask_path')}`")
    lines.append("")
    lines.append("## Model result")
    lines.append("")
    lines.append(f"- P(out-of-context): {eval_obj.get('prob_out_of_context'):.6f}")
    lines.append(f"- threshold: {eval_obj.get('threshold'):.4f}")
    lines.append(f"- prediction: out-of-context")
    lines.append(f"- correct: {eval_obj.get('correct')}")
    lines.append("")
    lines.append("## Prior compatibility cues")
    lines.append("")
    lines.append(f"- replacement class: {cues.get('replacement_class_name')}")
    lines.append(f"- p_replacement: {cues.get('p_replacement'):.8f}")
    lines.append(f"- log_p_replacement: {cues.get('log_p_replacement'):.6f}")
    lines.append(f"- p_max: {cues.get('p_max'):.8f}")
    lines.append(f"- margin: {cues.get('margin'):.8f}")
    lines.append(f"- normalized entropy: {cues.get('normalized_entropy'):.6f}")
    lines.append(f"- ratio: {cues.get('ratio'):.8f}")
    lines.append("")
    lines.append("## Top expected objects")
    lines.append("")
    for item in eval_obj.get("top_expected_objects", [])[:10]:
        lines.append(f"- {item['rank']}. {item['name']} ({item['prob'] * 100:.2f}%)")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", default="tri_view_compat/outputs/framework_case_search/ooc_for_fig1/ooc_framework_candidates.csv")
    parser.add_argument("--rank", type=int, default=9)
    parser.add_argument("--eval_json", default="tri_view_compat/outputs/framework_case_search/ooc_for_fig1/rank_009_model_eval.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="tri_view_compat/outputs/paper_figures/fig1_rank009_components")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(args.candidates_csv)
    row_df = cand[cand["rank"] == args.rank].copy()
    if len(row_df) != 1:
        raise RuntimeError(f"Expected exactly one row with rank={args.rank}, got {len(row_df)}")
    row = row_df.iloc[0]

    with open(args.eval_json, "r") as f:
        eval_obj = json.load(f)

    img_path = resolve_path(row["image_path"], root)
    mask_path = resolve_path(row["mask_path"], root)

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    binary = mask_to_binary(mask, img.size)

    bbox = parse_bbox(row.get("bbox", ""))
    if bbox is None:
        bbox = bbox_from_binary(binary)
    if bbox is None:
        raise RuntimeError("Could not determine bbox.")

    full_box = draw_box(img, bbox)
    mask_overlay = make_mask_overlay(img, binary)
    masked_context = make_masked_context(img, binary)
    crop = img.crop(pad_bbox(bbox, img.width, img.height, pad_ratio=0.16))

    # Raw visual components
    img.save(out_dir / "00_full_image_raw.png")
    full_box.save(out_dir / "01_full_image_target_box.png")
    mask_overlay.save(out_dir / "02_full_image_mask_overlay.png")
    crop.save(out_dir / "03_object_crop_replacement.png")
    masked_context.save(out_dir / "04_masked_context.png")

    # Standard-width versions for easy insertion into PPT/AI/Inkscape
    resize_to_width(full_box, 900).save(out_dir / "01_full_image_target_box_w900.png")
    resize_to_width(crop, 600).save(out_dir / "03_object_crop_replacement_w600.png")
    resize_to_width(masked_context, 900).save(out_dir / "04_masked_context_w900.png")

    # Text / reasoning cards
    replacement = str(row["replacement_object"])
    expected = str(row["class_name"])
    prob_ooc = float(eval_obj["prob_out_of_context"])
    threshold = float(eval_obj["threshold"])

    save_card(
        out_dir / "05_replacement_text_prompt_card.png",
        "Replacement object text",
        [
            ("Prompt:", (60, 60, 60), False),
            (f"a photo of an {replacement}" if replacement[0].lower() in "aeiou" else f"a photo of a {replacement}", (30, 30, 30), True),
            ("Encoded by frozen CLIP text encoder", (90, 90, 90), False),
        ],
        size=(1100, 360),
        accent=(90, 100, 170),
    )

    save_prior_card(out_dir / "06_masked_context_prior_card.png", eval_obj)

    cues = eval_obj.get("compatibility_cues", {})
    save_card(
        out_dir / "07_explicit_compatibility_cues_card.png",
        "Explicit compatibility cues",
        [
            (f"Inserted object: {replacement}", (30, 30, 30), True),
            (f"p({replacement} | context) = {float(cues.get('p_replacement', 0)):.8f}", (170, 45, 45), True),
            (f"margin = {float(cues.get('margin', 0)):.6f}", (60, 60, 60), False),
            (f"ratio = {float(cues.get('ratio', 0)):.6f}", (60, 60, 60), False),
            ("Compatibility: low", (170, 45, 45), True),
        ],
        size=(1100, 470),
        accent=(180, 80, 70),
    )

    save_card(
        out_dir / "08_final_prediction_card.png",
        "Final prediction",
        [
            (f"P(out-of-context) = {prob_ooc:.6f}", (170, 45, 45), True),
            (f"Validation threshold = {threshold:.2f}", (60, 60, 60), False),
            ("Prediction: out-of-context", (170, 45, 45), True),
            ("Ground truth: out-of-context", (60, 60, 60), False),
        ],
        size=(1100, 420),
        accent=(170, 45, 45),
    )

    save_card(
        out_dir / "09_case_identity_card.png",
        "Training sample for Fig.1",
        [
            (f"coco_index: {row.get('coco_index')}", (60, 60, 60), False),
            (f"Original / expected object: {expected}", (30, 30, 30), True),
            (f"Replacement object: {replacement}", (170, 45, 45), True),
            ("Label: out-of-context", (170, 45, 45), True),
        ],
        size=(1100, 430),
        accent=(90, 90, 90),
    )

    save_triptych(out_dir / "10_triptych_views.png", full_box, crop, masked_context)
    save_metadata_md(out_dir / "README_fig1_rank009_components.md", row, eval_obj)

    # Save compact JSON summary
    summary = {
        "rank": int(row["rank"]),
        "row_index": int(row["row_index"]),
        "coco_index": int(row["coco_index"]),
        "label": int(row["label"]),
        "class_name": str(row["class_name"]),
        "replacement_object": replacement,
        "bbox": str(bbox),
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "prob_out_of_context": prob_ooc,
        "threshold": threshold,
        "prediction": "out-of-context",
        "correct": bool(eval_obj["correct"]),
        "compatibility_cues": cues,
        "top_expected_objects": eval_obj.get("top_expected_objects", [])[:10],
    }
    with open(out_dir / "fig1_rank009_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Fig.1 components to: {out_dir}")
    print()
    print("Generated files:")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
