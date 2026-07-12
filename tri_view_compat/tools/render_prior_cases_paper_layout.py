import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_font(size=28, bold=False):
    candidates = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)

    return ImageFont.load_default()


def get_bbox_from_mask(mask):
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_with_padding(img, bbox, pad_ratio=0.08):
    w, h = img.size
    x1, y1, x2, y2 = bbox

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return img.crop((x1, y1, x2 + 1, y2 + 1))


def make_masked_context(img, mask, fill=(127, 127, 127)):
    arr = np.array(img).copy()
    m = np.array(mask)
    arr[m > 0] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(arr)


def resize_contain(img, box_w, box_h, bg=(255, 255, 255)):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img_r = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (box_w, box_h), bg)
    x = (box_w - new_w) // 2
    y = (box_h - new_h) // 2
    canvas.paste(img_r, (x, y))
    return canvas


def draw_wrapped(draw, text, xy, font, fill, max_width, line_gap=6):
    x, y = xy
    words = str(text).split()
    lines = []
    cur = ""

    for w in words:
        trial = w if not cur else cur + " " + w
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap

    return y


def parse_top5(top5_str):
    items = []
    if pd.isna(top5_str):
        return items

    for part in str(top5_str).split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, val = part.rsplit(":", 1)
        name = name.strip()
        try:
            prob = float(val)
        except Exception:
            continue
        items.append((name, prob))

    return items[:5]


def label_name(y):
    return "in-context" if int(y) == 0 else "out-of-context"


def draw_target_box(img, bbox, color, width=5):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return img


def choose_interpretation(case_row):
    correct = bool(int(case_row["correct"]))
    label = int(case_row["label"])
    pred = int(case_row["pred"])
    p_rep = float(case_row["p_replacement"])
    rank = int(case_row["replacement_rank"])

    if not correct:
        return (
            "Failure: an incorrect or ambiguous context prior may give a low prior "
            "to a plausible replacement object and mislead the final decision."
        )

    if label == 0 and pred == 0:
        if p_rep >= 0.30 or rank <= 3:
            return (
                "High p(replacement | context) supports a context match "
                "for the inserted object."
            )
        else:
            return (
                "Although the final prediction is correct, the context prior is not "
                "highly confident about the replacement object."
            )

    if label == 1 and pred == 1:
        return (
            "Low p(replacement | context) usually indicates a context mismatch "
            "for the inserted object."
        )

    return "The prior compatibility provides an interpretable cue for the final decision."


def draw_case_card(case_no, case_row, split_row, card_w=2300, card_h=790):
    correct = bool(int(case_row["correct"]))
    label = int(case_row["label"])
    pred = int(case_row["pred"])

    green = (20, 125, 55)
    red = (200, 30, 30)
    black = (20, 20, 20)
    gray = (95, 95, 95)
    light_green = (232, 247, 237)
    light_red = (253, 235, 235)
    border = green if correct else red
    bg = light_green if correct else light_red

    card = Image.new("RGB", (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    # Fonts
    f_title = load_font(40, bold=True)
    f_subtitle = load_font(27, bold=True)
    f_body = load_font(27, bold=False)
    f_body_bold = load_font(28, bold=True)
    f_small = load_font(24, bold=False)
    f_badge = load_font(38, bold=True)

    # Card background and border
    margin = 18
    draw.rounded_rectangle(
        [margin, margin, card_w - margin, card_h - margin],
        radius=24,
        fill=bg,
        outline=border,
        width=4,
    )

    # Case badge
    badge_x, badge_y = 36, 34
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + 66, badge_y + 58],
        radius=10,
        fill=border,
    )
    draw.text((badge_x + 22, badge_y + 7), str(case_no), font=f_badge, fill=(255, 255, 255))

    idx = int(case_row["idx"])
    coco_index = str(case_row["coco_index"])
    title = f"Case idx={idx} | coco_index={coco_index}"
    draw.text((120, 42), title, font=f_title, fill=black)

    # Load images
    img = Image.open(split_row["image_path"]).convert("RGB")
    mask = Image.open(split_row["mask_path"]).convert("L")
    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        bbox = (0, 0, img.size[0] - 1, img.size[1] - 1)

    full_boxed = draw_target_box(img, bbox, (255, 0, 0), width=5)
    crop = crop_with_padding(img, bbox)
    masked = make_masked_context(img, mask)

    # Layout
    top_y = 112
    img_w, img_h = 520, 365
    gap_x = 55

    x1 = 70
    x2 = x1 + img_w + gap_x
    x3 = x2 + img_w + gap_x

    text_x = x3 + img_w + 65
    text_w = card_w - text_x - 80

    # Titles
    draw.text((x1 + 95, top_y - 38), "Full image + target box", font=f_subtitle, fill=black)
    draw.text((x2 + 155, top_y - 38), "Object crop", font=f_subtitle, fill=black)
    draw.text((x3 + 130, top_y - 38), "Masked context", font=f_subtitle, fill=black)

    # Images
    card.paste(resize_contain(full_boxed, img_w, img_h), (x1, top_y))
    card.paste(resize_contain(crop, img_w, img_h), (x2, top_y))
    card.paste(resize_contain(masked, img_w, img_h), (x3, top_y))

    # Left information below first image
    info_y = top_y + img_h + 28
    rep = str(case_row["replacement_object"])
    prob_ooc = float(case_row["prob_ooc"]) * 100.0

    info_lines = [
        (f"Replacement object: {rep}", border),
        (f"GT: {label_name(label)}", border),
        (f"Prediction: {label_name(pred)}", border),
        (f"P(out-of-context): {prob_ooc:.1f}%", border),
        (f"Correct: {str(correct)}", border),
    ]

    y = info_y
    for txt, col in info_lines:
        draw.text((x1, y), txt, font=f_body_bold, fill=col)
        y += 40

    # Top-5 expected objects below middle image
    top5_x = x2 + 10
    top5_y = info_y
    draw.text((top5_x, top5_y), "Top-5 expected objects", font=f_body_bold, fill=black)
    draw.text((top5_x, top5_y + 34), "from masked context:", font=f_body_bold, fill=black)

    y = top5_y + 82
    top5 = parse_top5(case_row.get("top5", ""))
    for i, (name, p) in enumerate(top5, 1):
        draw.text((top5_x, y), f"{i}. {name}: {p * 100:.1f}%", font=f_body, fill=black)
        y += 38

    # Prior compatibility right panel
    y = top_y + 10
    draw.text((text_x, y), "Prior compatibility:", font=f_body_bold, fill=black)
    y += 55

    p_rep = float(case_row["p_replacement"]) * 100.0
    rank = int(case_row["replacement_rank"])

    draw.text((text_x, y), f"p(replacement | context): {p_rep:.2f}%", font=f_body, fill=black)
    y += 40
    draw.text((text_x, y), f"replacement rank: {rank}", font=f_body, fill=black)
    y += 70

    draw.text((text_x, y), "Interpretation:", font=f_body_bold, fill=black)
    y += 44

    interp = choose_interpretation(case_row)
    draw_wrapped(draw, interp, (text_x, y), f_body, black, text_w, line_gap=7)

    return card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case_summary",
        default="tri_view_compat/outputs/case_vis/prior_v2_original/case_summary.csv",
    )
    parser.add_argument(
        "--split_csv",
        default="tri_view_compat/outputs/splits/test.csv",
    )
    parser.add_argument(
        "--case_indices",
        nargs="+",
        type=int,
        default=[281, 168, 879],
        help="Use the 'idx' column from case_summary.csv / test split.",
    )
    parser.add_argument(
        "--out_dir",
        default="tri_view_compat/outputs/paper_figures/case_vis_selected_v2",
    )
    parser.add_argument("--card_w", type=int, default=2300)
    parser.add_argument("--card_h", type=int, default=790)
    parser.add_argument("--gap", type=int, default=28)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_df = pd.read_csv(args.case_summary)
    split_df = pd.read_csv(args.split_csv)

    # case idx is the integer row index in test split.
    case_df["idx"] = case_df["idx"].astype(int)

    cards = []
    selected_rows = []

    for case_no, idx in enumerate(args.case_indices, 1):
        rows = case_df[case_df["idx"] == idx]
        if len(rows) == 0:
            raise RuntimeError(f"Cannot find idx={idx} in {args.case_summary}")

        case_row = rows.iloc[0]
        split_row = split_df.iloc[idx]

        card = draw_case_card(
            case_no=case_no,
            case_row=case_row,
            split_row=split_row,
            card_w=args.card_w,
            card_h=args.card_h,
        )
        cards.append(card)
        selected_rows.append(case_row)

        indiv_path = out_dir / f"case_{case_no}_idx{idx}.png"
        card.save(indiv_path)
        print("saved:", indiv_path)

    sheet_w = args.card_w
    sheet_h = args.card_h * len(cards) + args.gap * (len(cards) - 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))

    y = 0
    for card in cards:
        sheet.paste(card, (0, y))
        y += args.card_h + args.gap

    out_png = out_dir / "selected_cases_paper_layout.png"
    out_pdf = out_dir / "selected_cases_paper_layout.pdf"
    out_csv = out_dir / "selected_case_summary.csv"

    sheet.save(out_png)
    sheet.save(out_pdf, "PDF", resolution=300.0)

    pd.DataFrame(selected_rows).to_csv(out_csv, index=False)

    print("saved:", out_png)
    print("saved:", out_pdf)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
