import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Vector-friendly PDF text.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"


def open_rgb(path):
    return Image.open(path).convert("RGB")


def open_mask(path):
    return Image.open(path).convert("L")


def resolve_path(path, fallback):
    if isinstance(path, float) and np.isnan(path):
        path = ""
    path = str(path)
    if path and Path(path).exists():
        return path
    if Path(fallback).exists():
        return fallback
    return path


def get_bbox_from_mask(mask):
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_masked_context(img, mask, neutral=(140, 140, 140)):
    arr = np.array(img).copy()
    m = np.array(mask) > 0
    arr[m] = neutral
    return Image.fromarray(arr)


def prepare_square_canvas(img, mask=None, canvas_size=900):
    """
    Put the original image on a white square canvas while preserving aspect ratio.
    Return canvas image and normalized bbox coordinates if mask is provided.
    """
    w, h = img.size
    scale = min(canvas_size / w, canvas_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")

    ox = (canvas_size - new_w) // 2
    oy = (canvas_size - new_h) // 2
    canvas.paste(resized, (ox, oy))

    bbox_norm = None
    if mask is not None:
        bbox = get_bbox_from_mask(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = (x1 * scale + ox) / canvas_size
            y1 = (y1 * scale + oy) / canvas_size
            x2 = (x2 * scale + ox) / canvas_size
            y2 = (y2 * scale + oy) / canvas_size
            # Matplotlib image axis uses y from bottom if extent=[0,1,0,1].
            bbox_norm = (x1, 1.0 - y2, x2 - x1, y2 - y1)

    return canvas, bbox_norm


def format_percent(p):
    p = float(p)
    pct = p * 100.0
    if pct < 0.001:
        return "<0.001%"
    if pct < 1.0:
        return f"{pct:.1f}%"
    return f"{pct:.1f}%"


def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ["true", "1", "yes"]


def parse_topk(row, topk=3):
    s = row.get("topk_expected_objects", "")
    try:
        items = json.loads(str(s))
        return [(str(x["name"]), float(x["prob"])) for x in items[:topk]]
    except Exception:
        pass

    s = str(row.get("topk_expected_str", ""))
    out = []
    for part in s.split(";")[:topk]:
        part = part.strip()
        if ":" not in part:
            continue
        name, prob = part.rsplit(":", 1)
        prob = prob.replace("%", "").strip()
        try:
            prob = float(prob) / 100.0
        except Exception:
            prob = 0.0
        out.append((name.strip(), prob))
    return out


def short_label(label_name):
    label_name = str(label_name).lower()
    if "out" in label_name:
        return "OOC"
    return "IC"


def pred_short(pred_name):
    pred_name = str(pred_name).lower()
    if "out" in pred_name:
        return "OOC"
    return "IC"


def draw_image_panel(ax, img_canvas, bbox_norm=None, gt=None):
    ax.imshow(img_canvas, extent=[0, 1, 0, 1], interpolation="lanczos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Light boundary.
    ax.add_patch(
        patches.Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#D5D5D5",
            linewidth=0.45,
            clip_on=False,
        )
    )

    # Target bbox.
    if bbox_norm is not None:
        x, y, w, h = bbox_norm
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h,
                transform=ax.transAxes,
                fill=False,
                edgecolor="#D62728",
                linewidth=1.05,
                clip_on=False,
            )
        )

    # Compact GT badge.
    if gt is not None:
        color = "#8B1A1A" if gt == "OOC" else "#1B6E3B"
        badge_w = 0.145 if gt == "OOC" else 0.105
        badge_h = 0.078

        badge = patches.FancyBboxPatch(
            (0.043, 0.858),
            badge_w,
            badge_h,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
        )
        ax.add_patch(badge)
        ax.text(
            0.043 + badge_w / 2,
            0.858 + badge_h / 2,
            gt,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.9,
            fontweight="bold",
            color="white",
        )


def add_col_title(fig, x_center, y, text):
    fig.text(
        x_center,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=6.85,
        fontweight="bold",
        color="#111111",
    )


def draw_evidence_panel(ax, row):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    black = "#111111"
    dark = "#333333"
    gray = "#555555"
    red = "#B22222"
    green = "#16803A"

    gt = short_label(row["label_name"])
    rep = str(row["replacement_object"])

    base_pred = pred_short(row["baseline_pred_name"])
    ours_pred = pred_short(row["ours_pred_name"])
    base_prob = float(row["baseline_prob_ooc"])
    ours_prob = float(row["ours_prob_ooc"])
    base_ok = parse_bool(row["baseline_correct"])
    ours_ok = parse_bool(row["ours_correct"])

    topk = parse_topk(row, topk=3)
    p_r = float(row["p_replacement"])
    margin = float(row["margin"])

    if p_r * 100.0 < 0.001:
        p_r_text = "p_r < 0.001%"
    else:
        p_r_text = f"p_r = {format_percent(p_r)}"

    if margin * 100.0 < 0.001:
        margin_text = "margin < 0.001%"
    else:
        margin_text = f"margin = {format_percent(margin)}"

    x0 = 0.00

    # GT.
    y = 0.955
    ax.text(
        x0, y, f"GT: {gt}",
        fontsize=6.85,
        fontweight="bold",
        color=black,
        va="top",
    )

    # Replacement.
    y -= 0.118
    ax.text(
        x0, y, "Replacement:",
        fontsize=6.65,
        color=black,
        va="top",
    )
    ax.text(
        0.50, y, rep,
        fontsize=6.65,
        fontweight="bold",
        color=black,
        va="top",
    )

    # Prior.
    y -= 0.138
    ax.text(
        x0, y, "Prior:",
        fontsize=6.8,
        fontweight="bold",
        color=black,
        va="top",
    )

    y -= 0.108
    for i, (name, prob) in enumerate(topk):
        fw = "bold" if i == 0 else "normal"
        ax.text(
            0.03, y, f"{i + 1}. {name}",
            fontsize=6.45,
            fontweight=fw,
            color=black,
            va="top",
        )
        ax.text(
            0.88, y, format_percent(prob),
            fontsize=6.45,
            color=gray,
            va="top",
            ha="right",
        )
        y -= 0.092

    # Explicit cues, two lines to avoid crowding.
    y -= 0.012
    ax.text(
        x0, y, p_r_text,
        fontsize=6.5,
        fontweight="bold",
        color=dark,
        va="top",
    )
    y -= 0.092
    ax.text(
        x0, y, margin_text,
        fontsize=6.5,
        fontweight="bold",
        color=dark,
        va="top",
    )

    # Predictions.
    y -= 0.128
    base_mark = "✓" if base_ok else "×"
    base_color = green if base_ok else red
    ax.text(
        x0,
        y,
        f"Base: {base_mark} {base_pred},  P(OOC)={base_prob:.3f}",
        fontsize=6.55,
        fontweight="bold",
        color=base_color,
        va="top",
    )

    y -= 0.108
    ours_mark = "✓" if ours_ok else "×"
    ours_color = green if ours_ok else red
    ax.text(
        x0,
        y,
        f"Ours: {ours_mark} {ours_pred}, P(OOC)={ours_prob:.3f}",
        fontsize=6.55,
        fontweight="bold",
        color=ours_color,
        va="top",
    )


def build_case_rows(df, case_rows):
    out = []
    for rid in case_rows:
        sub = df[df["row_index"].astype(int) == int(rid)]
        if len(sub) == 0:
            raise ValueError(f"Cannot find row_index={rid} in CSV.")
        out.append(sub.iloc[0])
    return out


def ensure_paths(row):
    coco_index = int(row["coco_index"])
    image_fallback = f"task_data/images/testing_images/{coco_index}.png"
    mask_fallback = f"task_data/masks/bbox_masks_testing/{coco_index}.png"

    image_path = resolve_path(row.get("image_path", ""), image_fallback)
    mask_path = resolve_path(row.get("mask_path", ""), mask_fallback)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    return image_path, mask_path


def render_figure(rows, out_pdf, out_png, fig_w=3.45, fig_h=3.05):
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    left = 0.036
    right = 0.982
    top_title_y = 0.935

    col_gap = 0.026
    content_w = right - left
    usable_w = content_w - 2 * col_gap

    # Single-column layout: 24%, 24%, 52%.
    w_input = usable_w * 0.24
    w_mask = usable_w * 0.24
    w_ev = usable_w * 0.52

    x_input = left
    x_mask = x_input + w_input + col_gap
    x_ev = x_mask + w_mask + col_gap

    row_h = 0.34
    y_top = 0.535
    y_bottom = 0.105

    # Make image axes square in physical units.
    img_ax_h = w_input * fig_w / fig_h
    img_y_offset = (row_h - img_ax_h) / 2

    # Compact titles.
    add_col_title(fig, x_input + w_input / 2, top_title_y, "Input + target")
    add_col_title(fig, x_mask + w_mask / 2, top_title_y, "Masked ctx.")
    add_col_title(fig, x_ev + w_ev / 2, top_title_y, "Evidence + predictions")

    y_positions = [y_top, y_bottom]

    for row, y0 in zip(rows, y_positions):
        image_path, mask_path = ensure_paths(row)
        img = open_rgb(image_path)
        mask = open_mask(mask_path)

        input_canvas, bbox_norm = prepare_square_canvas(img, mask, canvas_size=900)
        masked_img = make_masked_context(img, mask)
        masked_canvas, _ = prepare_square_canvas(masked_img, None, canvas_size=900)

        gt = short_label(row["label_name"])

        ax_input = fig.add_axes([x_input, y0 + img_y_offset, w_input, img_ax_h])
        draw_image_panel(ax_input, input_canvas, bbox_norm=bbox_norm, gt=gt)

        ax_mask = fig.add_axes([x_mask, y0 + img_y_offset, w_mask, img_ax_h])
        draw_image_panel(ax_mask, masked_canvas, bbox_norm=None, gt=None)

        ax_ev = fig.add_axes([x_ev, y0, w_ev, row_h])
        draw_evidence_panel(ax_ev, row)

    # Separator line between two rows.
    sep_y = (y_top + y_bottom + row_h) / 2
    fig.add_artist(
        Line2D(
            [left, right],
            [sep_y, sep_y],
            transform=fig.transFigure,
            color="#D9D9D9",
            linewidth=0.55,
        )
    )

    out_pdf = Path(out_pdf)
    out_png = Path(out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Keep exact physical width. Do not use bbox_inches="tight".
    fig.savefig(out_pdf, format="pdf", facecolor="white")
    fig.savefig(out_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="tri_view_compat/outputs/qualitative_selection/selected_cases.csv",
    )
    parser.add_argument("--out_pdf", default="figures/fig_2.pdf")
    parser.add_argument("--out_png", default="figures/fig_2_preview.png")
    parser.add_argument(
        "--case_rows",
        nargs="+",
        type=int,
        default=[377, 308],
        help="row_index values to render. Default: OOC sheep->sandwich and IC tennis-racket case.",
    )
    parser.add_argument("--fig_h", type=float, default=3.05)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rows = build_case_rows(df, args.case_rows)

    render_figure(
        rows=rows,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
        fig_w=3.45,
        fig_h=args.fig_h,
    )

    print(f"[Saved] {args.out_pdf}")
    print(f"[Saved] {args.out_png}")
    print("[Info] Figure width is fixed to 3.45 inches.")
    print("[Info] PDF text, boxes, and lines are vector; image panels are raster.")


if __name__ == "__main__":
    main()
