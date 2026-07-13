from pathlib import Path
import ast

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def parse_bbox(s):
    try:
        obj = ast.literal_eval(str(s))
        if isinstance(obj, tuple) and len(obj) == 4:
            return tuple(int(x) for x in obj)
    except Exception:
        pass
    return None


def mask_to_binary(mask_img, size):
    mask = mask_img.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)

    arr = np.array(mask)
    binary = arr > 127

    # 如果 mask 大部分为白色，说明可能反了，取较小区域作为 target
    if binary.mean() > 0.5:
        binary = ~binary

    return binary


def bbox_from_binary(binary):
    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(bbox, w, h, pad_ratio=0.16):
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


def resize_to_width(img, width):
    img = img.convert("RGB")
    h = int(round(img.height * width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def main():
    root = Path("/home/ubuntu/ztl/COinCO")
    csv_path = root / "tri_view_compat/outputs/framework_case_search/ooc_for_fig1/ooc_framework_candidates.csv"
    out_dir = root / "tri_view_compat/outputs/paper_figures/fig1_rank009_core_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    row = df[df["rank"] == 9].iloc[0]

    img_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    img = Image.open(img_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")

    binary = mask_to_binary(mask_img, img.size)

    bbox = parse_bbox(row.get("bbox", ""))
    if bbox is None:
        bbox = bbox_from_binary(binary)
    if bbox is None:
        raise RuntimeError("Could not get bbox from mask.")

    full_box = draw_box(img, bbox)
    crop = img.crop(pad_bbox(bbox, img.width, img.height))
    masked_context = make_masked_context(img, binary)

    # Target mask M: white target, black background
    target_mask = Image.fromarray((binary.astype(np.uint8) * 255), mode="L")

    # Target mask M: red target, white background, good for framework figure
    rgb_mask = np.ones((*binary.shape, 3), dtype=np.uint8) * 255
    rgb_mask[binary] = np.array([220, 45, 45], dtype=np.uint8)
    target_mask_red = Image.fromarray(rgb_mask)

    # Save original-resolution components
    full_box.save(out_dir / "full_image_I_with_target_box.png")
    crop.save(out_dir / "object_crop_Ir.png")
    masked_context.save(out_dir / "masked_context_Ic.png")
    target_mask.save(out_dir / "target_mask_M_binary.png")
    target_mask_red.save(out_dir / "target_mask_M_red.png")

    # Save normalized-width versions, easier for drawing Fig.1
    resize_to_width(full_box, 900).save(out_dir / "full_image_I_with_target_box_w900.png")
    resize_to_width(crop, 600).save(out_dir / "object_crop_Ir_w600.png")
    resize_to_width(masked_context, 900).save(out_dir / "masked_context_Ic_w900.png")
    resize_to_width(target_mask_red, 900).save(out_dir / "target_mask_M_red_w900.png")

    print("Saved core Fig.1 views to:")
    print(out_dir)
    print()
    for p in sorted(out_dir.iterdir()):
        print(p.name)


if __name__ == "__main__":
    main()
