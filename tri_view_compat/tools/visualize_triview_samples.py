import os
import argparse
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/ztl/COinCO"

def get_bbox_from_mask(mask):
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def make_object_crop(img, bbox, pad_ratio=0.08):
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
    img_arr = np.array(img).copy()
    mask_arr = np.array(mask)
    img_arr[mask_arr > 0] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(img_arr)

def draw_bbox(img, bbox):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="tri_view_compat/outputs/splits/test.csv")
    parser.add_argument("--out_dir", default="tri_view_compat/outputs/vis_triview")
    parser.add_argument("--num", type=int, default=24)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(ROOT, args.csv), dtype={"coco_index": str})

    # sample balanced if possible
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    n_each = args.num // 2

    samples = []
    if len(df0) > 0:
        samples.extend(df0.sample(min(n_each, len(df0)), random_state=args.seed).to_dict("records"))
    if len(df1) > 0:
        samples.extend(df1.sample(min(args.num - len(samples), len(df1)), random_state=args.seed).to_dict("records"))

    if len(samples) < args.num:
        remain = df.sample(args.num - len(samples), random_state=args.seed + 1).to_dict("records")
        samples.extend(remain)

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Visualizing {len(samples)} samples to {args.out_dir}")

    for i, row in enumerate(samples):
        idx = str(row["coco_index"])
        label = int(row["label"])
        original = str(row["class_name"])
        repl = str(row["replacement_object"])
        prior = int(row["prior_label"])

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        bbox = get_bbox_from_mask(mask)
        if bbox is None:
            print(f"[WARN] empty mask: {idx}")
            continue

        full_bbox = draw_bbox(img, bbox)
        crop = make_object_crop(img, bbox)
        masked_context = make_masked_context(img, mask)

        # mask visualization
        mask_rgb = Image.fromarray(np.stack([np.array(mask)] * 3, axis=-1))

        x1, y1, x2, y2 = bbox
        W, H = img.size
        area_ratio = ((x2 - x1 + 1) * (y2 - y1 + 1)) / (W * H)

        fig = plt.figure(figsize=(16, 5))
        title = (
            f"idx={idx} | label={label} | original={original} | replacement={repl} | "
            f"prior_label={prior} | bbox={bbox} | area={area_ratio:.3f}"
        )
        fig.suptitle(title, fontsize=11)

        axes = [
            ("Full + bbox", full_bbox),
            ("Mask", mask_rgb),
            ("Object crop", crop),
            ("Masked context", masked_context),
        ]

        for j, (name, im) in enumerate(axes):
            ax = plt.subplot(1, 4, j + 1)
            ax.imshow(im)
            ax.set_title(name)
            ax.axis("off")

        out_path = os.path.join(args.out_dir, f"{i:03d}_idx{idx}_label{label}.jpg")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)

        print(out_path)

if __name__ == "__main__":
    main()
