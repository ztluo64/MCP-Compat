import os
import pandas as pd
from PIL import Image
import numpy as np

ROOT = "/home/ubuntu/ztl/COinCO"

context_train = os.path.join(ROOT, "task_data/context_prediction/balanced/training_val_data.csv")
context_test = os.path.join(ROOT, "task_data/context_prediction/testing_data.csv")

inpaint_train = os.path.join(ROOT, "task_data/inpainting_info/training_inpainting_info.csv")
inpaint_test = os.path.join(ROOT, "task_data/inpainting_info/testing_inpainting_info.csv")

prior_train = os.path.join(ROOT, "task_data/objects_from_context_prediction/training_data.csv")
prior_val = os.path.join(ROOT, "task_data/objects_from_context_prediction/validation_data.csv")
prior_test = os.path.join(ROOT, "task_data/objects_from_context_prediction/testing_data.csv")

def show_df(name, path):
    df = pd.read_csv(path)
    print(f"\n===== {name} =====")
    print("path:", path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("head:")
    print(df.head(5).to_string(index=False))
    if "label" in df.columns:
        print("label counts:")
        print(df["label"].value_counts().sort_index().to_string())
    return df

ctx_tr = show_df("context_train_balanced", context_train)
ctx_te = show_df("context_test", context_test)
inp_tr = show_df("inpaint_train", inpaint_train)
inp_te = show_df("inpaint_test", inpaint_test)
pri_tr = show_df("prior_train", prior_train)
pri_val = show_df("prior_val", prior_val)
pri_te = show_df("prior_test", prior_test)

print("\n===== ID overlap =====")
def overlap(a, b, name):
    sa = set(a["coco_index"].astype(str))
    sb = set(b["coco_index"].astype(str))
    print(f"{name}: {len(sa & sb)} / {len(sa)}")

overlap(ctx_tr, inp_tr, "context_train ∩ inpaint_train")
overlap(ctx_te, inp_te, "context_test ∩ inpaint_test")
overlap(ctx_tr, pri_tr, "context_train ∩ prior_train")
overlap(ctx_tr, pri_val, "context_train ∩ prior_val")
overlap(ctx_te, pri_te, "context_test ∩ prior_test")

print("\n===== joined test samples =====")
inp_te_map = inp_te.copy()
inp_te_map["coco_index"] = inp_te_map["coco_index"].astype(str)
inp_te_map = inp_te_map.set_index("coco_index")

pri_te_map = pri_te.copy()
pri_te_map["coco_index"] = pri_te_map["coco_index"].astype(str)
pri_te_map = pri_te_map.set_index("coco_index")

for _, row in ctx_te.head(12).iterrows():
    idx = str(row["coco_index"])
    item = {
        "coco_index": idx,
        "context_label": int(row["label"]),
    }
    if idx in inp_te_map.index:
        item.update({
            "original_object": inp_te_map.loc[idx, "class_name"],
            "replacement_object": inp_te_map.loc[idx, "replacement_object"],
            "object_index": inp_te_map.loc[idx, "object_index"],
        })
    if idx in pri_te_map.index:
        item["prior_label"] = int(pri_te_map.loc[idx, "label"])
    else:
        item["prior_label"] = None
    print(item)

print("\n===== image / mask / bbox sanity check =====")
for _, row in ctx_te.head(8).iterrows():
    idx = str(row["coco_index"])
    img_path = os.path.join(ROOT, "task_data/images/testing_images", f"{idx}.png")
    mask_path = os.path.join(ROOT, "task_data/masks/bbox_masks_testing", f"{idx}.png")

    ok_img = os.path.exists(img_path)
    ok_mask = os.path.exists(mask_path)
    print(f"\nidx={idx}, image_exists={ok_img}, mask_exists={ok_mask}")

    if ok_img and ok_mask:
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        arr = np.array(mask)
        ys, xs = np.where(arr > 0)

        print("image_size:", img.size)
        print("mask_size:", mask.size)
        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            w, h = img.size
            area_ratio = ((x2 - x1 + 1) * (y2 - y1 + 1)) / (w * h)
            print("bbox:", [int(x1), int(y1), int(x2), int(y2)])
            print("area_ratio:", round(float(area_ratio), 4))
        else:
            print("empty mask!")
