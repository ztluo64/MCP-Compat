import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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
    img_arr = np.array(img).copy()
    mask_arr = np.array(mask)
    img_arr[mask_arr > 0] = np.array(fill, dtype=np.uint8)
    return Image.fromarray(img_arr)


def geometry_from_bbox(bbox, image_size):
    x1, y1, x2, y2 = bbox
    W, H = image_size

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    area_ratio = (bw * bh) / float(W * H)
    aspect_ratio = bw / max(bh, 1)

    geom = [
        x1 / W,
        y1 / H,
        x2 / W,
        y2 / H,
        cx / W,
        cy / H,
        bw / W,
        bh / H,
        area_ratio,
        aspect_ratio,
    ]
    return torch.tensor(geom, dtype=torch.float32)


class COinCOTriViewDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_size=224,
        crop_pad_ratio=0.08,
        masked_fill=(127, 127, 127),
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, dtype={"coco_index": str})
        self.crop_pad_ratio = crop_pad_ratio
        self.masked_fill = masked_fill

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        bbox = get_bbox_from_mask(mask)
        if bbox is None:
            bbox = (0, 0, image.size[0] - 1, image.size[1] - 1)

        object_crop = crop_with_padding(image, bbox, self.crop_pad_ratio)
        masked_context = make_masked_context(image, mask, self.masked_fill)
        geometry = geometry_from_bbox(bbox, image.size)

        replacement_object = str(row["replacement_object"])
        class_name = str(row["class_name"])
        text_prompt = f"a photo of a {replacement_object}"

        sample = {
            "coco_index": str(row["coco_index"]),
            "full_image": self.transform(image),
            "object_crop": self.transform(object_crop),
            "masked_context": self.transform(masked_context),
            "geometry": geometry,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "prior_label": torch.tensor(int(row["prior_label"]), dtype=torch.long),
            "replacement_object": replacement_object,
            "class_name": class_name,
            "text_prompt": text_prompt,
        }
        return sample
