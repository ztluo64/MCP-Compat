import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


def draw_bbox(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    arr = np.asarray(mask)
    ys, xs = np.where(arr > 0)

    if len(xs) == 0:
        raise RuntimeError(f"Empty mask: {mask_path}")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    image = image.copy()
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=5,
    )

    return image


def build_prompt(obj, specialist):

    common = f"""The object to analyze is {obj}, which is located inside the red bounding box in the image.

Considering image quality is not a factor, determine whether the object {obj} inside the red bounding box is in-context or out-of-context.

Important:
- Assume that the object {obj} genuinely exists in the scene, regardless of how realistic or natural it appears visually.
- Analyze only the object {obj} inside the red bounding box.
- Do not reinterpret or substitute the given object with another category.
- Ignore visual quality, realism, and rendering quality.
"""

    if specialist == "cooccurrence":
        criterion = """
Criterion:
Co-occurrence: Determine whether the object inside the red bounding box would normally or commonly appear together with the other objects or scene elements in the image.
If the object is unusual or uncommon in this context, it is considered out-of-context.
"""

    elif specialist == "location":
        criterion = """
Criterion:
Location: Evaluate whether the object inside the red bounding box is placed in a physically and contextually reasonable position, such as being supported by a surface, on the ground, or in a plausible environment.
If the object is floating in the air, embedded in another object, or placed in an unusual spot, it is considered out-of-context.
"""

    elif specialist == "size":
        criterion = """
Criterion:
Size: Judge whether the object's size inside the red bounding box is reasonable relative to other objects and the environment in this image.
Always consider the object's real-world size, not as a toy or miniature version.
If its size is unusual or implausible relative to the scene, it is considered out-of-context.
"""

    else:
        raise ValueError(specialist)

    ending = """
The threshold is "unusual" or "uncommon", not merely "impossible".

For this score-based evaluation, do not provide an analysis.
The final decision must be exactly one of:

In-context
Out-of-context

Final decision:"""

    return common + criterion + ending


@torch.inference_mode()
def get_score(
    model,
    processor,
    messages,
    device,
    ic_token,
    ooc_token,
):
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    next_logits = outputs.logits[0, -1].float()

    ic_logit = next_logits[ic_token]
    ooc_logit = next_logits[ooc_token]

    pair_logits = torch.stack(
        [ic_logit, ooc_logit]
    )

    pair_probs = torch.softmax(
        pair_logits,
        dim=0,
    )

    return {
        "score_ooc": float(pair_probs[1].item()),
        "margin": float((ooc_logit - ic_logit).item()),
        "ic_logit": float(ic_logit.item()),
        "ooc_logit": float(ooc_logit.item()),
    }


def load_done_ids(path):
    path = Path(path)

    if not path.exists():
        return set()

    done = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                done.add(int(x["coco_index"]))

    return done


def main(args):

    model_path = args.model_path

    print("specialist:", args.specialist)
    print("model:", model_path)
    print("csv:", args.csv)
    print("rank/world:", args.rank, "/", args.world_size)
    print("output:", args.output)

    print("\nLoading model...")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        use_fast=False,
    )

    device = next(model.parameters()).device
    print("device:", device)

    ic_ids = processor.tokenizer(
        " In-context",
        add_special_tokens=False,
    )["input_ids"]

    ooc_ids = processor.tokenizer(
        " Out-of-context",
        add_special_tokens=False,
    )["input_ids"]

    print("IC tokenization :", ic_ids)
    print("OOC tokenization:", ooc_ids)

    if not ic_ids or not ooc_ids:
        raise RuntimeError("Empty verbalizer tokenization")

    ic_token = int(ic_ids[0])
    ooc_token = int(ooc_ids[0])

    if ic_token == ooc_token:
        raise RuntimeError(
            "First discriminative tokens are identical"
        )

    print("IC first token :", ic_token)
    print("OOC first token:", ooc_token)

    df = pd.read_csv(args.csv)

    required = {
        "coco_index",
        "label",
        "replacement_object",
        "image_path",
        "mask_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"CSV missing columns: {missing}"
        )

    df = (
        df.sort_values("coco_index")
        .reset_index(drop=True)
    )

    # deterministic multi-GPU sharding
    df = (
        df.iloc[args.rank::args.world_size]
        .reset_index(drop=True)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    done_ids = load_done_ids(out_path)

    if done_ids:
        print(
            f"Resuming: {len(done_ids)} samples "
            "already present."
        )

    mode = "a" if out_path.exists() else "w"

    with out_path.open(
        mode,
        encoding="utf-8",
    ) as fout:

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
        ):
            coco_index = int(row["coco_index"])

            if coco_index in done_ids:
                continue

            label = int(row["label"])
            obj = str(row["replacement_object"])

            boxed = draw_bbox(
                row["image_path"],
                row["mask_path"],
            )

            prompt = build_prompt(
                obj,
                args.specialist,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": boxed,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            score = get_score(
                model=model,
                processor=processor,
                messages=messages,
                device=device,
                ic_token=ic_token,
                ooc_token=ooc_token,
            )

            result = {
                "coco_index": coco_index,
                "label": label,
                "replacement_object": obj,
                "specialist": args.specialist,
                **score,
            }

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--specialist",
        required=True,
        choices=[
            "cooccurrence",
            "location",
            "size",
        ],
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the released COinCO specialist checkpoint.",
    )

    parser.add_argument(
        "--processor_path",
        required=True,
        help="Path to the Qwen2.5-VL-3B-Instruct processor/tokenizer.",
    )

    parser.add_argument(
        "--csv",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    main(args)
