#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.executable
import os
os.getcwd()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import pandas as pd
from tqdm import tqdm
import csv
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Load InternVL model
model_path = config.INTERNVL_38B_MODEL
print("Loading InternVL model and tokenizer...")
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
print("Model loaded successfully!")

# Generation config
generation_config = dict(max_new_tokens=1024, do_sample=True)



# In[ ]:


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def generate_prompt(new_object):
    prompt = f"""The object to analyze is {new_object}, which is located inside the red bounding box in the image.
Considering image quality is not a factor, do you think the object {new_object} inside the red bounding box is in-context or out-of-context?
Important: Please assume that the object {new_object} inside the red bounding box genuinely exists in the scene, regardless of how realistic or natural it appears visually. Even if the object looks artificially added, poorly rendered, or unrealistic in terms of visual quality, treat it as if it is truly present in that location. Focus only on whether its presence makes contextual sense, not on its visual realism.

The criteria for determining whether an object is in- or out-of-context are as follows:
1. Location: Evaluate whether the object inside the red bounding box is placed in a physically and contextually reasonable position, such as being supported by a surface, on the ground, or in a plausible environment.
   If the object is floating in the air, embedded in another object, or placed in an unusual spot, it is considered out-of-context.
2. Co-occurrence: Determine whether the object inside the red bounding box can reasonably appear together with other objects in the scene.
   If the combination of objects is unusual or uncommon in real-world contexts, it is considered out-of-context.
3. Size: Judge whether the object's size inside the red bounding box is reasonable relative to other objects and the environment in this image.
   For example, a banana larger than a chair or a cat larger than a car would be unusual and out-of-context.

Important notes:
- If the object violates ANY ONE of these three criteria (even just one), this object will be considered as out-of-context.
- The threshold is "unusual" or "uncommon", NOT "impossible". If something is uncommon or atypical in real-world contexts, it should be considered out-of-context.
- Each criterion is evaluated independently. Your summary judgment for each criterion must be consistent with your analysis. If you identify something as unusual, uncommon, or abnormal in your analysis for a specific criterion, then that criterion's summary MUST be marked as out-of-context. Conversely, if your analysis describes the situation as normal, common, or typical, then mark it as in-context.
- Analyze only the object {new_object} inside the red bounding box.
- Do not reinterpret or substitute the given object with another category. Even if the object visually resembles another type, always analyze it strictly as the given object {new_object}.
- Ignore the visual quality, realism, or rendering quality of the object. Focus solely on whether its contextual presence makes sense.
Please provide the analysis of the object according to all three criteria, then give a summary of each criterion, and finally provide your final decision.

For example, your answer could be:

Analysis:
Location: The {new_object} inside the red bounding box is floating in mid-air without any support.
Co-occurrence: It is unusual for a {new_object} to appear together with [other objects in the scene].
Size: The size of this {new_object} looks normal in this environment.

Summary:
Location: Out-of-context
Co-occurrence: Out-of-context
Size: In-context
Final decision: Out-of-context.

Or your answer could be:

Analysis:
Location: The {new_object} inside the red bounding box is normally placed on the ground/surface.
Co-occurrence: A {new_object} commonly appears together with [other objects in the scene].
Size: The {new_object} looks proportionally normal in this environment.

Summary:
Location: In-context
Co-occurrence: In-context
Size: In-context
Final decision: In-context.
Your answer is:"""
    return prompt

    
def internvl_response(prompt, image_path, model, tokenizer, generation_config):
    # 加载并预处理图片
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # 构建完整question（加上<image>标记）
    question = f'<image>\n{prompt}'
    
    # 单图单轮对话
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    
    return response

    
def process_csv(csv_file_path, model, tokenizer, generation_config, source_path=config.TRAINING_IMAGES_WITH_BBOX):
    df = pd.read_csv(csv_file_path)
    output_file = os.path.join(config.CONTEXT_REASONING_OUTPUT_ROOT, "context_reasoning_internvl_part2_continue.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['coco_index', 'response'])
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            coco_index = row['coco_index']
            if coco_index > 107_504:
                replacement_object = row['replacement_object']
                image_path = os.path.join(source_path, f"{coco_index}.png")
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                try:
                    prompt = generate_prompt(replacement_object)
                    response = internvl_response(prompt, image_path, model, tokenizer, generation_config)
                    print(f'{response}')
                    writer.writerow([coco_index, response])
                    file.flush()
                except Exception as e:
                    print(f"Error processing {coco_index}: {str(e)}")
                    continue

            
# Process CSV
csv_file_path = config.TRAINING_CSV
source_path = config.TRAINING_IMAGES_WITH_BBOX
process_csv(csv_file_path, model, tokenizer, generation_config, source_path)



# In[ ]:




