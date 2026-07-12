#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl 
# pip install transformers


# In[2]:


import sys
sys.executable


# In[3]:


import os
os.getcwd()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config


# In[4]:


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# In[5]:


import pandas as pd
from tqdm import tqdm
import csv
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# In[6]:





# In[23]:


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

# In[24]:


def qwen_response(prompt, image_path, model, processor):
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


# In[25]:


def process_csv(csv_file_path, model, processor, source_path=config.TRAINING_IMAGES_WITH_BBOX):
    df = pd.read_csv(csv_file_path)
    output_file = os.path.join(config.CONTEXT_REASONING_OUTPUT_ROOT, "context_reasoning_qwen_part2.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['coco_index', 'response'])
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            coco_index = row['coco_index']
            if coco_index > 60_000:
                replacement_object = row['replacement_object']
                image_path = os.path.join(source_path, f"{coco_index}.png")
                prompt = generate_prompt(replacement_object)
                response = qwen_response(prompt, image_path, model, processor)
                print(f'{response}')
                writer.writerow([coco_index, response])
                file.flush()



# In[ ]:





# In[26]:

model_path = config.QWEN_72B_MODEL
print("Loading Qwen2.5-VL model and processor...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
print("Model loaded successfully!")


print(model.config._attn_implementation_internal)


# In[27]:


csv_file_path = config.TRAINING_CSV
source_path = config.TRAINING_IMAGES_WITH_BBOX
process_csv(csv_file_path, model, processor, source_path)


# In[ ]:




