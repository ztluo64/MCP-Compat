import sys
sys.executable
import os
os.getcwd()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
from tqdm import tqdm
import csv
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


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

    
def molmo_response(prompt, image_path, model, processor):
    # model.to(dtype=torch.bfloat16)
    inputs = processor.process(
        images=[Image.open(image_path)],
        text=prompt
    )
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

    
def process_csv(csv_file_path, model, processor, source_path=config.TRAINING_IMAGES_WITH_BBOX):
    df = pd.read_csv(csv_file_path)
    output_file = os.path.join(config.CONTEXT_REASONING_OUTPUT_ROOT, "context_reasoning_molmo_part1.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['coco_index', 'response'])
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            coco_index = row['coco_index']
            if coco_index <= 60_000:
                replacement_object = row['replacement_object']
                image_path = os.path.join(source_path, f"{coco_index}.png")
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                try:
                    prompt = generate_prompt(replacement_object)
                    response = molmo_response(prompt, image_path, model, processor)
                    print(f'{response}')
                    writer.writerow([coco_index, response])
                    file.flush()
                except Exception as e:
                    print(f"Error processing {coco_index}: {str(e)}")
                    continue
            # if coco_index > 1011:
            #     break

            
# Load Molmo model
model_path = config.MOLMO_72B_MODEL
print("Loading Molmo model and processor...")
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
print("Model loaded successfully!")


# Process CSV
csv_file_path = config.TRAINING_CSV
source_path = config.TRAINING_IMAGES_WITH_BBOX
process_csv(csv_file_path, model, processor, source_path)














