import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process images in batches')
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2], default=0,
                      help='GPU ID to use')
    parser.add_argument('--batch', type=int, choices=[1, 2, 3], required=True,
                      help='Batch number to process (1 for 0-2500, 2 for 2500-5000)')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pandas as pd
from tqdm import tqdm
import csv
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig



def find_objects_by_coco_index(df, coco_index: int):
    try:
        row = df[df['coco_index'] == coco_index]
        
        if len(row) == 0:
            print(f"COCO index {coco_index} is not found")
            return None
        
        objects_str = row.iloc[0]['objects']
        objects_list = objects_str.split(',') if isinstance(objects_str, str) else []
        
        return objects_list
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def generate_prompt(object_list):
    prompt = f"""Given the following list of objects: {object_list}

    For each object in object_list, analyze if it is significantly out-of-context based on these two criteria:

    1. Location: The spatial location of the object is clearly unreasonable (e.g., floating in mid-air, underwater objects on land)
    2. Size: The object's size is obviously unrealistic (e.g., a mouse larger than a house, a car smaller than a shoe)

    Only analyze objects from {object_list}. For each object that shows obvious out-of-context characteristics, provide your analysis in this format:

    Object: [object name from object_list]
    Analysis:
    - Location: (Explain if the location is clearly unreasonable)
    - Size: (Explain if the size is obviously unrealistic)

    Note: 
    - Only analyze objects that appear in object_list
    - Only report objects with severe and obvious context violations
    - Minor size or location discrepancies should be ignored
    - If unsure, consider the object as in-context

    Final decision: List only the objects from object_list that have obvious and significant context violations. If no objects are clearly out-of-context, respond with "None".

    Example output:
    Object: car
    Analysis:
    - Location: The car is floating in the mid air, which is clearly impossible
    - Size: Size appears normal

    Object: elephant
    Analysis:
    - Location: The elephant appears in the playground, which is unusual
    - Size: The elephant is extremely tiny, only about the size of a house cat, which is completely unrealistic

    Object: handbag
    Analysis:
    - Location: The handbag is placed on a table in the room, which is completely normal
    - Size: The handbag appears to be normal size for a typical handbag

    Final decision: car, elephant

    Your answer is:
    """
    return prompt

def molmo_response(prompt, image_path, model, processor):
    model.to(dtype=torch.bfloat16)
    try:
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
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return f"Error: {str(e)}"

def main():
    
    # Load model, use you own path
    model_path = '/data2/ty45972_data2/weights/Molmo-7B-D-0924'
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        # torch_dtype='auto',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        # torch_dtype='auto',
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    print(model.dtype)
    # Read CSV file
    csv_path = "../task_data/fake_localization/inpainting_dataset_objects.csv"
    df = pd.read_csv(csv_path)
    
    # Determine batch range
    if args.batch == 1:
        df_batch = df[df['coco_index'] <= 2500]
    elif args.batch == 2:
        df_batch = df[df['coco_index'] > 2500]
    
    # Prepare output file
    output_file = f'../task_data/fake_localization/context_objects_prediction/context_results_batch_{args.batch}.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['coco_index', 'response'])
        
        # Process each image in the batch
        for _, row in tqdm(df_batch.iterrows(), total=len(df_batch)):
            coco_index = row['coco_index']
            image_path = f"../task_data/images/testing_images/{coco_index}.png"
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                print(f"Skipping {coco_index}: Image not found")
                continue
                
            objects = find_objects_by_coco_index(df, coco_index)
            if objects:
                prompt = generate_prompt(objects)
                response = molmo_response(prompt, image_path, model, processor)
                writer.writerow([coco_index, response])
                f.flush() 

if __name__ == "__main__":
    main()