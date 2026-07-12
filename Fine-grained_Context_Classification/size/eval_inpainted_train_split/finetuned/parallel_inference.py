#!/usr/bin/env python3
"""
Multi-GPU Parallel Inference for Qwen2.5-VL-3B Size Context Reasoning
Runs separate model instances on each GPU with batch_size=1 for stability
"""

import torch
import torch.multiprocessing as mp
import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import webdataset as wds
from datasets import Dataset, load_from_disk
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
)
from multiprocessing import Queue, Process
import threading
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
import config

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'model_id': config.QWEN_3B_MODEL,
    'output_dir': './qwen25_finetuned_size_specialist',
    'max_length': 2048,
    'use_4bit': True,
    'use_lora': True,
    'num_gpus': 4,  # Number of GPUs to use
}

# ============================================================================
# Helper Functions
# ============================================================================

def generate_size_prompt(new_object):
    """Generate a prompt specifically for size analysis"""
    prompt = f"""The object to analyze is {new_object}, which is located inside the red bounding box in the image.
    Considering image quality is not a factor, do you think the object {new_object} inside the red bounding box is in-context or out-of-context?
    Important: Please assume that the object {new_object} inside the red bounding box genuinely exists in the scene, regardless of how realistic or natural it appears visually. Even if the object looks artificially added, poorly rendered, or unrealistic in terms of visual quality, treat it as if it is truly present in that location. Focus only on whether its presence makes contextual sense, not on its visual realism.
    The criterion for determining whether an object is in- or out-of-context is as follows:
    Size: Judge whether the object's size inside the red bounding box is reasonable relative to other objects and the environment in this image.
       For example, a banana larger than a chair or a cat larger than a car would be unusual and out-of-context.
       Important: Always consider the object's real-world size, not as a toy or miniature version. For instance, a car should be evaluated as a full-sized vehicle, not as a toy car.
    Important notes:
    - The threshold is "unusual" or "uncommon", NOT "impossible". If something is uncommon or atypical in real-world contexts, it should be considered out-of-context.
    - Your summary judgment must be consistent with your analysis. If you identify something as unusual, uncommon, or abnormal in your analysis, then your summary MUST be marked as out-of-context. Conversely, if your analysis describes the situation as normal, common, or typical, then mark it as in-context.
    - Analyze only the object {new_object} inside the red bounding box.
    - Do not reinterpret or substitute the given object with another category. Even if the object visually resembles another type, always analyze it strictly as the given object {new_object}.
    - Ignore the visual quality, realism, or rendering quality of the object. Focus solely on whether its contextual presence makes sense.
    Please provide the analysis of the object according to the criterion, then give a summary, and finally provide your final decision.
    For example, your answer could be:
    Analysis:
    Size: The size of this {new_object} looks abnormally large/small compared to [other objects in the scene].
    Summary:
    Final decision: Out-of-context.
    Or your answer could be:
    Analysis:
    Size: The {new_object} looks proportionally normal in this environment.
    Summary:
    Final decision: In-context.
    Your answer is:"""
    return prompt

def load_object_info():
    """Load the object information from CSV"""
    csv_path = config.TRAINING_CSV
    df = pd.read_csv(csv_path)
    object_map = dict(zip(df['coco_index'], df['replacement_object']))
    return object_map

def extract_predicted_label(predicted_response):
    """Extract the Size label from the Summary section"""
    
    # Try to find "Summary:" section and extract Size conclusion
    summary_pattern = r'Summary:.*?Size:\s*(In-context|Out-of-context)'
    match = re.search(summary_pattern, predicted_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Fallback 1: Look for "Size Conclusion:" 
    conclusion_pattern = r'Size Conclusion:\s*(In-context|Out-of-context)'
    match = re.search(conclusion_pattern, predicted_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Fallback 2: Look at the very end of the response (last 100 chars)
    last_part = predicted_response[-100:].lower() if len(predicted_response) > 100 else predicted_response.lower()
    if "out-of-context" in last_part or "out of context" in last_part:
        return "Out-of-context"
    elif "in-context" in last_part or "in context" in last_part:
        return "In-context"
    
    return "Unknown"

# ============================================================================
# Model Loading Function
# ============================================================================

def load_model_on_gpu(gpu_id, model_path, use_4bit=True, use_lora=True):
    """Load a model instance on a specific GPU"""
    
    # Set the specific GPU
    torch.cuda.set_device(gpu_id)
    
    # Quantization config
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load model on specific GPU
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={'': gpu_id},  # Force to specific GPU
        local_files_only=True,
        torch_dtype=torch.bfloat16  # 或者改成 dtype=torch.bfloat16 避免 warning
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
         model,
        config.SIZE_FINETUNED_MODEL,
        is_trainable=False  # 推理模式
    ) 
    
    model = model.merge_and_unload()
        
        
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    

    model.eval()
    return model, processor

# ============================================================================
# Worker Process Function
# ============================================================================

def worker_process(gpu_id, input_queue, output_queue, model_path, object_map, config):
    """Worker process that runs on a specific GPU"""
    try:
        # Set GPU for this process
        torch.cuda.set_device(gpu_id)
        print(f"Worker {gpu_id}: Loading model on GPU {gpu_id}...")
        
        # Load model on this GPU
        model, processor = load_model_on_gpu(
            gpu_id, 
            model_path, 
            use_4bit=config['use_4bit'],
            use_lora=config['use_lora']
        )
        print(f"Worker {gpu_id}: Model loaded successfully!")
        
        # Process samples
        while True:
            item = input_queue.get()
            
            if item is None:  # Shutdown signal
                print(f"Worker {gpu_id}: Shutting down...")
                break
            
            sample_idx, sample, coco_index = item
            
            try:
                # Get object name
                object_name = object_map.get(coco_index, "object")
                
                # Generate prompt
                user_prompt = generate_size_prompt(object_name)
                
                messages = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
                
                # Process
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = processor(
                    text=[text],
                    images=[sample['image_data']],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=config['max_length']
                ).to(f'cuda:{gpu_id}')
                
                # Generate
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )
                
                # Decode
                predicted_response = processor.batch_decode(
                    outputs[:, inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )[0]
                
                # Extract label
                predicted_label = extract_predicted_label(predicted_response)
                
                result = {
                    'idx': sample_idx,
                    'predicted_label': predicted_label,
                    'predicted_response': predicted_response,
                    'gt_label': sample['label'],
                    'gt_response': sample['ground_truth_response']
                }
                
                output_queue.put(result)
                
            except Exception as e:
                print(f"Worker {gpu_id}: Error processing sample {sample_idx}: {e}")
                output_queue.put({
                    'idx': sample_idx,
                    'predicted_label': 'Unknown',
                    'predicted_response': f'Error: {str(e)}',
                    'gt_label': sample['label'],
                    'gt_response': sample['ground_truth_response']
                })
    
    except Exception as e:
        print(f"Worker {gpu_id} fatal error: {e}")

# ============================================================================
# Main Parallel Inference Function
# ============================================================================

def parallel_inference_multiprocess(eval_dataset, object_map, model_path, config, num_samples=None):
    """Run parallel inference using multiple GPUs with separate processes"""
    
    num_gpus = config['num_gpus']
    
    if num_samples is None:
        num_samples = len(eval_dataset)
    else:
        num_samples = min(num_samples, len(eval_dataset))
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting parallel inference on {num_gpus} GPUs")
    print(f"📊 Processing {num_samples} samples")
    print(f"{'='*60}")
    
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create queues
    input_queue = mp.Queue(maxsize=num_gpus * 2)
    output_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, input_queue, output_queue, model_path, object_map, config)
        )
        p.start()
        processes.append(p)
        time.sleep(2)  # Stagger process starts to avoid memory issues
    
    print(f"✅ All {num_gpus} workers started!")
    
    # Producer thread to feed samples
    def producer():
        for i in range(num_samples):
            sample = eval_dataset[i]
            input_queue.put((i, sample, sample['coco_index']))
        
        # Send shutdown signals
        for _ in range(num_gpus):
            input_queue.put(None)
    
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    
    # Collect results
    results = {}
    pbar = tqdm(total=num_samples, desc="Processing samples")
    
    while len(results) < num_samples:
        try:
            result = output_queue.get(timeout=1)
            results[result['idx']] = result
            pbar.update(1)
        except:
            continue
    
    pbar.close()
    
    # Wait for processes to finish
    producer_thread.join()
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    
    # Sort results by index
    sorted_results = [results[i] for i in range(num_samples)]
    
    # Extract outputs
    predicted_labels = [r['predicted_label'] for r in sorted_results]
    predicted_responses = [r['predicted_response'] for r in sorted_results]
    gt_labels = [r['gt_label'] for r in sorted_results]
    gt_responses = [r['gt_response'] for r in sorted_results]
    
    return predicted_labels, gt_labels, predicted_responses, gt_responses

# ============================================================================
# Results Analysis Functions
# ============================================================================

def print_results(predicted_labels, gt_labels):
    """Print detailed results analysis"""
    
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Basic metrics
    accuracy = accuracy_score(gt_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gt_labels, predicted_labels, 
        labels=['In-context', 'Out-of-context'],
        average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gt_labels, predicted_labels, average='macro', zero_division=0
    )
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n📊 Macro Averages:")
    print(f"   Precision: {precision_macro:.4f}")
    print(f"   Recall:    {recall_macro:.4f}")
    print(f"   F1-Score:  {f1_macro:.4f}")
    
    print(f"\n📋 Per-Class Performance:")
    for i, label in enumerate(['In-context', 'Out-of-context']):
        print(f"\n   {label}:")
        print(f"     Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1: {f1[i]:.4f}")
        print(f"     Support: {support[i]} samples")
    
    # Confusion Matrix
    cm = confusion_matrix(gt_labels, predicted_labels, 
                          labels=['In-context', 'Out-of-context'])
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    print("\n📊 Confusion Matrix:")
    print("\n                    Predicted")
    print("                 In-context  Out-of-context")
    print(f"Actual In-context     {cm[0,0]:4d}          {cm[0,1]:4d}    ({cm_percent[0,0]:5.1f}%  {cm_percent[0,1]:5.1f}%)")
    print(f"       Out-of-context {cm[1,0]:4d}          {cm[1,1]:4d}    ({cm_percent[1,0]:5.1f}%  {cm_percent[1,1]:5.1f}%)")
    
    # Error analysis
    incorrect_indices = [i for i in range(len(predicted_labels)) 
                        if predicted_labels[i] != gt_labels[i]]
    num_errors = len(incorrect_indices)
    
    false_positives = sum(1 for i in incorrect_indices 
                         if gt_labels[i] == 'In-context' and predicted_labels[i] == 'Out-of-context')
    false_negatives = sum(1 for i in incorrect_indices 
                         if gt_labels[i] == 'Out-of-context' and predicted_labels[i] == 'In-context')
    unknown_count = sum(1 for label in predicted_labels if label == 'Unknown')
    
    print(f"\n❌ Error Analysis:")
    print(f"   Total Errors: {num_errors}/{len(predicted_labels)} ({num_errors/len(predicted_labels)*100:.2f}%)")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   Unknown predictions: {unknown_count}")
    
    # Model bias assessment
    if abs(false_positives - false_negatives) < 5:
        bias = "BALANCED ✓"
    elif false_negatives > false_positives:
        bias = "BIASED towards In-context ⚠️"
    else:
        bias = "BIASED towards Out-of-context ⚠️"
    print(f"\n⚖️  Model Bias: {bias}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run the parallel inference"""
    
    print("\n" + "="*60)
    print("QWEN3-VL-4B MULTI-GPU PARALLEL INFERENCE")
    print("="*60)
    
    # Load cached dataset
    cache_dir = '../../cached_dataset'
    if os.path.exists(cache_dir):
        print("\n📦 Loading cached dataset...")
        eval_dataset = load_from_disk(os.path.join(cache_dir, 'test'))
        print(f"✅ Loaded {len(eval_dataset)} evaluation samples")
    else:
        print("❌ Error: Cached dataset not found! Please run the training notebook first.")
        return
    
    # Load object mapping
    print("\n📚 Loading object mapping...")
    object_map = load_object_info()
    print(f"✅ Loaded {len(object_map)} object mappings")
    
    # Run parallel inference
    print("\n" + "="*60)
    print("STARTING PARALLEL INFERENCE")
    print("="*60)
    
    start_time = time.time()
    
    predicted_labels, gt_labels, predicted_responses, gt_responses = parallel_inference_multiprocess(
        eval_dataset=eval_dataset,
        object_map=object_map,
        model_path=CONFIG['model_id'],
        config=CONFIG,
        num_samples=len(eval_dataset)  # Process full eval set
    )
    
    elapsed_time = time.time() - start_time
    samples_per_second = len(gt_labels) / elapsed_time
    
    print(f"\n✅ Processing complete!")
    print(f"⏱️  Time: {elapsed_time:.2f} seconds")
    print(f"🚀 Speed: {samples_per_second:.2f} samples/second")
    
    # Print results
    print_results(predicted_labels, gt_labels)
    
    # Save results
    print("\n💾 Saving results...")
    results_df = pd.DataFrame({
        'predicted_label': predicted_labels,
        'ground_truth_label': gt_labels,
        'predicted_response': predicted_responses,
        'ground_truth_response': gt_responses
    })
    results_df.to_csv('parallel_inference_results.csv', index=False)
    print("✅ Results saved to parallel_inference_results.csv")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()