#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Qwen2.5-VL-3B for Context Reasoning (Location)
# 
# This notebook fine-tunes Qwen2.5-VL-3B-Instruct on your WebDataset for context reasoning based on location analysis.

# In[1]:


import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config
sys.executable


# In[2]:


# # Install required packages
# !pip install -q transformers accelerate peft bitsandbytes
# !pip install -q webdataset pillow
# !pip install -q trl


# In[3]:


from pycocotools.coco import COCO
import requests
from io import BytesIO


# In[4]:


import torch
import torchvision

# In[5]:


import os
import re
import glob
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import webdataset as wds
import matplotlib.pyplot as plt


# In[6]:


from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL 使用这个
    BitsAndBytesConfig,
    TrainingArguments
)

from trl import SFTTrainer, SFTConfig
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# In[7]:


from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


# In[8]:


from peft import PeftModel


# # ============================================================================
# # Configuration
# # ============================================================================

# In[9]:


fintune_path = config.CO_OCCURRENCE_FINETUNED_MODEL

# In[10]:


# Get the actual tar files
tar_files = sorted(glob.glob(os.path.join(config.CO_OCCURRENCE_WEBDATASET_DIR, 'train-*.tar')))
print(f"Found {len(tar_files)} tar files:")
for f in tar_files:
    print(f"  - {f}")

CONFIG = {
    'model_id': config.QWEN_3B_MODEL,
    'webdataset_files': tar_files,
    'output_dir': './qwen25_finetuned_co_occurrence_specialist',
    'batch_size': 1,
    'gradient_accumulation_steps': 16,
    'num_epochs': 3,
    'learning_rate': 2e-4,
    'max_length': 2048,  # Increased for full responses
    'warmup_ratio': 0.1,
    'logging_steps': 10,
    'save_steps': 500,
    'eval_steps': 100,
    'use_4bit': True,
    'use_lora': True,
    'dataset_size': 5000,
}

print("\nConfiguration:")
for k, v in CONFIG.items():
    if k != 'webdataset_files':
        print(f"  {k}: {v}")


# # ============================================================================
# # co_occurrence-specific Prompt Generator
# # ============================================================================
# 

# In[11]:

def generate_co_occurrence_prompt(new_object):
    prompt = f"""The object to analyze is {new_object}, which is located inside the red bounding box in the image.
    Considering image quality is not a factor, do you think the object {new_object} inside the red bounding box is in-context or out-of-context?
    Important: Please assume that the object {new_object} inside the red bounding box genuinely exists in the scene, regardless of how realistic or natural it appears visually. Even if the object looks artificially added, poorly rendered, or unrealistic in terms of visual quality, treat it as if it is truly present in that location. Focus only on whether its presence makes contextual sense, not on its visual realism.
    The criterion for determining whether an object is in- or out-of-context is as follows:
    Co-occurrence: Determine whether the object inside the red bounding box can reasonably appear together with other objects in the scene.
       If the combination of objects is unusual or uncommon in real-world contexts, it is considered out-of-context.
    Important notes:
    - The threshold is "unusual" or "uncommon", NOT "impossible". If something is uncommon or atypical in real-world contexts, it should be considered out-of-context.
    - Your summary judgment must be consistent with your analysis. If you identify something as unusual, uncommon, or abnormal in your analysis, then your summary MUST be marked as out-of-context. Conversely, if your analysis describes the situation as normal, common, or typical, then mark it as in-context.
    - Analyze only the object {new_object} inside the red bounding box.
    - Do not reinterpret or substitute the given object with another category. Even if the object visually resembles another type, always analyze it strictly as the given object {new_object}.
    - Ignore the visual quality, realism, or rendering quality of the object. Focus solely on whether its contextual presence makes sense.
    Please provide the analysis of the object according to the Co-occurrence criterion, then give a summary, and finally provide your final decision.
    For example, your answer could be:
    Analysis:
    Co-occurrence: It is unusual for a {new_object} to appear together with [other objects in the scene].
    Summary:
    Final decision: Out-of-context.
    Or your answer could be:
    Analysis:
    Co-occurrence: A {new_object} commonly appears together with [other objects in the scene].
    Summary:
    Final decision: In-context.
    Your answer is:"""
    return prompt

# # ============================================================================
# # Load Model and Processor from Local Path
# # ============================================================================
# 

# In[12]:


# Quantization config
bnb_config = None
if CONFIG['use_4bit']:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


# In[13]:


# 改用 Qwen2VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONFIG['model_id'],
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    torch_dtype=torch.bfloat16  # 或者改成 dtype=torch.bfloat16 避免 warning
)


# In[14]:


# # 加载LoRA适配器
# model = PeftModel.from_pretrained(
#      model,
#     fintune_path, 
#     is_trainable=False  # 推理模式
# ) 


# # In[15]:


# model = model.merge_and_unload()


# In[16]:


processor = AutoProcessor.from_pretrained(
    CONFIG['model_id'],
    # fintune_path,
    local_files_only=True
)

# Set padding token if not set
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token


# # ============================================================================
# # Load Data and Extract co_occurrence-related Parts
# # ============================================================================
# 

# In[17]:


def load_object_info():
    """Load the object information from CSV"""
    csv_path = config.TRAINING_CSV

    df = pd.read_csv(csv_path)
    # Create a dictionary mapping coco_index to replacement_object
    object_map = dict(zip(df['coco_index'], df['replacement_object']))
    return object_map


# In[18]:


# Load object mapping
object_map = load_object_info()


# # ============================================================================
# # Training Arguments
# # ============================================================================
# 

# In[19]:


sft_config = SFTConfig(
    # Standard training parameters
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    learning_rate=CONFIG['learning_rate'],
    warmup_ratio=CONFIG['warmup_ratio'],
    logging_steps=CONFIG['logging_steps'],
    save_steps=CONFIG['save_steps'],
    eval_steps=CONFIG['eval_steps'],
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    bf16=True if torch.cuda.is_available() else False,
    gradient_checkpointing=True if CONFIG['use_4bit'] else False,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    optim="paged_adamw_32bit" if CONFIG['use_4bit'] else "adamw_torch",
    
    # SFT-specific parameters with CORRECTED names
    dataset_text_field="text",
    max_length=CONFIG['max_length'],  # Changed from max_seq_length to max_length
    packing=False,
)


# # ============================================================================
# # Test Inference
# # ============================================================================

# In[20]:


def extract_predicted_label(predicted_response):
    """Extract the Co-occurrence label from the Summary section"""
    
    # Try to find "Summary:" section and extract Co-occurrence conclusion
    summary_pattern = r'Summary:.*?Co-occurrence:\s*(In-context|Out-of-context)'
    match = re.search(summary_pattern, predicted_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Fallback 1: Look for "Co-occurrence Conclusion:" 
    conclusion_pattern = r'Co-occurrence Conclusion:\s*(In-context|Out-of-context)'
    match = re.search(conclusion_pattern, predicted_response, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    # Fallback 2: Look at the very end of the response (last 100 chars)
    last_part = predicted_response[-100:].lower()
    if "out-of-context" in last_part or "out of context" in last_part:
        return "Out-of-context"
    elif "in-context" in last_part or "in context" in last_part:
        return "In-context"
    
    return "Unknown"


# In[21]:


def load_test_dataset():
    """加载测试数据集"""
    cache_dir = '../cached_dataset'
    test_dataset_path = os.path.join(cache_dir, 'test')
    
    if os.path.exists(test_dataset_path):
        print(f"Loading test dataset from {test_dataset_path}")
        test_dataset = load_from_disk(test_dataset_path)
        print(f"✅ Loaded {len(test_dataset)} test samples")
        return test_dataset
    else:
        print(f"❌ Test dataset not found at {test_dataset_path}")
        print("Available directories:")
        if os.path.exists(cache_dir):
            print(os.listdir(cache_dir))
        return None

# 加载测试数据集
test_dataset = load_test_dataset()

if test_dataset is None:
    raise ValueError("Cannot proceed without test dataset!")

# 查看数据集结构
print("\n📊 Test dataset info:")
print(f"  Keys: {test_dataset.features.keys()}")
print(f"\nSample entry:")
print(test_dataset[0])


# In[22]:


def print_final_report(results):
    """打印包含 Accuracy, Precision, Recall, F1 的最终报告"""
    
    print(f"\n{'='*80}")
    print("最终报告")
    print(f"{'='*80}")
    
    if not results:
        print("\n⚠️ 没有测试结果")
        return
    
    # 计算混淆矩阵
    # 对于这个测试：所有样本都应该是 In-context
    # TP: 正确预测为 In-context
    # FN: 错误预测为 Out-of-context (shortcut!)
    
    tp = sum(1 for r in results if r['correct'] and r['prediction'] == 'In-context')
    fn = sum(1 for r in results if not r['correct'] and r['prediction'] == 'Out-of-context')
    fp = sum(1 for r in results if not r['correct'] and r['prediction'] == 'In-context')  # 理论上=0
    tn = 0  # 没有真正的 Out-of-context 样本
    
    total = len(results)
    
    # 计算指标
    accuracy = tp / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印结果
    print(f"\n📊 测试统计:")
    print(f"  总样本数: {total}")
    print(f"  正确 (True Positive):  {tp}")
    print(f"  错误 (False Negative): {fn}")
    
    print(f"\n🎯 评估指标:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # 判断
    print(f"\n{'='*40}")
    if accuracy >= 0.95:
        print("✅ 结论: 模型鲁棒，无明显捷径学习")
        print("   模型能够正确识别原始图片中的对象为符合上下文")
    elif accuracy >= 0.85:
        print("⚠️ 结论: 存在轻微捷径学习")
        print("   模型有时会将原始对象误判为不符合上下文")
    elif accuracy >= 0.70:
        print("⚠️ 结论: 存在中等程度捷径学习")
        print("   模型较频繁地依赖inpainting伪影而非真实上下文")
    else:
        print("🚨 结论: 严重捷径学习，模型受损")
        print("   模型主要依赖视觉伪影判断，而非语义上下文")
    
    # 显示失败案例
    failures = [r for r in results if not r['correct']]
    if failures:
        print(f"\n❌ 失败案例 ({len(failures)} 个):")
        for f in failures:
            print(f"  - {f['object']:15s} (Index: {f['dataset_idx']:5d}, ID: {f['image_id']:6d})")
    else:
        print(f"\n✅ 无失败案例！所有测试样本均正确识别")
    
    print(f"{'='*40}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
        'tp': tp,
        'fn': fn
    }




def test_original_images_with_metrics(model, processor, test_dataset, coco_data, coco, num_samples=20):
    """
    测试函数：包含详细的评估指标
    """
    
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    print("\n" + "="*80)
    print("测试原始COCO图片 - 捷径学习检测")
    print("="*80)
    
    # 加载CSV
    csv_path = config.TRAINING_CSV
    df = pd.read_csv(csv_path)
    
    # 类别映射
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    results = []
    tested = 0
    
    print(f"\n开始测试最多 {num_samples} 个样本...")
    
    for i in tqdm(range(len(test_dataset)), desc="Testing"):
        if tested >= num_samples:
            break
        
        sample = test_dataset[i]
        dataset_idx = sample['coco_index']
        
        # 从CSV获取对象信息
        matching_rows = df[df['coco_index'] == dataset_idx]
        if matching_rows.empty:
            continue
        
        row = matching_rows.iloc[0]
        object_index = int(row['object_index'])
        
        try:
            # 动态获取 image_id
            _, annotations = coco_data[dataset_idx]
            
            if len(annotations) == 0:
                continue
            
            image_id = annotations[0]['image_id']
            
            # 获取图片
            img_info = coco.loadImgs(image_id)[0]
            img_path = config.COCO_TRAIN_IMAGES
            img_file = os.path.join(img_path, img_info['file_name'])
            
            if not os.path.exists(img_file):
                continue
            
            image = Image.open(img_file).convert('RGB')
            
            # 验证 object_index
            if object_index >= len(annotations):
                continue
            
            # 获取目标对象
            target_ann = annotations[object_index]
            actual_object_name = cat_id_to_name[target_ann['category_id']]
            bbox = target_ann['bbox']
            
            # 画红框
            from PIL import ImageDraw
            img_with_box = image.copy()
            draw = ImageDraw.Draw(img_with_box)
            x, y, w, h = bbox
            draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
            
        except Exception as e:
            print(f"\n⚠️ 错误 (idx={dataset_idx}): {e}")
            continue
        
        print(f"\n✅ 测试 #{tested+1}")
        print(f"   Dataset Index: {dataset_idx} -> Image ID: {image_id}")
        print(f"   Object: '{actual_object_name}'")
        
        # 生成prompt
        user_prompt = generate_co_occurrence_prompt(actual_object_name)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=[img_with_box],
            return_tensors="pt",
            padding=True,
            max_length=CONFIG['max_length']
        )
        
        inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        predicted_response = processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        predicted_label = extract_predicted_label(predicted_response)
        expected = "In-context"
        is_correct = predicted_label == expected
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_with_box)
        ax.axis('off')
        
        color = 'green' if is_correct else 'red'
        status = "✓ PASS" if is_correct else "🚨 SHORTCUT!"
        
        ax.set_title(
            f"Original COCO Image #{tested+1}\n"
            f"Dataset Index: {dataset_idx} | Image ID: {image_id}\n"
            f"Object: {actual_object_name} | Prediction: {predicted_label}\n"
            f"{status}",
            fontsize=11, color=color, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"   预测: {predicted_label}")
        
        if not is_correct:
            print(f"\n   🚨 检测到捷径学习！")
            print(f"   模型认为原始'{actual_object_name}'不符合上下文")
            print(f"\n   推理过程:\n{predicted_response[:400]}")
        else:
            print(f"   ✓ 正确识别为符合上下文")
        
        results.append({
            'dataset_idx': dataset_idx,
            'image_id': image_id,
            'object': actual_object_name,
            'prediction': predicted_label,
            'correct': is_correct
        })
        
        tested += 1
    
    # 使用新的报告函数
    metrics = print_final_report(results)
    
    return results, metrics



coco = COCO(config.COCO_TRAIN_ANNOTATIONS)
coco_data = torchvision.datasets.CocoDetection(
    config.COCO_TRAIN_IMAGES + '/',
    config.COCO_TRAIN_ANNOTATIONS
)


results, metrics = test_original_images_with_metrics(
    model,
    processor,
    test_dataset,
    coco_data,
    coco,
    num_samples=len(test_dataset),
)


print(f"\n✅ 测试完成！")
print(f"\n📈 最终指标总结:")
print(f"  Accuracy:  {metrics['accuracy']:.2%}")
print(f"  Precision: {metrics['precision']:.2%}")
print(f"  Recall:    {metrics['recall']:.2%}")
print(f"  F1-Score:  {metrics['f1']:.2%}")