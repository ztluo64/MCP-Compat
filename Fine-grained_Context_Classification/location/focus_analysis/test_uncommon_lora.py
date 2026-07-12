#!/usr/bin/env python
# coding: utf-8
"""
测试 Qwen2.5-VL-3B (LoRA Finetuned) 在所有 uncommon 图片上的表现
- uncommon_images: ground truth = Out-of-context
- 统计 category-location pair 的预测分布
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

import re
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'model_id': config.QWEN_3B_MODEL,
    'lora_path': config.LOCATION_FINETUNED_MODEL,
    'data_dir': config.FOCUS_DATA_DIR,
    'max_length': 2048,
    'use_4bit': True,
    'seed': 42,
}

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# ============================================================================
# Prompt Generator (无红框版本，提供location hint)
# ============================================================================

def generate_location_prompt(category, location):
    """生成prompt，提供category和location作为hint，无红框描述"""
    prompt = f"""The image shows a {category} in a {location} environment.
Considering image quality is not a factor, do you think this {category} is in-context or out-of-context?

Important: Please assume that the {category} genuinely exists in the scene, regardless of how realistic or natural it appears visually. Focus only on whether its presence makes contextual sense, not on its visual realism.

The criterion for determining whether an object is in- or out-of-context is as follows:
Location: Evaluate whether the {category} is placed in a physically and contextually reasonable position and environment.
- If the {category} is in an unusual or uncommon environment for this type of object, it is considered out-of-context.
- If the {category} is in a typical, common environment where you would expect to find it, it is considered in-context.

Important notes:
- The threshold is "unusual" or "uncommon", NOT "impossible". If something is uncommon or atypical in real-world contexts, it should be considered out-of-context.
- Your summary judgment must be consistent with your analysis.

Please provide the analysis according to the Location criterion, then give a summary, and finally provide your final decision.

For example, your answer could be:
Analysis:
Location: The {category} is in an unusual environment - [explanation].
Summary: [brief summary]
Final decision: Out-of-context.

Or your answer could be:
Analysis:
Location: The {category} is in a typical/common environment - [explanation].
Summary: [brief summary]
Final decision: In-context.

Your answer is:"""
    return prompt

# ============================================================================
# Load Model (with LoRA)
# ============================================================================

print("Loading base model...")
bnb_config = None
if CONFIG['use_4bit']:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CONFIG['model_id'],
    quantization_config=bnb_config,
    # device_map="auto",
    device_map={"": 3},
    local_files_only=True,
    torch_dtype=torch.bfloat16
)

print(f"Loading LoRA adapter from {CONFIG['lora_path']}...")
model = PeftModel.from_pretrained(
    model,
    CONFIG['lora_path'],
    is_trainable=False  # 推理模式
)

print("Merging LoRA weights...")
model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained(CONFIG['model_id'], local_files_only=True)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("✅ LoRA Finetuned Model loaded!")

# ============================================================================
# Load and Sample Data
# ============================================================================

def load_and_sample_data():
    """加载CSV，返回所有uncommon数据（过滤掉snow）"""
    data_dir = CONFIG['data_dir']
    
    # 读取CSV
    uncommon_df = pd.read_csv(os.path.join(data_dir, 'uncommon_images.csv'))
    
    print(f"Original - Uncommon: {len(uncommon_df)}")
    
    # 过滤掉 location 为 snow 的样本
    uncommon_df = uncommon_df[uncommon_df['location'] != 'snow']
    
    print(f"After removing snow - Uncommon: {len(uncommon_df)}")
    
    # 添加标签和图片路径
    uncommon_df = uncommon_df.copy()
    uncommon_df['ground_truth'] = 'Out-of-context'
    uncommon_df['image_path'] = uncommon_df['new_filename'].apply(
        lambda x: os.path.join(data_dir, 'uncommon_images', x)
    )
    
    # shuffle
    uncommon_df = uncommon_df.sample(frac=1, random_state=CONFIG['seed']).reset_index(drop=True)
    
    print(f"✅ Total uncommon samples: {len(uncommon_df)}")
    
    # 打印 category-location pair 分布
    print(f"\n📊 Category-Location distribution:")
    pair_counts = uncommon_df.groupby(['category', 'location']).size().reset_index(name='count')
    for _, row in pair_counts.iterrows():
        print(f"   {row['category']:10s} - {row['location']:10s}: {row['count']}")
    
    return uncommon_df

# ============================================================================
# Extract Label
# ============================================================================

def extract_predicted_label(response):
    """从模型回答中提取预测标签"""
    # 尝试匹配 "Final decision: xxx"
    pattern = r'Final decision:\s*(In-context|Out-of-context)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        label = match.group(1)
        return 'In-context' if 'in' in label.lower() and 'out' not in label.lower() else 'Out-of-context'
    
    # Fallback: 检查最后100个字符
    last_part = response[-150:].lower()
    if "out-of-context" in last_part or "out of context" in last_part:
        return "Out-of-context"
    elif "in-context" in last_part or "in context" in last_part:
        return "In-context"
    
    return "Unknown"

# ============================================================================
# Inference
# ============================================================================

def run_inference(data_df, dataset_name="dataset"):
    """对所有样本进行推理，带时间统计"""
    import time
    
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    results = []
    correct_count = 0
    
    total_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting inference on [{dataset_name}] - {len(data_df)} samples")
    print(f"{'='*60}")
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Inference [{dataset_name}]"):
        sample_start_time = time.time()
        
        image_path = row['image_path']
        category = row['category']
        location = row['location']
        ground_truth = row['ground_truth']
        
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            continue
        
        # 生成prompt
        user_prompt = generate_location_prompt(category, location)
        
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
            images=[image],
            return_tensors="pt",
            padding=True,
            max_length=CONFIG['max_length']
        )
        
        inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
                 if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
        
        response = processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        prediction = extract_predicted_label(response)
        is_correct = prediction == ground_truth
        if is_correct:
            correct_count += 1
        
        sample_time = time.time() - sample_start_time
        
        results.append({
            'filename': row['new_filename'],
            'category': category,
            'location': location,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': is_correct,
            'response': response,
            'inference_time': sample_time
        })
        
        # 每个样本都打印完整结果
        sample_num = len(results)
        elapsed = time.time() - total_start_time
        avg_time = elapsed / sample_num
        eta = avg_time * (len(data_df) - sample_num)
        
        status = "✓" if is_correct else "✗"
        running_acc = correct_count / sample_num
        
        print(f"\n{'='*70}")
        print(f"[{sample_num}/{len(data_df)}] {status} | Acc: {running_acc:.2%} | Time: {sample_time:.2f}s | ETA: {eta/60:.1f}min")
        print(f"File: {row['new_filename']}")
        print(f"Category: {category} | Location: {location}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction: {prediction}")
        print(f"\n--- Full Response ---")
        print(response)
        print(f"{'='*70}")
    
    # 最终统计
    total_time = time.time() - total_start_time
    avg_time_per_sample = total_time / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"✅ [{dataset_name}] Inference complete!")
    print(f"  Total: {len(results)} | Correct: {correct_count} | Accuracy: {correct_count/len(results):.2%}")
    print(f"  ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  ⏱️  Avg per sample: {avg_time_per_sample:.2f}s")
    print(f"{'='*60}")
    
    return results, {
        'total_time': total_time,
        'avg_time_per_sample': avg_time_per_sample,
        'num_samples': len(results)
    }

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_results(results):
    """计算评估指标"""
    df = pd.DataFrame(results)
    
    # 过滤掉Unknown
    valid_df = df[df['prediction'] != 'Unknown']
    unknown_count = len(df) - len(valid_df)
    
    if unknown_count > 0:
        print(f"⚠️ {unknown_count} samples with Unknown predictions")
    
    y_true = valid_df['ground_truth'].tolist()
    y_pred = valid_df['prediction'].tolist()
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', 
                                                                pos_label='In-context')
    
    # 混淆矩阵
    labels = ['In-context', 'Out-of-context']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Valid predictions: {len(valid_df)}")
    print(f"Unknown predictions: {unknown_count}")
    
    print(f"\n🎯 Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 In-ctx  Out-ctx")
    print(f"  Actual In-ctx   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Actual Out-ctx  {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    print(f"\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels))
    
    # 按类别统计
    print("\n📊 Per-category accuracy:")
    for cat in df['category'].unique():
        cat_df = valid_df[valid_df['category'] == cat]
        if len(cat_df) > 0:
            cat_acc = cat_df['correct'].mean()
            print(f"  {cat:10s}: {cat_acc:.2%} ({cat_df['correct'].sum()}/{len(cat_df)})")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'total': len(df),
        'valid': len(valid_df),
        'unknown': unknown_count
    }
    
    return metrics

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import time
    
    overall_start = time.time()
    
    # 加载所有 uncommon 数据
    uncommon_df = load_and_sample_data()
    
    output_dir = CONFIG['data_dir']
    
    # ========== 运行所有 uncommon (Out-of-context) ==========
    print("\n" + "="*80)
    print("📍 Running ALL UNCOMMON images (Ground Truth: Out-of-context)")
    print("="*80)
    
    uncommon_results, uncommon_time_stats = run_inference(uncommon_df, dataset_name="UNCOMMON")
    uncommon_metrics = evaluate_results(uncommon_results)
    uncommon_metrics['time_stats'] = uncommon_time_stats
    
    # 保存完整结果
    uncommon_results_df = pd.DataFrame(uncommon_results)
    uncommon_results_df.to_csv(os.path.join(output_dir, 'uncommon_all_inference_results_lora.csv'), index=False)
    with open(os.path.join(output_dir, 'uncommon_all_inference_metrics_lora.json'), 'w') as f:
        json.dump(uncommon_metrics, f, indent=2)
    
    # ========== 统计 category-location pair 的预测分布 ==========
    print("\n" + "="*80)
    print("📊 CATEGORY-LOCATION PAIR STATISTICS")
    print("="*80)
    
    # 创建统计 DataFrame
    stats_df = uncommon_results_df.groupby(['category', 'location']).agg(
        total=('prediction', 'count'),
        pred_out_of_context=('prediction', lambda x: (x == 'Out-of-context').sum()),
        pred_in_context=('prediction', lambda x: (x == 'In-context').sum()),
        pred_unknown=('prediction', lambda x: (x == 'Unknown').sum()),
        correct=('correct', 'sum')
    ).reset_index()
    
    # 计算比例
    stats_df['out_of_context_rate'] = stats_df['pred_out_of_context'] / stats_df['total']
    stats_df['accuracy'] = stats_df['correct'] / stats_df['total']
    
    # 按 out_of_context_rate 降序排列
    stats_df = stats_df.sort_values('out_of_context_rate', ascending=False)
    
    print("\n🔴 Pairs most likely predicted as OUT-OF-CONTEXT (descending):")
    print("-" * 90)
    print(f"{'Category':<12} {'Location':<12} {'Total':>8} {'Pred OOC':>10} {'Pred IC':>10} {'OOC Rate':>10} {'Accuracy':>10}")
    print("-" * 90)
    for _, row in stats_df.iterrows():
        print(f"{row['category']:<12} {row['location']:<12} {row['total']:>8} {row['pred_out_of_context']:>10} {row['pred_in_context']:>10} {row['out_of_context_rate']:>10.2%} {row['accuracy']:>10.2%}")
    print("-" * 90)
    
    # 保存统计结果
    stats_df.to_csv(os.path.join(output_dir, 'uncommon_pair_statistics_lora.csv'), index=False)
    
    # ========== 总结 ==========
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY (LoRA Finetuned Model)")
    print("="*80)
    
    print(f"\n🔴 UNCOMMON (Out-of-context):")
    print(f"   Total samples: {len(uncommon_results)}")
    print(f"   Accuracy: {uncommon_metrics['accuracy']:.2%}")
    print(f"   Time: {uncommon_time_stats['total_time']:.1f}s ({uncommon_time_stats['total_time']/60:.1f} min)")
    
    print(f"\n📈 Top 5 pairs MOST likely to be predicted as Out-of-context:")
    for i, (_, row) in enumerate(stats_df.head(5).iterrows()):
        print(f"   {i+1}. {row['category']}-{row['location']}: {row['out_of_context_rate']:.2%} ({row['pred_out_of_context']}/{row['total']})")
    
    print(f"\n📉 Top 5 pairs LEAST likely to be predicted as Out-of-context:")
    for i, (_, row) in enumerate(stats_df.tail(5).iloc[::-1].iterrows()):
        print(f"   {i+1}. {row['category']}-{row['location']}: {row['out_of_context_rate']:.2%} ({row['pred_out_of_context']}/{row['total']})")
    
    print(f"\n⏱️  Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
    print(f"   Avg per sample: {overall_time/len(uncommon_results):.2f}s")
    
    print(f"\n✅ All results saved to {output_dir}")
    print(f"   - uncommon_all_inference_results_lora.csv (完整预测结果)")
    print(f"   - uncommon_all_inference_metrics_lora.json (评估指标)")
    print(f"   - uncommon_pair_statistics_lora.csv (category-location pair 统计)")