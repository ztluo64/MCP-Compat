#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Qwen2.5-VL-3B for Context Reasoning (Size) - Multi-GPU Version with Accelerate
# 
# This script fine-tunes Qwen2.5-VL-3B-Instruct using Accelerate for multi-GPU training.
# Optimized for 4x H100 GPUs with 80GB memory each.
#
# Enhanced with comprehensive logging system:
# - Separate log files for each GPU rank
# - Detailed training metrics logging
# - Progress tracking and ETA estimation
# - Error handling with full traceback
# - Custom callback for step-by-step monitoring

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import config
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
import torch
import time
import logging
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,  # Qwen2.5-VL 使用这个
    BitsAndBytesConfig,
    TrainingArguments
)

from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================================================
# MULTI-GPU SPECIFIC: Environment Setup
# ============================================================================
# Prevent tokenizer parallelism issues in multi-process training
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Increase timeout for distributed training
os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour timeout instead of 30 minutes

# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logging(output_dir='./qwen25_finetuned_co_occurrence_specialist_multi_gpu'):
    """
    Setup logging configuration for multi-GPU training.
    Each process gets its own log file, main process also logs to console.
    """
    # Create logs directory
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get rank for file naming
    rank = get_rank() if torch.distributed.is_initialized() else 0
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set up log file path
    log_file = log_dir / f'training_rank{rank}_{timestamp}.log'
    
    # Configure logging format
    log_format = f'[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s'
    
    # Clear any existing handlers
    logger = logging.getLogger()
    logger.handlers.clear()
    
    # Set logging level
    logger.setLevel(logging.INFO)
    
    # File handler - all processes write to their own file
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler - only for main process
    if is_main_process():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(console_handler)
    
    # Log initial setup info
    logging.info(f"Logging initialized for rank {rank}")
    logging.info(f"Log file: {log_file}")
    
    return logger

# Check if we're in a distributed setup
def is_main_process():
    """Check if this is the main process in distributed training"""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def get_rank():
    """Get current process rank"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def get_world_size():
    """Get total number of processes"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

# ============================================================================
# Configuration - Optimized for 4x H100 GPUs
# ============================================================================
# Get the actual tar files
tar_files = sorted(glob.glob(os.path.join(config.CO_OCCURRENCE_WEBDATASET_DIR, 'train-*.tar')))

# Initialize logging before we start
output_dir = './qwen25_finetuned_co_occurrence_specialist_multi_gpu'
logger = setup_logging(output_dir)

if is_main_process():
    logging.info(f"Found {len(tar_files)} tar files:")
    for f in tar_files[:5]:  # Show first 5 files
        logging.info(f"  - {f}")
    logging.info(f"  ... and {len(tar_files) - 5} more")

CONFIG = {
    'model_id': config.QWEN_3B_MODEL,
    'webdataset_files': tar_files,
    'output_dir': output_dir,
    
    # UPDATED: Batch size optimized for H100 (80GB each)
    # With 4 GPUs, effective batch size = 4 * 4 * 4 = 64
    'batch_size': 4,  # Per GPU batch size
    'gradient_accumulation_steps': 4,  # Reduced from 16 since we have more GPUs
    
    'num_epochs': 3,
    'learning_rate': 2e-4,
    'max_length': 2048,
    'warmup_ratio': 0.1,
    'logging_steps': 25,
    'save_steps': 250,
    'eval_steps': 125,
    
    # UPDATED: Consider disabling 4-bit for H100 with ample memory
    'use_4bit': False,  # H100 has 80GB, we can use full precision for better quality
    'use_lora': True,
    'dataset_size': 10000,
    
    # MULTI-GPU: Additional settings
    'dataloader_num_workers': 2,  # Reduced to avoid worker issues
    'ddp_find_unused_parameters': False,  # For efficiency
}

if is_main_process():
    logging.info("Configuration (Multi-GPU Optimized):")
    logging.info(f"  Total effective batch size: {CONFIG['batch_size'] * get_world_size() * CONFIG['gradient_accumulation_steps']}")
    for k, v in CONFIG.items():
        if k != 'webdataset_files':
            logging.info(f"  {k}: {v}")

# ============================================================================
# Utility Functions
# ============================================================================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)


# ============================================================================
# Model and Processor Setup
# ============================================================================
if is_main_process():
    logging.info(f"Loading model from: {CONFIG['model_id']}...")

# Load processor first (all processes need it)
processor = AutoProcessor.from_pretrained(CONFIG['model_id'])

# Configure 4-bit quantization if needed
if CONFIG['use_4bit']:
    logging.info("Using 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG['model_id'],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}  # Map to current device
    )


else:
    logging.info("Using full precision (bfloat16) for H100...")
    # Full precision for H100
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CONFIG['model_id'],
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}  # Map to current device
    )

if is_main_process():
    logging.info("✅ Model loaded successfully!")

# Apply LoRA
if CONFIG['use_lora']:
    logging.info("Applying LoRA configuration...")
    if CONFIG['use_4bit']:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,  # Use CAUSAL_LM for vision-language models
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    
    if is_main_process():
        logging.info(f"LoRA Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_percentage:.2f}%)")
        model.print_trainable_parameters()
        logging.info("✅ LoRA applied successfully!")

# ============================================================================
# Data Loading Functions
# ============================================================================


def load_object_info():
    """Load the object information from CSV"""
    csv_path = config.TRAINING_CSV
    df = pd.read_csv(csv_path)
    object_map = dict(zip(df['coco_index'], df['replacement_object']))
    return object_map

# Load object mapping
object_map = load_object_info()



# ============================================================================
# Load and Prepare Dataset - WITH IMPROVED DISTRIBUTED HANDLING
# ============================================================================
cache_dir = '../cached_dataset'

def load_or_create_dataset():
    """Load dataset with proper distributed synchronization"""
    
    # Step 1: Main process checks/creates cache
    if is_main_process():
        assert os.path.exists(cache_dir)
        logging.info("📦 Loading cached dataset...")
        dataset_exists = True

    # Step 2: Synchronize all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Step 3: All processes load from cache
    # Give a small delay for file system synchronization
    if not is_main_process():
        time.sleep(2)
    
    train_dataset = load_from_disk(os.path.join(cache_dir, 'train'))
    eval_dataset = load_from_disk(os.path.join(cache_dir, 'eval'))
    test_dataset = load_from_disk(os.path.join(cache_dir, 'test'))
    
    if is_main_process():
        logging.info(f"✅ Loaded from cache: {len(train_dataset)} train, {len(eval_dataset)} eval, {len(test_dataset)} test")
    
    return train_dataset, eval_dataset, test_dataset

# Load datasets
train_dataset, eval_dataset, test_dataset = load_or_create_dataset()

if is_main_process():
    logging.info(f"📊 Final dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")

# ============================================================================
# Custom Data Collator
# ============================================================================
class MultimodalDataCollator:
    """修复版本：正确处理assistant marker（包含换行符）"""
    
    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length
        
        # ✅ 修复：assistant marker必须包含换行符（单反斜杠）
        self.assistant_marker = "<|im_start|>assistant\n"
        
        # ✅ 修复：使用正确的tokenizer方法
        self.assistant_marker_ids = self.processor.tokenizer(
            self.assistant_marker,
            add_special_tokens=False,
            return_tensors="pt"
        )['input_ids'][0].tolist()
        
        print(f"Assistant marker: {repr(self.assistant_marker)}")
        print(f"Assistant marker token IDs: {self.assistant_marker_ids}")
    
    def __call__(self, features):
        texts = [f['text'] for f in features]
        images = [f['image_data'] for f in features]
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Initialize labels  
        batch['labels'] = batch['input_ids'].clone()
        
        # Process each sample
        for idx in range(len(texts)):
            input_ids = batch['input_ids'][idx]
            input_ids_list = input_ids.tolist()
            
            # Find where assistant content actually starts
            assistant_content_start = None
            marker_len = len(self.assistant_marker_ids)
            
            # Search for assistant marker in token sequence
            for i in range(len(input_ids_list) - marker_len + 1):
                if input_ids_list[i:i+marker_len] == self.assistant_marker_ids:
                    # Found marker! Content starts right after it
                    assistant_content_start = i + marker_len
                    break
            
            if assistant_content_start is not None:
                # Mask everything before assistant's actual content
                batch['labels'][idx, :assistant_content_start] = -100
            else:
                # Couldn't find - mask everything
                batch['labels'][idx, :] = -100
                if idx == 0:  # Only warn once
                    print(f"Warning: Could not find assistant marker in sample {idx}")
        
        # Mask padding
        if self.processor.tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] == self.processor.tokenizer.pad_token_id] = -100
        
        return batch
# ============================================================================
# Custom Training Callback for Enhanced Logging
# ============================================================================
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    """Custom callback for detailed logging during training"""
    
    def __init__(self):
        self.training_start_time = None
        self.step_times = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.training_start_time = time.time()
        if is_main_process():
            logging.info("=" * 60)
            logging.info("Training started")
            logging.info(f"Total training steps: {state.max_steps}")
            logging.info("=" * 60)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        # Log every N steps (more frequently than the default)
        if state.global_step % 10 == 0 and is_main_process():
            # Get current learning rate
            lr = kwargs.get('lr', 0)
            if hasattr(state, 'log_history') and state.log_history:
                last_log = state.log_history[-1]
                if 'learning_rate' in last_log:
                    lr = last_log['learning_rate']
            
            # Estimate time remaining
            if self.training_start_time:
                elapsed = time.time() - self.training_start_time
                steps_done = state.global_step
                steps_remaining = state.max_steps - steps_done
                if steps_done > 0:
                    time_per_step = elapsed / steps_done
                    eta = steps_remaining * time_per_step
                    eta_str = f"{eta/3600:.2f}h" if eta > 3600 else f"{eta/60:.1f}m"
                else:
                    eta_str = "N/A"
                
                logging.info(f"Step {state.global_step}/{state.max_steps} | "
                           f"LR: {lr:.2e} | ETA: {eta_str}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging metrics"""
        if logs and is_main_process():
            # Format log string
            log_items = []
            for key, value in logs.items():
                if key not in ['epoch', 'total_flos']:  # Skip some verbose keysmulti_gpu_fixed
                    if isinstance(value, float):
                        log_items.append(f"{key}: {value:.4f}")
                    else:
                        log_items.append(f"{key}: {value}")
            
            if log_items:
                logging.info(f"Metrics - {' | '.join(log_items)}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics and is_main_process():
            logging.info("=" * 40)
            logging.info("Evaluation Results:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logging.info(f"  {key}: {value:.4f}")
                else:
                    logging.info(f"  {key}: {value}")
            logging.info("=" * 40)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when saving a checkpoint"""
        if is_main_process():
            logging.info(f"💾 Checkpoint saved at step {state.global_step}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if is_main_process() and self.training_start_time:
            total_time = time.time() - self.training_start_time
            logging.info("=" * 60)
            logging.info("Training completed!")
            logging.info(f"Total training time: {total_time/3600:.2f} hours")
            logging.info(f"Total steps: {state.global_step}")
            if state.global_step > 0:
                logging.info(f"Average time per step: {total_time/state.global_step:.2f} seconds")
            logging.info("=" * 60)

# ============================================================================
# Training Arguments - OPTIMIZED FOR MULTI-GPU
# ============================================================================
sft_config = SFTConfig(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    
    # BATCH SIZE & ACCUMULATION
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
    
    # LEARNING RATE & OPTIMIZATION
    learning_rate=CONFIG['learning_rate'],
    warmup_ratio=CONFIG['warmup_ratio'],
    optim="adamw_torch" if not CONFIG['use_4bit'] else "paged_adamw_32bit",
    
    # LOGGING & SAVING
    logging_steps=CONFIG['logging_steps'],
    save_steps=CONFIG['save_steps'],
    eval_steps=CONFIG['eval_steps'],
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # PRECISION & MEMORY
    bf16=True,  # Always use bf16 on H100
    gradient_checkpointing=False,  # H100 has enough memory
    
    # MULTI-GPU SPECIFIC
    ddp_find_unused_parameters=CONFIG['ddp_find_unused_parameters'],
    dataloader_num_workers=CONFIG['dataloader_num_workers'],
    dataloader_pin_memory=True,  # Enable for better performance
    
    # REPORTING
    report_to="none",  # Change to "tensorboard" or "wandb" if needed
    push_to_hub=False,
    remove_unused_columns=False,
    
    # SFT-specific
    dataset_text_field="text",
    max_length=CONFIG['max_length'],
    packing=False,
    
    # Additional optimizations for H100
    tf32=True,  # Enable TF32 for H100
    dataloader_persistent_workers=False,  # Set to False to avoid worker issues
    
    # Disable dataset preprocessing in parallel (causes issues with multi-GPU)
    dataset_kwargs={
        "skip_prepare_dataset": True  # Skip the problematic dataset preparation step
    }
)

# ============================================================================
# Prepare datasets manually to avoid SFTTrainer's problematic preparation
# ============================================================================
if is_main_process():
    logging.info("🔧 Preparing datasets for training...")

# Add EOS token to text field manually
def add_eos_token(example):
    """Add EOS token to the text field"""
    if not example['text'].endswith(processor.tokenizer.eos_token):
        example['text'] = example['text'] + processor.tokenizer.eos_token
    return example

# Process datasets with proper synchronization
if is_main_process():
    logging.info("Adding EOS tokens to datasets...")
    train_dataset = train_dataset.map(add_eos_token, num_proc=1)
    eval_dataset = eval_dataset.map(add_eos_token, num_proc=1)
    logging.info("✅ EOS tokens added")

# Synchronize after dataset preparation
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# ============================================================================
# Initialize SFTTrainer
# ============================================================================
if is_main_process():
    logging.info("🚀 Initializing SFTTrainer for co_occurrence Specialist (Multi-GPU)...")
    logging.info(f"   Using {get_world_size()} GPUs")
    logging.info(f"   Effective batch size: {CONFIG['batch_size'] * get_world_size() * CONFIG['gradient_accumulation_steps']}")

# Initialize trainer with error handling
try:
    # Create custom callback
    logging_callback = LoggingCallback()
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=MultimodalDataCollator(processor, CONFIG['max_length']),
        callbacks=[
            logging_callback,  # ✅ 英文逗号
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        ],
    )
    
    if is_main_process():
        logging.info("✅ SFTTrainer initialized!")
        
except Exception as e:
    logging.error(f"❌ Error initializing trainer on rank {get_rank()}: {e}")
    raise

# ============================================================================
# Train with error handling
# ============================================================================
if is_main_process():
    logging.info("🎯 Starting Multi-GPU training for Size Specialist Model...")
    logging.info("="*60)
    logging.info(f"Training configuration:")
    logging.info(f"  - Number of epochs: {CONFIG['num_epochs']}")
    logging.info(f"  - Learning rate: {CONFIG['learning_rate']}")
    logging.info(f"  - Train dataset size: {len(train_dataset)}")
    logging.info(f"  - Eval dataset size: {len(eval_dataset)}")
    logging.info(f"  - Steps per epoch: ~{len(train_dataset) // (CONFIG['batch_size'] * get_world_size())}")

try:
    # Start training
    training_start_time = time.time()
    logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer.train()
    
    training_duration = time.time() - training_start_time
    if is_main_process():
        logging.info(f"Training completed in {training_duration/3600:.2f} hours")
    
    # Save the final model (only main process)
    if is_main_process():
        logging.info("💾 Saving final model...")
        trainer.save_model(os.path.join(CONFIG['output_dir'], "final_model"))
        processor.save_pretrained(os.path.join(CONFIG['output_dir'], "final_model"))
        logging.info(f"✅ Training completed! Co-occurrence specialist model saved to {CONFIG['output_dir']}/final_model")
        
        # Log final training metrics
        if hasattr(trainer, 'state'):
            logging.info("Final training metrics:")
            if hasattr(trainer.state, 'best_metric'):
                logging.info(f"  Best metric: {trainer.state.best_metric}")
            if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                final_log = trainer.state.log_history[-1]
                for key, value in final_log.items():
                    if isinstance(value, (int, float)):
                        logging.info(f"  {key}: {value:.4f}")
        
except Exception as e:
    logging.error(f"❌ Training error on rank {get_rank()}: {e}")
    logging.exception("Full traceback:")
    # Ensure all processes exit together
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    raise

# Clean up
if torch.distributed.is_initialized():
    torch.distributed.barrier()
    if is_main_process():
        logging.info("✅ All processes completed successfully!")
        logging.info(f"Log files saved in: {CONFIG['output_dir']}/logs/")