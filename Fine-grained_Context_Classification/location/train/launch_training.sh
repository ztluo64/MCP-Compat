#!/bin/bash

# Multi-GPU Training Launch Script for Qwen2.5-VL-3B Fine-tuning
# Works with or without accelerate config file

echo "🚀 Starting Qwen2.5-VL-3B Multi-GPU Training on 4x H100"
echo "============================================"
echo "Timestamp: $(date)"

# Set environment variables for optimal performance on H100
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Increase timeout for large model training
export NCCL_TIMEOUT=3600  # 1 hour timeout (default is 30 minutes)
export NCCL_BLOCKING_WAIT=1  # Better error messages

# Optional: Enable NCCL optimizations for H100
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Enable better error reporting
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# Create output directory if it doesn't exist
mkdir -p ./qwen25_finetuned_location_specialist_multi_gpu

# Choose which script to run (fixed or original)
TRAINING_SCRIPT="qwen25_3B_accelerate_multi_gpu_fixed.py"
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "⚠️ Fixed script not found, using original script..."
    TRAINING_SCRIPT="qwen25_3B_accelerate_multi_gpu.py"
fi

echo "📝 Using script: $TRAINING_SCRIPT"

# Track if we successfully launched training
LAUNCHED=false

# Method 1: Try with simple config file if it exists
if [ -f "accelerate_config_simple.yaml" ]; then
    echo -e "\n✅ Found accelerate_config_simple.yaml"
    echo "Attempting Method 1: Launching with simple config file..."
    accelerate launch \
        --config_file accelerate_config_simple.yaml \
        $TRAINING_SCRIPT
    EXIT_CODE=$?
    LAUNCHED=true
    
# Method 2: Try with original config file if simple doesn't exist
elif [ -f "accelerate_config.yaml" ]; then
    echo -e "\n✅ Found accelerate_config.yaml"
    echo "Attempting Method 2: Launching with original config file..."
    accelerate launch \
        --config_file accelerate_config.yaml \
        $TRAINING_SCRIPT
    EXIT_CODE=$?
    LAUNCHED=true

# Method 3: Use command line arguments if no config files exist
else
    echo -e "\n⚠️ No config file found"
    echo "Method 3: Launching with command line arguments..."
    
    accelerate launch \
        --multi_gpu \
        --num_processes 4 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        --main_process_port 29500 \
        $TRAINING_SCRIPT
    
    EXIT_CODE=$?
    LAUNCHED=true
fi

# Check final status
if [ "$LAUNCHED" = true ]; then
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "\n============================================"
        echo "✅ Training completed successfully!"
        echo "Timestamp: $(date)"
    else
        echo -e "\n============================================"
        echo "❌ Training failed with exit code: $EXIT_CODE"
        echo "Timestamp: $(date)"
        echo ""
        echo "Check the log files in ./qwen3_finetuned_size_specialist_multi_gpu/logs/"
        exit $EXIT_CODE
    fi
else
    echo -e "\n============================================"
    echo "❌ Failed to launch training"
    echo "Timestamp: $(date)"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check your Accelerate installation: pip show accelerate"
    echo "2. Try updating Accelerate: pip install --upgrade accelerate"
    echo "3. Check GPU availability: nvidia-smi"
    exit 1
fi