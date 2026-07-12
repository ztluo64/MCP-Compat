# Context Reasoning Results

This directory contains context reasoning outputs from three VLMs (InternVL, Molmo, Qwen) on inpainted images.

## Download CSV files

Download from HuggingFace and place in the corresponding subdirectories:

Source: https://huggingface.co/datasets/COinCO/COinCO-dataset/tree/main/context_reasoning

```
context_reasoning-res/
├── internvl/
│   └── context_reasoning_internvl.csv          # InternVL results on training split
├── molmo/
│   └── context_reasoning_molmo.csv             # Molmo results on training split
└── qwen/
    ├── context_reasoning_qwen.csv              # Qwen results on training split
    └── context_reasoning_qwen_testing.csv      # Qwen results on testing split
```

The training CSVs (internvl, molmo, qwen) are used by `SFT_data/*-WebDataset.ipynb` to build WebDatasets via multi-model consensus.
The testing CSV (qwen only) is used by `SFT_data/*-WebDataset-Test.ipynb` to build test WebDatasets.
