# CSV Mappings

Download the following files from HuggingFace and place them in this directory:

- `training_inpainting_info.csv`
- `testing_inpainting_info.csv`

Source: https://huggingface.co/datasets/COinCO/COinCO-dataset/tree/main/inpainting_info

These CSV files map `coco_index` to inpainting metadata (object category, bounding box, in/out-of-context label, etc.) and are required by the data generation and evaluation scripts.
