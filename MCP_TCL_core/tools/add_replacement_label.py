import os
import pandas as pd

split_dir = "tri_view_compat/outputs/splits"
splits = ["train", "val", "test"]

dfs = []
for s in splits:
    path = os.path.join(split_dir, f"{s}.csv")
    df = pd.read_csv(path)
    df["split_name"] = s
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

def norm_name(x):
    return str(x).strip().lower()

# Build class_name -> prior_label mapping.
mapping = {}
conflicts = []

for _, row in all_df.iterrows():
    name = norm_name(row["class_name"])
    label = int(row["prior_label"])
    if name in mapping and mapping[name] != label:
        conflicts.append((name, mapping[name], label))
    mapping[name] = label

print("num class_name mappings:", len(mapping))
if conflicts:
    print("WARNING: conflicting mappings found:")
    for c in conflicts[:20]:
        print(c)
    raise RuntimeError("Conflicting class_name -> prior_label mappings.")

missing = sorted(set(norm_name(x) for x in all_df["replacement_object"]) - set(mapping.keys()))
print("missing replacement names:", missing)

if missing:
    raise RuntimeError(
        "Some replacement_object names are not found in class_name -> prior_label mapping. "
        "Need fallback COCO category mapping."
    )

for s in splits:
    path = os.path.join(split_dir, f"{s}.csv")
    df = pd.read_csv(path)
    df["replacement_label"] = df["replacement_object"].apply(lambda x: mapping[norm_name(x)]).astype(int)
    df.to_csv(path, index=False)
    print(f"saved {path}, shape={df.shape}")
    print(df[["coco_index", "class_name", "prior_label", "replacement_object", "replacement_label", "label"]].head())

print("done")
