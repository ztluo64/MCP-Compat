import os
import pandas as pd
from pathlib import Path

split_dir = Path("tri_view_compat/outputs/splits")
splits = ["train", "val", "test"]


def norm_name(x):
    return str(x).strip().lower()


# Build class_name -> prior_label mapping from TRAIN split only.
train_path = split_dir / "train.csv"
train_df = pd.read_csv(train_path)
train_df["class_name_norm"] = train_df["class_name"].apply(norm_name)

mapping = {}
conflicts = []

for _, row in train_df.iterrows():
    name = row["class_name_norm"]
    label = int(row["prior_label"])

    if name in mapping and mapping[name] != label:
        conflicts.append((name, mapping[name], label))

    mapping[name] = label

print("num class_name mappings from train:", len(mapping))

if conflicts:
    print("WARNING: conflicting mappings found:")
    for c in conflicts[:20]:
        print(c)
    raise RuntimeError("Conflicting class_name -> prior_label mappings in train split.")

if len(mapping) != 80:
    raise RuntimeError(f"Expected 80 COCO class mappings, got {len(mapping)}.")

total_mismatch = 0

for s in splits:
    path = split_dir / f"{s}.csv"
    df = pd.read_csv(path)

    repl_norm = df["replacement_object"].apply(norm_name)
    missing = sorted(set(repl_norm) - set(mapping.keys()))

    print(f"{s}: missing replacement names = {len(missing)}")
    if missing:
        print("missing examples:", missing[:20])
        raise RuntimeError(f"Missing replacement names in {s}: {missing[:20]}")

    new_label = repl_norm.apply(lambda x: mapping[x]).astype(int)

    if "replacement_label" in df.columns:
        mismatch = (df["replacement_label"].astype(int) != new_label).sum()
        print(f"{s}: mismatch with existing replacement_label = {mismatch}")
        total_mismatch += int(mismatch)

    df["replacement_label"] = new_label
    df.to_csv(path, index=False)
    print(f"saved {path}, shape={df.shape}")

print("total mismatch with existing labels:", total_mismatch)

if total_mismatch != 0:
    raise RuntimeError("Train-only mapping differs from existing replacement_label values.")

print("Done. Replacement labels are generated using train-only class mapping.")
