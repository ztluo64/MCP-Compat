import os
import pandas as pd

SEED = 777

def build_one(in_csv, out_csv):
    df = pd.read_csv(in_csv)

    print("\nInput:", in_csv)
    print("shape:", df.shape)
    print("label counts:")
    print(df["label"].value_counts().sort_index())

    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]

    n = min(len(df0), len(df1))

    df0_bal = df0.sample(n=n, random_state=SEED)
    df1_bal = df1.sample(n=n, random_state=SEED)

    out = pd.concat([df0_bal, df1_bal], axis=0)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    print("Output:", out_csv)
    print("shape:", out.shape)
    print("label counts:")
    print(out["label"].value_counts().sort_index())


def main():
    build_one(
        "tri_view_compat/outputs/splits/test.csv",
        "tri_view_compat/outputs/splits_balanced/test_balanced_seed777.csv",
    )

    # 如果 Qwen tri-view / canvas split 已存在，也顺手生成对应 balanced split。
    if os.path.exists("qwen_lora/outputs/splits_triview/test.csv"):
        build_one(
            "qwen_lora/outputs/splits_triview/test.csv",
            "qwen_lora/outputs/splits_triview/test_balanced_seed777.csv",
        )

    if os.path.exists("qwen_lora/outputs/splits_canvas/test.csv"):
        build_one(
            "qwen_lora/outputs/splits_canvas/test.csv",
            "qwen_lora/outputs/splits_canvas/test_balanced_seed777.csv",
        )


if __name__ == "__main__":
    main()
