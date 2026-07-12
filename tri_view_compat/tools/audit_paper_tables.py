import json
from pathlib import Path

ROOT = Path("tri_view_compat/outputs/checkpoints")

def load_metric(path):
    path = Path(path)
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)

    if "test_calibrated" in data:
        return data["test_calibrated"]
    if "test" in data:
        return data["test"]

    raise KeyError(f"Unknown metric format: {path}")

def pct(x):
    return f"{float(x) * 100:.2f}"

def th(x):
    return f"{float(x):.2f}"

def print_table(title, rows):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print("| Method | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for method, path in rows:
        m = load_metric(path)
        if m is None:
            print(f"| {method} | MISSING | - | - | - | - | - | - |")
            continue

        print(
            f"| {method} | {th(m['threshold'])} | "
            f"{pct(m['acc'])} | {pct(m['balanced_acc'])} | {pct(m['macro_f1'])} | "
            f"{pct(m['f1_label0'])} | {pct(m['f1_label1'])} | {pct(m['auc'])} |"
        )

table1 = [
    ("Full only", ROOT / "full_only_bs256/threshold_metrics_original_test.json"),
    ("Object crop only", ROOT / "crop_only_bs256/threshold_metrics_original_test.json"),
    ("Masked context only", ROOT / "masked_only_bs256/threshold_metrics_original_test.json"),
    ("Tri-view", ROOT / "triview_no_geo_text_bs256/threshold_metrics_original_test.json"),
    ("Tri-view + Text", ROOT / "triview_text_no_geo_bs256/threshold_metrics_original_test.json"),
    ("Ours", ROOT / "triview_text_prior_v2_lam02_bs256/threshold_metrics_original_test.json"),
]

table2 = [
    ("Full only", ROOT / "full_only_bs256/threshold_metrics_balanced_test.json"),
    ("Tri-view", ROOT / "triview_no_geo_text_bs256/threshold_metrics_balanced_test.json"),
    ("Tri-view + Text", ROOT / "triview_text_no_geo_bs256/threshold_metrics_balanced_test.json"),
    ("Ours", ROOT / "triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json"),
]

table3 = [
    ("Tri-view + Text", ROOT / "triview_text_no_geo_bs256/threshold_metrics_balanced_test.json"),
    ("+ Aux-only Prior", ROOT / "triview_text_prior_aux_lam02_bs256/threshold_metrics_balanced_test.json"),
    ("+ Latent Prior Compat. λ=0.1", ROOT / "triview_text_prior_lam01_bs256/threshold_metrics_balanced_test.json"),
    ("+ Latent Prior Compat. λ=0.2", ROOT / "triview_text_prior_lam02_bs256/threshold_metrics_balanced_test.json"),
    ("Ours", ROOT / "triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json"),
]

print_table("TABLE 1: Original replacement-aware test", table1)
print_table("TABLE 2: Balanced test subset", table2)
print_table("TABLE 3: Prior compatibility ablation on balanced test", table3)
