import json
from pathlib import Path

ROOT = Path("tri_view_compat/outputs/checkpoints")
TOL = 0.011

EXPECTED = {
    "Table1": {
        "Full only": ("full_only_bs256/threshold_metrics_original_test.json", [0.42, 75.85, 63.87, 64.46, 44.34, 84.58, 72.72]),
        "Object crop only": ("crop_only_bs256/threshold_metrics_original_test.json", [0.36, 75.77, 68.21, 67.38, 50.84, 83.92, 78.17]),
        "Masked context only": ("masked_only_bs256/threshold_metrics_original_test.json", [0.44, 70.61, 59.13, 59.05, 37.30, 80.80, 66.63]),
        "Tri-view": ("triview_no_geo_text_bs256/threshold_metrics_original_test.json", [0.33, 78.06, 69.51, 69.38, 53.07, 85.68, 80.22]),
        "Tri-view + Text": ("triview_text_no_geo_bs256/threshold_metrics_original_test.json", [0.38, 81.43, 75.73, 74.80, 61.88, 87.73, 85.18]),
        "Ours": ("triview_text_prior_v2_lam02_bs256/threshold_metrics_original_test.json", [0.37, 81.93, 76.93, 75.71, 63.41, 88.00, 85.45]),
    },
    "Table2": {
        "Full only": ("full_only_bs256/threshold_metrics_balanced_test.json", [0.42, 63.76, 63.76, 61.88, 53.41, 70.35, 72.16]),
        "Tri-view": ("triview_no_geo_text_bs256/threshold_metrics_balanced_test.json", [0.33, 68.26, 68.26, 67.56, 62.80, 72.31, 78.15]),
        "Tri-view + Text": ("triview_text_no_geo_bs256/threshold_metrics_balanced_test.json", [0.38, 74.28, 74.28, 74.06, 71.68, 76.44, 83.52]),
        "Ours": ("triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json", [0.37, 75.99, 75.99, 75.82, 73.80, 77.84, 84.05]),
    },
    "Table3": {
        "Tri-view + Text": ("triview_text_no_geo_bs256/threshold_metrics_balanced_test.json", [0.38, 74.28, 74.28, 74.06, 71.68, 76.44, 83.52]),
        "+ Aux-only Prior": ("triview_text_prior_aux_lam02_bs256/threshold_metrics_balanced_test.json", [0.33, 73.83, 73.83, 73.60, 71.16, 76.05, 82.65]),
        "+ Latent Prior Compat. λ=0.1": ("triview_text_prior_lam01_bs256/threshold_metrics_balanced_test.json", [0.38, 73.47, 73.47, 73.21, 70.59, 75.84, 83.50]),
        "+ Latent Prior Compat. λ=0.2": ("triview_text_prior_lam02_bs256/threshold_metrics_balanced_test.json", [0.34, 75.27, 75.27, 74.97, 72.25, 77.70, 84.15]),
        "Ours": ("triview_text_prior_v2_lam02_bs256/threshold_metrics_balanced_test.json", [0.37, 75.99, 75.99, 75.82, 73.80, 77.84, 84.05]),
    }
}

FIELDS = ["threshold", "acc", "balanced_acc", "macro_f1", "f1_label0", "f1_label1", "auc"]

def load_metric(path):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data["test_calibrated"]

def to_percent(field, value):
    if field == "threshold":
        return float(value)
    return float(value) * 100

all_ok = True

for table, rows in EXPECTED.items():
    print(f"\n===== {table} =====")

    for method, (rel_path, expected_values) in rows.items():
        m = load_metric(rel_path)
        print(f"\n{method}")
        for field, expected in zip(FIELDS, expected_values):
            got = to_percent(field, m[field])
            diff = got - expected
            ok = abs(diff) <= TOL
            status = "OK" if ok else "MISMATCH"
            print(f"  {field:14s} got={got:8.3f} expected={expected:8.3f} diff={diff:+8.3f} {status}")
            if not ok:
                all_ok = False

print("\nFINAL:", "ALL OK" if all_ok else "HAS MISMATCH")
if not all_ok:
    raise SystemExit(1)
