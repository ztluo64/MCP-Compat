import json
import csv
from pathlib import Path

ROOT = Path("paper_audit/final_results")

def read_metrics(path):
    with open(path, "r") as f:
        obj = json.load(f)
    if "test_calibrated" in obj:
        return obj["test_calibrated"]
    return obj

def pct(x):
    return f"{x * 100:.2f}"

def th(x):
    return f"{x:.2f}"

def row_from_pair(name, type_, orig_path, bal_path):
    mo = read_metrics(orig_path)
    mb = read_metrics(bal_path)
    return {
        "type": type_,
        "method": name,
        "th": th(mo["threshold"]),
        "orig_acc": pct(mo["acc"]),
        "orig_bal_acc": pct(mo["balanced_acc"]),
        "orig_macro_f1": pct(mo["macro_f1"]),
        "orig_auc": pct(mo["auc"]),
        "bal_acc": pct(mb["acc"]),
        "bal_macro_f1": pct(mb["macro_f1"]),
        "bal_auc": pct(mb["auc"]),
        "orig_source": str(orig_path),
        "balanced_source": str(bal_path),
    }

table1 = [
    row_from_pair(
        "COinCO-VisualNet*", "COinCO-style",
        ROOT / "table1_comparison_sources/coinco_visualnet_original.json",
        ROOT / "table1_comparison_sources/coinco_visualnet_balanced.json",
    ),
    row_from_pair(
        "COinCO-SemanticNet*", "COinCO-style",
        ROOT / "table1_comparison_sources/coinco_semanticnet_original.json",
        ROOT / "table1_comparison_sources/coinco_semanticnet_balanced.json",
    ),
    row_from_pair(
        "COinCO-VisSemanticNet*", "COinCO-style",
        ROOT / "table1_comparison_sources/coinco_vissemanticnet_original.json",
        ROOT / "table1_comparison_sources/coinco_vissemanticnet_balanced.json",
    ),
    row_from_pair(
        "Tri-view + Text", "Ours variant",
        ROOT / "table1_comparison_sources/triview_text_original.json",
        ROOT / "table1_comparison_sources/triview_text_balanced.json",
    ),
    row_from_pair(
        "Ours", "Ours",
        ROOT / "table1_comparison_sources/ours_original.json",
        ROOT / "table1_comparison_sources/ours_balanced.json",
    ),
]

def row_single(name, path):
    m = read_metrics(path)
    return {
        "method": name,
        "th": th(m["threshold"]),
        "acc": pct(m["acc"]),
        "bal_acc": pct(m["balanced_acc"]),
        "macro_f1": pct(m["macro_f1"]),
        "f1_0": pct(m["f1_label0"]),
        "f1_1": pct(m["f1_label1"]),
        "auc": pct(m["auc"]),
        "source": str(path),
    }

table2 = [
    row_single("Full only", ROOT / "table2_input_ablation_sources/full_only_original.json"),
    row_single("Object crop only", ROOT / "table2_input_ablation_sources/crop_only_original.json"),
    row_single("Masked context only", ROOT / "table2_input_ablation_sources/masked_only_original.json"),
    row_single("Tri-view", ROOT / "table2_input_ablation_sources/triview_original.json"),
    row_single("Tri-view + Text", ROOT / "table2_input_ablation_sources/triview_text_original.json"),
    row_single("Ours", ROOT / "table2_input_ablation_sources/ours_original.json"),
]

table3 = [
    ("Tri-view + Text", "none", "-", ROOT / "table3_prior_ablation_sources/triview_text_balanced.json"),
    ("+ Decoupled Prior Prediction", "prior prediction only, not used by classifier", "0.20", ROOT / "table3_prior_ablation_sources/decoupled_prior_prediction_balanced.json"),
    ("+ Latent Prior Compat.", "latent prior-text compatibility", "0.10", ROOT / "table3_prior_ablation_sources/latent_prior_lam01_balanced.json"),
    ("+ Latent Prior Compat.", "latent prior-text compatibility", "0.20", ROOT / "table3_prior_ablation_sources/latent_prior_lam02_balanced.json"),
    ("Ours", "latent + explicit scalar compatibility", "0.20", ROOT / "table3_prior_ablation_sources/ours_balanced.json"),
]
table3_rows = []
for name, module, lam, path in table3:
    r = row_single(name, path)
    r["prior_module"] = module
    r["lambda_prior"] = lam
    table3_rows.append(r)

# Manifest
manifest_rows = []
for r in table1:
    manifest_rows.append({
        "table": "Table 1",
        "method": r["method"],
        "test": "original",
        "source": r["orig_source"],
    })
    manifest_rows.append({
        "table": "Table 1",
        "method": r["method"],
        "test": "balanced",
        "source": r["balanced_source"],
    })
for r in table2:
    manifest_rows.append({
        "table": "Table 2",
        "method": r["method"],
        "test": "original",
        "source": r["source"],
    })
for r in table3_rows:
    manifest_rows.append({
        "table": "Table 3",
        "method": r["method"],
        "test": "balanced",
        "source": r["source"],
    })

with open(ROOT / "manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["table", "method", "test", "source"])
    writer.writeheader()
    writer.writerows(manifest_rows)

# Markdown tables
md = []

md.append("# Final Paper Tables and Sources\n")
md.append("All thresholds are selected once on the validation set by maximizing Macro-F1 and then fixed for test evaluation.\n")

md.append("## Table 1. COinCO-style baselines and structured variants\n")
md.append("| Type | Method | Th. | Original Acc | Original Bal Acc | Original Macro-F1 | Original AUC | Balanced Acc | Balanced Macro-F1 | Balanced AUC |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in table1:
    md.append(f"| {r['type']} | {r['method']} | {r['th']} | {r['orig_acc']} | {r['orig_bal_acc']} | {r['orig_macro_f1']} | {r['orig_auc']} | {r['bal_acc']} | {r['bal_macro_f1']} | {r['bal_auc']} |")

md.append("\n## Table 2. Input decomposition ablation on original test\n")
md.append("| Method | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in table2:
    md.append(f"| {r['method']} | {r['th']} | {r['acc']} | {r['bal_acc']} | {r['macro_f1']} | {r['f1_0']} | {r['f1_1']} | {r['auc']} |")

md.append("\n## Table 3. Prior compatibility ablation on balanced test\n")
md.append("| Method | Prior module | λ_prior | Th. | Acc | Bal Acc | Macro-F1 | F1-0 | F1-1 | AUC |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in table3_rows:
    md.append(f"| {r['method']} | {r['prior_module']} | {r['lambda_prior']} | {r['th']} | {r['acc']} | {r['bal_acc']} | {r['macro_f1']} | {r['f1_0']} | {r['f1_1']} | {r['auc']} |")

with open(ROOT / "final_tables.md", "w") as f:
    f.write("\n".join(md) + "\n")

print("Saved:")
print(ROOT / "manifest.csv")
print(ROOT / "final_tables.md")
