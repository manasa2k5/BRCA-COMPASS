"""
plot_ablation.py
BRCA-COMPASS — Ablation Study Plot Generator
Save to : M:\brca1_pathogenicity\plot_ablation.py
Run     : python ablation_study.py   (generates results first)
          python plot_ablation.py    (then run this)
Output  : results\figures\manuscript\fig4_metrics_bar.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_PATH = Path("M:/brca1_pathogenicity/results/figures/manuscript/fig4_metrics_bar.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Paste your actual ablation results here ──────────────────────────────
# These come from the SUMMARY COMPARISON TABLE printed by ablation_study.py
# Replace with your real numbers after running ablation_study.py

RESULTS = [
    {
        "name"    : "Full Model\n(48 features)",
        "n"       : 48,
        "test_auc": 0.9987,
        "test_f1" : 0.9904,
        "test_mcc": 0.9775,
        "cv_auc"  : 0.9994,
        "cv_std"  : 0.0003,
        "gap"     : 0.0012,
    },
    {
        "name"    : "Exp A: No LOF/\nConsequence Proxies",
        "n"       : None,   # fill from your output
        "test_auc": None,
        "test_f1" : None,
        "test_mcc": None,
        "cv_auc"  : None,
        "cv_std"  : None,
        "gap"     : None,
    },
    {
        "name"    : "Exp B: dbNSFP\nScores Only",
        "n"       : None,
        "test_auc": None,
        "test_f1" : None,
        "test_mcc": None,
        "cv_auc"  : None,
        "cv_std"  : None,
        "gap"     : None,
    },
    {
        "name"    : "Exp C: ClinVar\nEngineered Only",
        "n"       : None,
        "test_auc": None,
        "test_f1" : None,
        "test_mcc": None,
        "cv_auc"  : None,
        "cv_std"  : None,
        "gap"     : None,
    },
    {
        "name"    : "Exp D: Conservative\n(No LOF/Submitters)",
        "n"       : None,
        "test_auc": None,
        "test_f1" : None,
        "test_mcc": None,
        "cv_auc"  : None,
        "cv_std"  : None,
        "gap"     : None,
    },
]

# ─────────────────────────────────────────────────────────────────────────
# NOTE: Instead of filling numbers manually above, you can run
# ablation_study.py and add this block at the very bottom of that file
# to auto-save results to a JSON, then load here automatically.
# See the EASIER METHOD comment below.
# ─────────────────────────────────────────────────────────────────────────

# EASIER METHOD — add these 3 lines to the bottom of ablation_study.py:
#
#   import json
#   results_out = [r for r in [r0,rA,rB,rC,rD] if r]
#   json.dump(results_out, open("results/ablation_results.json","w"), indent=2)
#
# Then replace the RESULTS list above with:
#
#   import json
#   raw = json.load(open("M:/brca1_pathogenicity/results/ablation_results.json"))
#   RESULTS = [
#       {
#           "name"    : r["name"].replace("EXPERIMENT ", "Exp ").replace("FULL MODEL", "Full Model"),
#           "n"       : r["n_features"],
#           "test_auc": r["test"]["ROC_AUC"],
#           "test_f1" : r["test"]["F1"],
#           "test_mcc": r["test"]["MCC"],
#           "cv_auc"  : r["cv_mean"],
#           "cv_std"  : r["cv_std"],
#           "gap"     : r["gap"],
#       }
#       for r in raw
#   ]


def plot_ablation(results):
    # Filter only rows with actual data
    valid = [r for r in results if r["test_auc"] is not None]
    if len(valid) == 0:
        print("ERROR: No numeric results found in RESULTS list.")
        print("Run ablation_study.py first, copy the numbers into RESULTS above,")
        print("or use the EASIER METHOD (JSON export) described in the comments.")
        return

    names    = [r["name"]     for r in valid]
    auc_vals = [r["test_auc"] for r in valid]
    f1_vals  = [r["test_f1"]  for r in valid]
    mcc_vals = [r["test_mcc"] for r in valid]
    cv_vals  = [r["cv_auc"]   for r in valid]
    cv_stds  = [r["cv_std"]   for r in valid]

    x    = np.arange(len(names))
    bw   = 0.20   # bar width
    cols = ["#2E86C1", "#28B463", "#E67E22", "#8E44AD"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    # ── LEFT: grouped bar chart ──────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - 1.5*bw, auc_vals, bw, label="Test ROC-AUC", color=cols[0], alpha=0.9)
    ax.bar(x - 0.5*bw, f1_vals,  bw, label="Test F1",      color=cols[1], alpha=0.9)
    ax.bar(x + 0.5*bw, mcc_vals, bw, label="Test MCC",     color=cols[2], alpha=0.9)
    ax.errorbar(x + 1.5*bw, cv_vals, yerr=cv_stds,
                fmt="D", color=cols[3], capsize=4, markersize=6,
                label="CV AUC ± SD")

    # Value labels on bars
    for i, (a, f, m) in enumerate(zip(auc_vals, f1_vals, mcc_vals)):
        ax.text(i - 1.5*bw, a + 0.003, f"{a:.4f}", ha="center", va="bottom",
                fontsize=7, color=cols[0], fontweight="bold")
        ax.text(i - 0.5*bw, f + 0.003, f"{f:.4f}", ha="center", va="bottom",
                fontsize=7, color=cols[1], fontweight="bold")
        ax.text(i + 0.5*bw, m + 0.003, f"{m:.4f}", ha="center", va="bottom",
                fontsize=7, color=cols[2], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(max(0, min(mcc_vals) - 0.08), 1.03)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation Study: Test Set Performance by Feature Configuration",
                 fontsize=11, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Full-model reference lines
    ax.axhline(auc_vals[0], color=cols[0], linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axhline(f1_vals[0],  color=cols[1], linestyle=":", linewidth=1.0, alpha=0.5)

    # ── RIGHT: delta AUC from full model ─────────────────────────────────
    ax2 = axes[1]
    deltas     = [v - auc_vals[0] for v in auc_vals]
    bar_colors = ["#AAB7B8" if i == 0 else
                  ("#2E86C1" if d >= 0 else "#E74C3C")
                  for i, d in enumerate(deltas)]
    bars = ax2.barh(names[::-1], deltas[::-1], color=bar_colors[::-1],
                    edgecolor="white", linewidth=0.5, alpha=0.92)

    for bar, delta in zip(bars, deltas[::-1]):
        xpos = bar.get_width()
        sign = "+" if xpos >= 0 else ""
        ax2.text(xpos + (0.0005 if xpos >= 0 else -0.0005),
                 bar.get_y() + bar.get_height()/2,
                 f"{sign}{xpos:.4f}",
                 va="center",
                 ha="left" if xpos >= 0 else "right",
                 fontsize=9, fontweight="bold",
                 color="#1A5276" if xpos >= 0 else "#922B21")

    ax2.axvline(0, color="black", linewidth=1.2)
    ax2.set_xlabel("ΔAUC vs Full Model", fontsize=11)
    ax2.set_title("AUC Drop by Ablated Feature Set\n(negative = worse than full model)",
                  fontsize=11, fontweight="bold", pad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.spines[["top", "right"]].set_visible(False)

    ref_patch = mpatches.Patch(color="#AAB7B8", label="Full model (reference)")
    neg_patch = mpatches.Patch(color="#E74C3C", label="Performance drop")
    ax2.legend(handles=[ref_patch, neg_patch], fontsize=9, loc="lower right")

    plt.suptitle(
        "BRCA-COMPASS Ablation Study  |  Feature Category Contribution Analysis",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"✅  Saved → {OUT_PATH}")


if __name__ == "__main__":
    print("="*60)
    print("BRCA-COMPASS | Ablation Study Plot")
    print("="*60)
    plot_ablation(RESULTS)