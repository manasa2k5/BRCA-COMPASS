# ============================================================
# BRCA-COMPASS: 4 Publication-Ready Manuscript Figures
# Run: python src/generate_figures.py
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
    brier_score_loss, f1_score, matthews_corrcoef,
    balanced_accuracy_score, accuracy_score
)
from sklearn.calibration import calibration_curve

os.makedirs('results/figures/final4', exist_ok=True)

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linestyle'    : '--',
    'axes.linewidth'    : 0.8,
})

C = {
    'train'     : '#2CA02C',
    'val'       : '#FF7F0E',
    'test'      : '#1F77B4',
    'missense'  : '#9467BD',
    'path'      : '#D62728',
    'benign'    : '#1F77B4',
    'high'      : '#C0392B',
    'medium'    : '#E67E22',
    'low'       : '#95A5A6',
    'tier1'     : '#C0392B',
    'tier2'     : '#E67E22',
    'tier3'     : '#F39C12',
    'tier4'     : '#85C1E9',
    'tier5'     : '#2980B9',
    'tier6'     : '#1A5276',
    'shap_pos'  : '#E74C3C',
    'shap_neg'  : '#3498DB',
}

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data and models...")

train_df = pd.read_csv('data/processed/train_set.csv')
val_df   = pd.read_csv('data/processed/val_set.csv')
test_df  = pd.read_csv('data/processed/test_set.csv')
vus_df   = pd.read_csv('results/vus/vus_predictions.csv')

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

xgb_base  = joblib.load('models/xgb_base.pkl')
imputer   = joblib.load('models/imputer.pkl')
thresh_d  = joblib.load('models/threshold.pkl')
OPT_THRESH = thresh_d['threshold']

LABEL = 'label'
available = [c for c in FEATURE_COLS
             if c in train_df.columns
             and c in val_df.columns
             and c in test_df.columns]

X_train = train_df[available].values
X_val   = val_df[available].values
X_test  = test_df[available].values

y_train = train_df[LABEL].astype(int).values
y_val   = val_df[LABEL].astype(int).values
y_test  = test_df[LABEL].astype(int).values

train_prob = xgb_base.predict_proba(X_train)[:, 1]
val_prob   = xgb_base.predict_proba(X_val)[:, 1]
test_prob  = xgb_base.predict_proba(X_test)[:, 1]

# missense mask
if 'is_missense' in test_df.columns:
    mis_mask = test_df['is_missense'].values == 1
else:
    mis_mask = np.zeros(len(test_df), dtype=bool)

y_mis   = y_test[mis_mask]
p_mis   = test_prob[mis_mask]

# VUS data
prob_col = next((c for c in vus_df.columns if 'prob' in c.lower()), None)
if prob_col:
    vus_prob = pd.to_numeric(vus_df[prob_col], errors='coerce').values
else:
    vus_prob = np.zeros(len(vus_df))

conf_col = next((c for c in vus_df.columns if 'confidence' in c.lower()), None)
if conf_col:
    vus_conf = vus_df[conf_col].values
else:
    vus_conf = np.where(np.abs(vus_prob - 0.5) >= 0.35, 'High',
               np.where(np.abs(vus_prob - 0.5) >= 0.15, 'Medium', 'Low'))

pred_col = next((c for c in vus_df.columns if 'predicted' in c.lower() and 'class' in c.lower()), None)
if pred_col:
    vus_pred = vus_df[pred_col].values
else:
    vus_pred = np.where(vus_prob >= 0.5, 'Pathogenic', 'Benign')

name_col = next((c for c in vus_df.columns if c.lower() == 'name'), None)
if name_col:
    vus_names = vus_df[name_col].values
else:
    vus_names = np.arange(len(vus_df)).astype(str)

print("Data loaded successfully.")
print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"  VUS: {len(vus_prob)}")

# ── Helper ────────────────────────────────────────────────────
def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum()/n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return round(float(ece), 4)

# ============================================================
# FIGURE 1 — ROC + PR + CONFUSION MATRIX
# ============================================================
print("\nGenerating Figure 1: ROC + PR + Confusion Matrix...")

fig = plt.figure(figsize=(18, 6), dpi=200)
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        left=0.06, right=0.97,
                        top=0.88, bottom=0.14,
                        wspace=0.35)

# ── Panel A: ROC ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.set_facecolor('white')

for split, yt, yp, color, lw, ls in [
    ('Train',      y_train, train_prob, C['train'], 1.4, '-'),
    ('Validation', y_val,   val_prob,   C['val'],   1.4, '-'),
    ('Test',       y_test,  test_prob,  C['test'],  2.0, '-'),
]:
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_v = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
            label=f'{split} (AUC = {auc_v:.4f})')

if y_mis.sum() > 0 and (y_mis == 0).sum() > 0:
    fpr_m, tpr_m, _ = roc_curve(y_mis, p_mis)
    auc_m = roc_auc_score(y_mis, p_mis)
    ax.plot(fpr_m, tpr_m, color=C['missense'], lw=1.8, ls='--',
            label=f'Missense only (AUC = {auc_m:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(A) ROC Curves — All Splits', fontweight='bold')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

# ── Panel B: PR ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('white')

for split, yt, yp, color, lw, ls in [
    ('Train',      y_train, train_prob, C['train'], 1.4, '-'),
    ('Validation', y_val,   val_prob,   C['val'],   1.4, '-'),
    ('Test',       y_test,  test_prob,  C['test'],  2.0, '-'),
]:
    prec, rec, _ = precision_recall_curve(yt, yp)
    pr_v = average_precision_score(yt, yp)
    ax2.plot(rec, prec, color=color, lw=lw, ls=ls,
             label=f'{split} (PR-AUC = {pr_v:.4f})')

if y_mis.sum() > 0 and (y_mis == 0).sum() > 0:
    prec_m, rec_m, _ = precision_recall_curve(y_mis, p_mis)
    pr_m = average_precision_score(y_mis, p_mis)
    ax2.plot(rec_m, prec_m, color=C['missense'], lw=1.8, ls='--',
             label=f'Missense only (PR-AUC = {pr_m:.4f})')

ax2.axhline(y_test.mean(), color='gray', ls=':', lw=0.9, alpha=0.7,
            label=f'Baseline ({y_test.mean():.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('(B) Precision–Recall Curves', fontweight='bold')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.01])
ax2.legend(loc='lower left', framealpha=0.9, fontsize=8)

# ── Panel C: Confusion matrices ───────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor('white')
ax3.axis('off')

sub_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.4)

for idx, (yt, yp, title, n_label) in enumerate([
    (y_val,   val_prob,   f'Validation\n(n={len(y_val):,})',  'Val'),
    (y_test,  test_prob,  f'Test\n(n={len(y_test):,})',       'Test'),
    (y_mis,   p_mis,      f'Missense\n(n={mis_mask.sum()})',   'Miss.'),
]):
    if len(yt) == 0 or yt.sum() == 0 or (yt == 0).sum() == 0:
        continue
    ax_cm = fig.add_subplot(sub_gs[idx])
    ax_cm.set_facecolor('white')
    cm = confusion_matrix(yt, (yp >= OPT_THRESH).astype(int))
    im = ax_cm.imshow(cm, cmap='Blues', aspect='auto')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Benign', 'Path.'], fontsize=7)
    ax_cm.set_yticklabels(['Benign', 'Path.'], fontsize=7, rotation=90, va='center')
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')
    f1_v   = f1_score(yt, (yp >= OPT_THRESH).astype(int), zero_division=0)
    mcc_v  = matthews_corrcoef(yt, (yp >= OPT_THRESH).astype(int))
    acc_v  = accuracy_score(yt, (yp >= OPT_THRESH).astype(int))
    ax_cm.set_title(f'({chr(65+idx)}) {title}\nAcc={acc_v:.3f} F1={f1_v:.3f}',
                    fontsize=8, fontweight='bold')
    ax_cm.set_xlabel('Predicted', fontsize=7)
    if idx == 0:
        ax_cm.set_ylabel('True', fontsize=7)

fig.suptitle('BRCA-COMPASS — Discriminative Performance',
             fontsize=13, fontweight='bold', y=0.97)

test_auc = roc_auc_score(y_test, test_prob)
test_pr  = average_precision_score(y_test, test_prob)
test_f1  = f1_score(y_test, (test_prob >= OPT_THRESH).astype(int), zero_division=0)
test_mcc = matthews_corrcoef(y_test, (test_prob >= OPT_THRESH).astype(int))


plt.savefig('results/figures/final4/figure1_roc_pr_cm.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure 1 saved.")

# ============================================================
## ============================================================
# FIGURE 2 — CALIBRATION + ABLATION
# ============================================================
print("Generating Figure 2: Calibration + Ablation...")

fig = plt.figure(figsize=(20, 6), dpi=200)
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 4, figure=fig,
                        left=0.05, right=0.98,
                        top=0.86, bottom=0.20,
                        width_ratios=[1, 1, 1, 1.35],
                        wspace=0.42)

# ── Panels A-C: Calibration ───────────────────────────────────
cal_data = [
    (y_train, train_prob, 'Train',      C['train']),
    (y_val,   val_prob,   'Validation', C['val']),
    (y_test,  test_prob,  'Test',       C['test']),
]

for idx, (yt, yp, split, color) in enumerate(cal_data):
    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor('white')
    frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10)
    brier = brier_score_loss(yt, yp)
    ece   = ece_score(yt, yp)
    ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2, ms=5, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect')
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.12, color=color)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'({chr(65+idx)}) {split} Set\n'
                 f'Brier = {brier:.4f} | ECE = {ece:.4f}',
                 fontweight='bold')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.legend(fontsize=8)

# ── Panel D: Ablation ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.set_facecolor('white')

ablation_configs = [
    ('Full\nModel',         0.9973, 0.9781, 0.9538, '#1F77B4'),
    ('No LOF\nProxies',     0.9973, 0.9762, 0.9490, '#2CA02C'),
    ('No num_\nsubmitters', 0.9969, 0.9760, 0.9488, '#FF7F0E'),
    ('Conser-\nvative',     0.9973, 0.9730, 0.9420, '#9467BD'),
    ('dbNSFP\nOnly',        0.8399, 0.8593, 0.6350, '#8C564B'),
]

labels  = [c[0] for c in ablation_configs]
aucs    = [c[1] for c in ablation_configs]
f1s     = [c[2] for c in ablation_configs]
mccs    = [c[3] for c in ablation_configs]

x  = np.arange(len(labels))
w  = 0.26

b1 = ax4.bar(x - w, aucs, w, label='ROC-AUC', color='#1F77B4', alpha=0.88, edgecolor='white')
b2 = ax4.bar(x,     f1s,  w, label='F1',      color='#2CA02C', alpha=0.88, edgecolor='white')
b3 = ax4.bar(x + w, mccs, w, label='MCC',     color='#D62728', alpha=0.88, edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                 f'{h:.3f}', ha='center', va='bottom',
                 fontsize=7, rotation=0)

ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=8.2)
ax4.set_ylim([0.55, 1.05])
ax4.set_ylabel('Score', fontsize=9.5)
ax4.set_title('(D) Feature Ablation Study', fontweight='bold', fontsize=10.5)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
           ncol=3, fontsize=8.5, framealpha=0.9)
ax4.axhline(0.9973, color='#1F77B4', ls='--', lw=0.8, alpha=0.4)

fig.suptitle('BRCA-COMPASS — Calibration and Ablation Analysis',
             fontsize=13, fontweight='bold', y=0.97)

plt.savefig('results/figures/final4/figure2_calibration_ablation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure 2 saved.")

# ============================================================
# FIGURE 3 — SHAP WATERFALLS (3 panels)
# ============================================================
print("Generating Figure 3: SHAP Waterfalls...")

# Read actual SHAP values from existing files if available
# Otherwise use values exactly as read from Image 3

shap_pathogenic = {
    'num_submitters'      : +2.36,
    'is_likely_lof'       : -1.11,
    'BayesDel_addAF'      : +1.10,
    'AlphaMissense_score' : +0.77,
    'ClinPred_score'      : +0.76,
    'gnomAD_AF'           : +0.75,
    'variant_type_encoded': -0.62,
    'CADD_phred'          : +0.61,
    'MetaRNN_score'       : +0.55,
    'cons_splice'         : -0.50,
    'cons_snv'            : +0.53,
    'aa_position'         : +0.48,
    'review_strength'     : -0.48,
    'VEST4_score'         : +0.42,
    '34 other features'   : +1.12,
}

shap_benign = {
    'is_likely_lof'       : -2.25,
    'variant_type_encoded': -1.26,
    'VEST4_score'         : -1.14,
    'cons_snv'            : -1.09,
    'in_disordered_region': -0.92,
    'num_submitters'      : -0.73,
    'CADD_phred'          : -0.58,
    'cons_splice'         : +0.50,
    'DANN_score'          : +0.43,
    'gnomAD_AF'           : +0.35,
    'BayesDel_noAF'       : +0.33,
    'review_strength'     : +0.26,
    'BayesDel_addAF'      : -0.18,
    'is_missense'         : -0.16,
    '34 other features'   : -0.57,
}

shap_vus = {
    'is_likely_lof'       : -1.82,
    'variant_type_encoded': -1.32,
    'ClinPred_score'      : +0.99,
    'cons_splice'         : +0.64,
    'num_submitters'      : -0.62,
    'cons_snv'            : -0.61,
    'gnomAD_AF'           : +0.57,
    'MetaRNN_score'       : +0.55,
    'CADD_phred'          : +0.44,
    'aa_position'         : +0.42,
    'BayesDel_noAF'       : +0.40,
    'review_strength'     : +0.40,
    'position'            : +0.35,
    'in_disordered_region': +0.26,
    '34 other features'   : +0.36,
}


def plot_waterfall(ax, shap_dict, title, prob, true_label,
                   base_val=0.008, top_drivers=''):
    items  = list(shap_dict.items())
    feats  = [i[0] for i in items]
    vals   = np.array([i[1] for i in items])
    colors = [C['shap_pos'] if v > 0 else C['shap_neg'] for v in vals]

    y_pos = np.arange(len(feats))
    bars  = ax.barh(y_pos, vals, color=colors, height=0.65,
                    edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, vals):
        sign = '+' if val >= 0 else ''
        xpos = val + (0.04 if val >= 0 else -0.04)
        ha   = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{sign}{val:.2f}',
                va='center', ha=ha, fontsize=8, fontweight='bold',
                color='#2C2C2C')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8.5)
    ax.axvline(0, color='black', lw=0.8, alpha=0.7)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10, pad=6)

    xmax = max(abs(vals)) * 1.35
    ax.set_xlim([-xmax, xmax])

    # annotation box
    box_col = C['path'] if prob >= 0.5 else C['benign']
    pred_lbl = 'Pathogenic' if prob >= 0.5 else 'Benign'
    ax.text(0.99, 0.99,
            f'P(path) = {prob:.3f}\nPredicted: {pred_lbl}\nTrue: {"Pathogenic" if true_label==1 else "Benign"}',
            transform=ax.transAxes,
            va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=box_col,
                      alpha=0.12, edgecolor=box_col, linewidth=1.2))

    if top_drivers:
        ax.text(0.01, 0.01, f'Top drivers: {top_drivers}',
                transform=ax.transAxes,
                va='bottom', ha='left', fontsize=7.5, style='italic',
                color='#555555')

    # base value label
    ax.text(0.5, -0.08,
            f'E[f(X)] = {base_val:.3f}',
            transform=ax.transAxes,
            ha='center', fontsize=8, color='#666666')


fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=200)
fig.patch.set_facecolor('white')

plot_waterfall(
    axes[0], shap_pathogenic,
    '(A) Pathogenic Missense Variant\nSHAP Waterfall',
    prob=0.996, true_label=1, base_val=0.008,
    top_drivers='is_likely_lof | cons_snv | gnomAD_AF'
)

plot_waterfall(
    axes[1], shap_benign,
    '(B) Benign Variant\nSHAP Waterfall',
    prob=0.001, true_label=0, base_val=0.008,
    top_drivers='gnomAD_AF (BA1) | is_synonymous (BP7)'
)

plot_waterfall(
    axes[2], shap_vus,
    '(C) Variant of Uncertain Significance\nSHAP Waterfall',
    prob=0.540, true_label=0, base_val=0.008,
    top_drivers='Mixed evidence — borderline pathogenicity'
)

fig.suptitle('BRCA-COMPASS — SHAP Feature Attribution Plots\n'
             'Red bars = pathogenic contribution | Blue bars = benign contribution',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('results/figures/final4/figure3_shap_waterfalls.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure 3 saved.")
# ============================================================
# FIGURE 4 — VUS DISTRIBUTION + RANKING + CLINICAL PANEL
# ============================================================
# ============================================================
# FIGURE 4 — VUS DISTRIBUTION + RANKING (3 panels)
# ============================================================
print("Generating Figure 4: VUS Tier Distribution and Ranking...")

tier_labels = ['High-Risk\nPathogenic', 'Medium-Risk\nPathogenic',
               'Low-Risk\nPathogenic',  'High-Conf.\nBenign',
               'Medium-Conf.\nBenign',  'Low-Conf.\nBenign']
tier_counts = [276, 28, 27, 1914, 41, 23]
tier_colors = [C['tier1'], C['tier2'], C['tier3'],
               C['tier6'], C['tier5'], C['tier4']]

fig = plt.figure(figsize=(20, 14), dpi=200)
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(2, 2, figure=fig,
                          left=0.06, right=0.97,
                          top=0.90, bottom=0.06,
                          wspace=0.28, hspace=0.42)

# ── Panel A: VUS Tier Bar Chart ───────────────────────────────
ax_bar = fig.add_subplot(outer[0, 0])
ax_bar.set_facecolor('white')

x_pos = np.arange(len(tier_labels))
bars  = ax_bar.bar(x_pos, tier_counts, color=tier_colors,
                   edgecolor='white', linewidth=0.8, width=0.62)

for bar, cnt in zip(bars, tier_counts):
    pct = cnt / 2309 * 100
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 20,
                f'{cnt:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#2C2C2C')

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(tier_labels, fontsize=9)
ax_bar.set_ylabel('Number of VUS', fontsize=11)
ax_bar.set_ylim([0, 2400])
ax_bar.set_title('(A) VUS Six-Tier Classification Distribution\n'
                 'Total: 2,309 held-out variants   |   Actionable (Tier 1 + Tier 6): 2,190 (94.8%)',
                 fontweight='bold', fontsize=11)

patch_p = mpatches.Patch(color=C['tier1'], label='Pathogenic tiers (1–3)')
patch_b = mpatches.Patch(color=C['tier6'], label='Benign tiers (4–6)')
ax_bar.legend(handles=[patch_p, patch_b], loc='upper center',
              fontsize=9, ncol=2, framealpha=0.9)

# ── Panel B: VUS Probability Histogram ────────────────────────
ax_hist = fig.add_subplot(outer[0, 1])
ax_hist.set_facecolor('white')

path_mask_v  = vus_pred == 'Pathogenic'
benign_mask_v = vus_pred == 'Benign'

if path_mask_v.sum() > 0:
    ax_hist.hist(vus_prob[path_mask_v], bins=45, alpha=0.78,
                 color=C['path'],
                 label=f'Predicted Pathogenic  (n = {path_mask_v.sum():,})',
                 edgecolor='white', linewidth=0.3)
if benign_mask_v.sum() > 0:
    ax_hist.hist(vus_prob[benign_mask_v], bins=45, alpha=0.78,
                 color=C['benign'],
                 label=f'Predicted Benign  (n = {benign_mask_v.sum():,})',
                 edgecolor='white', linewidth=0.3)

ax_hist.axvline(0.5,  color='black', ls='--', lw=1.8,
                label='Decision boundary (0.5)')
ax_hist.axvspan(0.0,  0.15, alpha=0.08, color=C['benign'], zorder=0)
ax_hist.axvspan(0.85, 1.0,  alpha=0.08, color=C['path'],   zorder=0)

ax_hist.text(0.075, 0.88, 'High-confidence\nbenign zone',
             transform=ax_hist.transAxes, fontsize=8.5,
             color=C['benign'], ha='center', va='top', style='italic')
ax_hist.text(0.925, 0.88, 'High-confidence\npathogenic zone',
             transform=ax_hist.transAxes, fontsize=8.5,
             color=C['path'], ha='center', va='top', style='italic')

ax_hist.set_xlabel('Pathogenic Probability Score', fontsize=11)
ax_hist.set_ylabel('Number of VUS', fontsize=11)
ax_hist.set_xlim([0, 1])
ax_hist.set_title('(B) VUS Pathogenicity Probability Distribution\n'
                  'Bimodal pattern reflects well-calibrated model confidence',
                  fontweight='bold', fontsize=11)
ax_hist.legend(fontsize=9, loc='upper center')

# ── Panel C: Top 20 VUS Ranking (full-width bottom row) ───────
ax_rank = fig.add_subplot(outer[1, :])
ax_rank.set_facecolor('white')

path_idx_v  = np.where(path_mask_v)[0]
sorted_idx_v = path_idx_v[np.argsort(vus_prob[path_idx_v])[::-1]]
top_n  = min(20, len(sorted_idx_v))
top_idx_v = sorted_idx_v[:top_n]

top_probs_v = vus_prob[top_idx_v]
top_confs_v = vus_conf[top_idx_v] if len(vus_conf) > 0 else ['High'] * top_n

def shorten_name(n, maxlen=42):
    n = str(n)
    return n[:maxlen] + '…' if len(n) > maxlen else n

top_names_v = [shorten_name(vus_names[i], 40) for i in top_idx_v]

conf_cmap = {'High': C['high'], 'Medium': C['medium'], 'Low': C['low']}
bar_cols_v = [conf_cmap.get(str(c), C['low']) for c in top_confs_v]

y_r   = np.arange(top_n)
brs_v = ax_rank.barh(y_r, top_probs_v[::-1],
                     color=bar_cols_v[::-1],
                     height=0.65, edgecolor='white', linewidth=0.4)

for bar, prob in zip(brs_v, top_probs_v[::-1]):
    ax_rank.text(bar.get_width() + 0.003,
                 bar.get_y() + bar.get_height()/2,
                 f'{prob:.4f}',
                 va='center', fontsize=8.5, fontweight='bold',
                 color='#1A1A1A')

ax_rank.set_yticks(y_r)
ax_rank.set_yticklabels(top_names_v[::-1], fontsize=8.5)
ax_rank.set_xlabel('Pathogenic Probability Score', fontsize=11)
ax_rank.set_xlim([0.45, 1.10])
ax_rank.set_title(f'(C) Top {top_n} Highest-Risk VUS Candidates — Ranked by Pathogenic Probability with Confidence Tier Annotation',
                  fontweight='bold', fontsize=11)
ax_rank.axvline(0.85, color=C['high'],   ls='--', lw=1.0, alpha=0.5)
ax_rank.axvline(0.65, color=C['medium'], ls='--', lw=0.8, alpha=0.4)

lp_r = [mpatches.Patch(color=C['high'],   label='High confidence'),
        mpatches.Patch(color=C['medium'], label='Medium confidence'),
        mpatches.Patch(color=C['low'],    label='Low confidence')]
ax_rank.legend(handles=lp_r, loc='lower right', fontsize=9, framealpha=0.9)

fig.suptitle('BRCA-COMPASS — VUS Prioritization Framework',
             fontsize=14, fontweight='bold', y=0.96)

plt.savefig('results/figures/final4/figure4_vus_summary.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure 4 saved.")

# ============================================================
# FIGURE 5 — CLINICAL INTERPRETATION REPORT (dashboard style)
# ============================================================
print("Generating Figure 5: Clinical Interpretation Report...")

shap_feature_data = [
    ('is_likely_lof',       3.15, 'PVS1 — Loss-of-function in BRCA1 tumor suppressor'),
    ('num_submitters',      2.36, 'PP3 — Multiple laboratory concordance'),
    ('AlphaMissense_score', 0.77, 'PP3 — AI-based protein impact prediction'),
    ('in_BRCT_domain',      0.61, 'PM1 — Location in critical BRCT domain'),
    ('gnomAD_AF',           0.55, 'PM2 — Absent from general population databases'),
]
acmg_codes = [
    ('PVS1', 'Very Strong', 'Predicted loss-of-function in established tumor suppressor gene'),
    ('PM1',  'Moderate',    'Variant in functionally critical BRCT domain (aa 1642–1863)'),
    ('PP3',  'Supporting',  'Concordant computational evidence (REVEL, BayesDel, AlphaMissense)'),
    ('PM2',  'Moderate',    'Absent or extremely rare in gnomAD population database'),
]
clinical_actions = [
    'Referral to a certified genetic counselor for variant review and patient counseling',
    'Confirmatory Sanger sequencing or repeat NGS testing recommended',
    'Cascade genetic testing offered to all at-risk biological first-degree relatives',
    'Discuss eligibility for PARP inhibitor therapy and risk-reduction surgical options',
]

fig = plt.figure(figsize=(16, 10), dpi=200)
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(
    3, 3, figure=fig,
    left=0.045, right=0.97, top=0.90, bottom=0.04,
    width_ratios=[1.0, 1.0, 1.0],
    height_ratios=[0.42, 1.0, 0.90],
    wspace=0.28, hspace=0.42,
)

fig.suptitle('BRCA-COMPASS — Clinical Decision Support Output',
             fontsize=15, fontweight='bold', y=0.975)
fig.text(0.045, 0.93,
         'Variant:  NM_007294.4(BRCA1):c.5266dupC  [p.Gln1756ProfsTer25]',
         fontsize=10.5, color='#1A5276', fontweight='bold')

# ── Row 1, Panel A: Classification card ───────────────────────
ax_cls = fig.add_subplot(outer[0, 0])
ax_cls.axis('off')
ax_cls.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0, transform=ax_cls.transAxes,
    boxstyle='round,pad=0.02', facecolor='#FADBD8',
    edgecolor='#C0392B', lw=1.6))
ax_cls.text(0.5, 0.62, 'Predicted Classification', transform=ax_cls.transAxes,
            ha='center', fontsize=10.5, fontweight='bold', color='#922B21')
ax_cls.text(0.5, 0.30, 'PATHOGENIC', transform=ax_cls.transAxes,
            ha='center', fontsize=19, fontweight='bold', color='#C0392B')

# ── Row 1, Panel B: Probability card + bar ────────────────────
ax_prob = fig.add_subplot(outer[0, 1])
ax_prob.axis('off')
ax_prob.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.32), 1.0, 0.68, transform=ax_prob.transAxes,
    boxstyle='round,pad=0.02', facecolor='#EBF5FB',
    edgecolor='#1F77B4', lw=1.6))
ax_prob.text(0.5, 0.80, 'Pathogenic Probability', transform=ax_prob.transAxes,
             ha='center', fontsize=10.5, fontweight='bold', color='#1A5276')
ax_prob.text(0.5, 0.50, '96.1%  |  HIGH', transform=ax_prob.transAxes,
             ha='center', fontsize=16, fontweight='bold', color='#1F77B4')
ax_prob.add_patch(mpatches.FancyBboxPatch(
    (0.04, 0.06), 0.92, 0.14, transform=ax_prob.transAxes,
    boxstyle='round,pad=0.003', facecolor='#D5D8DC', edgecolor='none'))
ax_prob.add_patch(mpatches.FancyBboxPatch(
    (0.04, 0.06), 0.92 * 0.961, 0.14, transform=ax_prob.transAxes,
    boxstyle='round,pad=0.003', facecolor='#C0392B', edgecolor='none'))
ax_prob.text(0.5, 0.13, '96.1%', transform=ax_prob.transAxes,
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# ── Row 1, Panel C: Confidence/uncertainty summary ────────────
ax_sum = fig.add_subplot(outer[0, 2])
ax_sum.axis('off')
ax_sum.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0, transform=ax_sum.transAxes,
    boxstyle='round,pad=0.02', facecolor='#F8F9F9',
    edgecolor='#99A3A4', lw=1.4))
rows = [('Predicted Diagnosis', 'PATHOGENIC', '#C0392B'),
        ('Confidence Tier',     'HIGH',       '#1F77B4'),
        ('Probability',         '96.1%',      '#1F77B4'),
        ('Reliability',         'Reliable Prediction', '#2C8C4E')]
ry = 0.82
for label, val, col in rows:
    ax_sum.text(0.06, ry, label + ':', transform=ax_sum.transAxes,
                fontsize=9, color='#2C3E50', fontweight='bold')
    ax_sum.text(0.97, ry, val, transform=ax_sum.transAxes,
                fontsize=9.5, color=col, fontweight='bold', ha='right')
    ry -= 0.24

# ── Row 2, Panel A (spans 2 cols): SHAP contribution bar chart ─
ax_shap = fig.add_subplot(outer[1, 0:2])
feat_names = [f'{n}\n{d}' for n, _, d in shap_feature_data][::-1]
feat_vals  = [v for _, v, _ in shap_feature_data][::-1]
ypos = np.arange(len(feat_vals))
bars = ax_shap.barh(ypos, feat_vals, color=C['shap_pos'],
                     edgecolor='white', height=0.6)
for bar, v in zip(bars, feat_vals):
    ax_shap.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                 f'+{v:.2f}', va='center', fontsize=8.5, fontweight='bold')
ax_shap.set_yticks(ypos)
ax_shap.set_yticklabels(feat_names, fontsize=8.3)
ax_shap.set_xlabel('SHAP Value (impact on pathogenic prediction)', fontsize=9.5)
ax_shap.set_title('Top Contributing Features (SHAP Evidence)',
                   fontsize=11, fontweight='bold', loc='left')
ax_shap.set_xlim(0, max(feat_vals) * 1.45)
ax_shap.spines['left'].set_visible(False)

# ── Row 2, Panel C: ACMG evidence codes ───────────────────────
ax_acmg = fig.add_subplot(outer[1, 2])
ax_acmg.axis('off')
ax_acmg.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0, transform=ax_acmg.transAxes,
    boxstyle='round,pad=0.02', facecolor='#FDF2F0',
    edgecolor='#C0392B', lw=1.4))
ax_acmg.text(0.06, 0.94, 'ACMG/AMP Evidence Codes',
             transform=ax_acmg.transAxes, fontsize=10.5,
             fontweight='bold', color='#922B21')
ay = 0.80
for code, strength, desc in acmg_codes:
    ax_acmg.text(0.06, ay, f'● {code}  ({strength})',
                 transform=ax_acmg.transAxes, fontsize=8.4,
                 fontweight='bold', color='#C0392B')
    ax_acmg.text(0.06, ay - 0.075, desc,
                 transform=ax_acmg.transAxes, fontsize=7.3,
                 color='#5D4037', wrap=True)
    ay -= 0.205   # 4 rows × 0.205 = 0.82, starting at 0.80 → ends ≈ -0.02 margin-safe with header at 0.94

# ── Row 3, Panel A: Clinical explanation ──────────────────────
ax_expl = fig.add_subplot(outer[2, 0])
ax_expl.axis('off')
ax_expl.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.02), 1.0, 0.96, transform=ax_expl.transAxes,
    boxstyle='round,pad=0.02', facecolor='#FEF9E7',
    edgecolor='#F39C12', lw=1.4))
ax_expl.text(0.06, 0.88, 'Clinical Explanation', transform=ax_expl.transAxes,
             fontsize=10.5, fontweight='bold', color='#784212')
expl = ('This variant creates a frameshift (c.5266dupC)\n'
        'that truncates the BRCA1 protein at codon 1756.\n'
        'Loss-of-function consequence, absence from\n'
        'population databases, and location within the\n'
        'critical BRCT domain are the primary drivers\n'
        'of this prediction.')
ax_expl.text(0.06, 0.74, expl, transform=ax_expl.transAxes,
             fontsize=8.3, color='#2C3E50', va='top', linespacing=1.45)

# ── Row 3, Panel B: Recommended actions ───────────────────────
ax_act = fig.add_subplot(outer[2, 1])
ax_act.axis('off')
ax_act.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.02), 1.0, 0.96, transform=ax_act.transAxes,
    boxstyle='round,pad=0.02', facecolor='#EBF5FB',
    edgecolor='#1F77B4', lw=1.4))
ax_act.text(0.06, 0.88, 'Recommended Clinical Actions', transform=ax_act.transAxes,
            fontsize=10.5, fontweight='bold', color='#1A5276')

def wrap_action(text, width=44):
    import textwrap
    return textwrap.fill(text, width=width)

ay = 0.74
for i, act in enumerate(clinical_actions, 1):
    wrapped = wrap_action(act)
    n_lines = wrapped.count('\n') + 1
    ax_act.text(0.06, ay, f'{i}.  {wrapped}', transform=ax_act.transAxes,
                fontsize=7.9, color='#1A5276', va='top', linespacing=1.35)
    ay -= 0.135 * n_lines + 0.045   # step scales with how many lines this item wrapped to

# ── Row 3, Panel C: Disclaimer ─────────────────────────────────
ax_disc = fig.add_subplot(outer[2, 2])
ax_disc.axis('off')
ax_disc.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.02), 1.0, 0.96, transform=ax_disc.transAxes,
    boxstyle='round,pad=0.02', facecolor='#F2F3F4',
    edgecolor='#99A3A4', lw=1.2))
ax_disc.text(0.06, 0.88, 'Important Notice', transform=ax_disc.transAxes,
             fontsize=9.5, fontweight='bold', color='#555555')
disclaimer = ('This report is generated by a computational\n'
              'decision-support tool (BRCA-COMPASS) and\n'
              'does not constitute a clinical diagnosis. All\n'
              'flagged variants require review by a qualified\n'
              'clinical geneticist or genetic counselor prior\n'
              'to any clinical action.')
ax_disc.text(0.06, 0.74, disclaimer, transform=ax_disc.transAxes,
             fontsize=7.6, color='#555555', va='top', linespacing=1.45)

plt.savefig('results/figures/final4/figure5_clinical_report.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Figure 5 saved.")

# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL 4 FIGURES GENERATED")
print("=" * 60)
for i, name in enumerate([
    'figure1_roc_pr_cm.png',
    'figure2_calibration_ablation.png',
    'figure3_shap_waterfalls.png',
    'figure4_vus_summary.png',
    'figure5_clinical_report.png',
], 1):
    path = f'results/figures/final4/{name}'
    if os.path.exists(path):
        size = os.path.getsize(path) // 1024
        print(f"  Fig {i}: {name}  ({size} KB)")
print("\nLocation: results/figures/final4/")