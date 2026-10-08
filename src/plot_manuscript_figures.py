# ============================================================
# MANUSCRIPT FIGURES — Complete Publication-Ready Plots
# File: src/plot_manuscript_figures.py
#
# Loads from SAVED split files (train_set.csv, val_set.csv,
# test_set.csv) — does NOT recompute splits from raw data.
# Uses your 6,009-variant expanded dataset correctly.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

os.makedirs('results/figures/manuscript', exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
})

COLORS = {
    'train'      : '#1D9E75',
    'val'        : '#EF9F27',
    'test'       : '#D85A30',
    'missense'   : '#7F77DD',
    'neutral'    : '#888780',
    'pathogenic' : '#E24B4A',
    'benign'     : '#378ADD',
}

print("=" * 65)
print("GENERATING MANUSCRIPT FIGURES (6,009-variant dataset)")
print("=" * 65)

# ============================================================
# LOAD SAVED SPLITS — do NOT recompute
# ============================================================
LABEL = 'label'

print("\n  Loading saved splits from data/processed/...")
train_df = pd.read_csv('data/processed/train_set.csv')
val_df   = pd.read_csv('data/processed/val_set.csv')
test_df  = pd.read_csv('data/processed/test_set.csv')

print(f"    Train : {len(train_df):,} variants")
print(f"    Val   : {len(val_df):,} variants")
print(f"    Test  : {len(test_df):,} variants")
print(f"    Total : {len(train_df)+len(val_df)+len(test_df):,} variants")

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

# Extract features and labels
available = [c for c in FEATURE_COLS
             if c in train_df.columns
             and c in val_df.columns
             and c in test_df.columns]
print(f"    Features available : {len(available)}")

X_train_imp = train_df[available].copy()
X_val_imp   = val_df[available].copy()
X_test_imp  = test_df[available].copy()

y_train = train_df[LABEL].astype(int)
y_val   = val_df[LABEL].astype(int)
y_test  = test_df[LABEL].astype(int)

# ── Load model ────────────────────────────────────────────────────────────────
xgb_base  = joblib.load('models/xgb_base.pkl')
# Use platt_scaler if it exists, otherwise just use xgb_base directly
import os
if os.path.exists('models/platt_scaler.pkl'):
    platt = joblib.load('models/platt_scaler.pkl')
    def get_cal_proba(X):
        raw = xgb_base.predict_proba(X)[:, 1].reshape(-1, 1)
        return platt.predict_proba(raw)[:, 1]
else:
    def get_cal_proba(X):
        return xgb_base.predict_proba(X)[:, 1]
imputer   = joblib.load('models/imputer.pkl')
OPT_THRESH = joblib.load('models/threshold.pkl')['threshold']

print(f"\n  Model loaded     : {type(xgb_base).__name__}")
print(f"  Threshold        : {OPT_THRESH:.4f}")

# ── Predictions ───────────────────────────────────────────────────────────────
train_proba = xgb_base.predict_proba(X_train_imp.values)[:, 1]
val_proba   = xgb_base.predict_proba(X_val_imp.values)[:, 1]
test_proba  = xgb_base.predict_proba(X_test_imp.values)[:, 1]

# ── Missense mask on test ─────────────────────────────────────────────────────
# Load scored file to get consequence annotations for test rows
df_scored = pd.read_csv(
    'data/processed/brca1_features_scored.csv', low_memory=False
)

# The test set was split from the full scored dataset
# Reconstruct test missense mask by matching label+feature fingerprint
# Simplest approach: use is_missense column from test_df if present
if 'is_missense' in test_df.columns:
    mis_mask = test_df['is_missense'].values == 1
else:
    # Fallback: no missense column in test file
    mis_mask = np.zeros(len(test_df), dtype=bool)
    print("  Warning: is_missense not in test_set.csv — missense plots skipped")

y_mis = y_test.values[mis_mask]
p_mis = test_proba[mis_mask]

print(f"\n  Missense in test : {mis_mask.sum():,} variants")
print(f"    Pathogenic     : {y_mis.sum():,}")
print(f"    Benign         : {(y_mis==0).sum():,}")

# ── Helper: ECE ───────────────────────────────────────────────────────────────
def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum()/n) * abs(
            y_true[mask].mean() - y_prob[mask].mean()
        )
    return round(float(ece), 4)

# ============================================================
# FIGURE 1 — ROC Curves
# ============================================================
print("\n[Figure 1] ROC Curves...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    'Figure 1. ROC-AUC Performance Across Data Splits\n'
    f'Train n={len(y_train):,} | Val n={len(y_val):,} | '
    f'Test n={len(y_test):,}',
    fontsize=12, fontweight='bold', y=1.02
)

ax = axes[0]
ax.set_title('(A) All Variant Types', fontweight='bold')
for split, y_s, p_s, color, lw in [
    ('Train',      y_train.values, train_proba, COLORS['train'], 1.8),
    ('Validation', y_val.values,   val_proba,   COLORS['val'],   1.8),
    ('Test',       y_test.values,  test_proba,  COLORS['test'],  2.5),
]:
    fpr, tpr, _ = roc_curve(y_s, p_s)
    auc_v = roc_auc_score(y_s, p_s)
    ax.plot(fpr, tpr, color=color, lw=lw,
            label=f'{split} (AUC = {auc_v:.4f})')
ax.plot([0,1],[0,1],'--', color=COLORS['neutral'],
        lw=1, alpha=0.6, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])

ax = axes[1]
ax.set_title('(B) Test Set — All vs Missense Only', fontweight='bold')
fpr, tpr, _ = roc_curve(y_test.values, test_proba)
auc_all = roc_auc_score(y_test.values, test_proba)
ax.plot(fpr, tpr, color=COLORS['test'], lw=2.5,
        label=f'All variants (AUC = {auc_all:.4f})')
if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
    fpr_m, tpr_m, _ = roc_curve(y_mis, p_mis)
    auc_m = roc_auc_score(y_mis, p_mis)
    ax.plot(fpr_m, tpr_m, color=COLORS['missense'], lw=2.5,
            linestyle='--',
            label=f'Missense only (AUC = {auc_m:.4f})')
ax.plot([0,1],[0,1],'--', color=COLORS['neutral'],
        lw=1, alpha=0.6, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig1_roc_curves.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig1_roc_curves.png")

# ============================================================
# FIGURE 2 — PR Curves
# ============================================================
print("[Figure 2] PR Curves...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    'Figure 2. Precision-Recall Curves Across Data Splits\n'
    f'Train n={len(y_train):,} | Val n={len(y_val):,} | '
    f'Test n={len(y_test):,}',
    fontsize=12, fontweight='bold', y=1.02
)

ax = axes[0]
ax.set_title('(A) All Variant Types', fontweight='bold')
for split, y_s, p_s, color, lw in [
    ('Train',      y_train.values, train_proba, COLORS['train'], 1.8),
    ('Validation', y_val.values,   val_proba,   COLORS['val'],   1.8),
    ('Test',       y_test.values,  test_proba,  COLORS['test'],  2.5),
]:
    prec, rec, _ = precision_recall_curve(y_s, p_s)
    pr_v = average_precision_score(y_s, p_s)
    ax.plot(rec, prec, color=color, lw=lw,
            label=f'{split} (PR-AUC = {pr_v:.4f})')
ax.axhline(y_train.mean(), color=COLORS['neutral'], linestyle=':',
           lw=1, alpha=0.7, label=f'Baseline ({y_train.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='upper right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])

ax = axes[1]
ax.set_title('(B) Test Set — All vs Missense Only', fontweight='bold')
prec, rec, _ = precision_recall_curve(y_test.values, test_proba)
pr_all = average_precision_score(y_test.values, test_proba)
ax.plot(rec, prec, color=COLORS['test'], lw=2.5,
        label=f'All variants (PR-AUC = {pr_all:.4f})')
if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
    prec_m, rec_m, _ = precision_recall_curve(y_mis, p_mis)
    pr_m = average_precision_score(y_mis, p_mis)
    ax.plot(rec_m, prec_m, color=COLORS['missense'], lw=2.5,
            linestyle='--',
            label=f'Missense only (PR-AUC = {pr_m:.4f})')
ax.axhline(y_test.mean(), color=COLORS['neutral'], linestyle=':',
           lw=1, alpha=0.7, label=f'Baseline ({y_test.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='upper right')
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig2_pr_curves.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig2_pr_curves.png")

# ============================================================
# FIGURE 3 — Confusion Matrices
# ============================================================
print("[Figure 3] Confusion Matrices...")

n_panels = 3 if (y_mis.sum() > 0 and (y_mis==0).sum() > 0) else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))
fig.suptitle(
    'Figure 3. Confusion Matrices at Optimal Threshold '
    f'(threshold = {OPT_THRESH:.3f})',
    fontsize=12, fontweight='bold', y=1.02
)

panels = [
    (axes[0], y_val.values,  val_proba,
     f'(A) Validation Set\nn = {len(y_val):,}'),
    (axes[1], y_test.values, test_proba,
     f'(B) Test Set\nn = {len(y_test):,}'),
]
if n_panels == 3:
    panels.append((
        axes[2], y_mis, p_mis,
        f'(C) Test — Missense Only\nn = {mis_mask.sum():,}'
    ))

for ax, y_s, p_s, title in panels:
    y_pred = (p_s >= OPT_THRESH).astype(int)
    cm     = confusion_matrix(y_s, y_pred)
    acc    = accuracy_score(y_s, y_pred)
    f1_v   = f1_score(y_s, y_pred, zero_division=0)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Pred Benign', 'Pred Pathogenic'],
        yticklabels=['True Benign', 'True Pathogenic'],
        cbar=True, linewidths=0.5, linecolor='white',
        annot_kws={'size': 13, 'weight': 'bold'}
    )
    ax.set_title(f'{title}\nAcc = {acc:.4f}  F1 = {f1_v:.4f}',
                 fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig3_confusion_matrices.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig3_confusion_matrices.png")

# ============================================================
# FIGURE 4 — Grouped Metrics Bar Chart
# ============================================================
print("[Figure 4] Metrics Bar Chart...")

metrics_dict = {}
for split, y_s, p_s in [
    ('Train',      y_train.values, train_proba),
    ('Validation', y_val.values,   val_proba),
    ('Test',       y_test.values,  test_proba),
]:
    y_pred = (p_s >= OPT_THRESH).astype(int)
    metrics_dict[split] = {
        'ROC-AUC'     : roc_auc_score(y_s, p_s),
        'PR-AUC'      : average_precision_score(y_s, p_s),
        'Accuracy'    : accuracy_score(y_s, y_pred),
        'Precision'   : precision_score(y_s, y_pred, zero_division=0),
        'Recall'      : recall_score(y_s, y_pred, zero_division=0),
        'F1 Score'    : f1_score(y_s, y_pred, zero_division=0),
        'MCC'         : matthews_corrcoef(y_s, y_pred),
        'Balanced Acc': balanced_accuracy_score(y_s, y_pred),
    }

metric_names = list(metrics_dict['Train'].keys())
x     = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title(
    'Figure 4. Performance Metrics Across Data Splits\n'
    f'Train n={len(y_train):,} | Val n={len(y_val):,} | '
    f'Test n={len(y_test):,}',
    fontsize=12, fontweight='bold'
)

for i, (split, color) in enumerate([
    ('Train',      COLORS['train']),
    ('Validation', COLORS['val']),
    ('Test',       COLORS['test']),
]):
    vals = [metrics_dict[split][m] for m in metric_names]
    bars = ax.bar(x + i*width, vals, width, label=split,
                  color=color, alpha=0.88,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=7.5, rotation=90
        )

ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, rotation=20, ha='right')
ax.set_ylabel('Score')
ax.set_ylim([0.88, 1.025])
ax.legend(loc='lower right')
ax.axhline(0.95, color='red', linestyle=':', lw=1.2, alpha=0.5,
           label='0.95 target')

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig4_metrics_bar.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig4_metrics_bar.png")

# ============================================================
# FIGURE 5 — Calibration Curves
# ============================================================
print("[Figure 5] Calibration Curves...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 5. Probability Calibration',
             fontsize=12, fontweight='bold', y=1.02)

for ax, y_s, p_s, split, color, label in [
    (axes[0], y_train.values, train_proba, 'Train',      COLORS['train'], 'A'),
    (axes[1], y_val.values,   val_proba,   'Validation', COLORS['val'],   'B'),
    (axes[2], y_test.values,  test_proba,  'Test',       COLORS['test'],  'C'),
]:
    frac_pos, mean_pred = calibration_curve(y_s, p_s, n_bins=10)
    brier = round(brier_score_loss(y_s, p_s), 4)
    ece   = ece_score(y_s, p_s)

    ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=2,
            markersize=5, label='Model')
    ax.plot([0,1],[0,1],'--', color=COLORS['neutral'],
            lw=1.5, label='Perfect calibration')
    ax.fill_between(mean_pred, frac_pos, mean_pred,
                    alpha=0.12, color=color)
    ax.set_title(
        f'({label}) {split} Set\nBrier = {brier} | ECE = {ece}',
        fontweight='bold'
    )
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(fontsize=9)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig5_calibration.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig5_calibration.png")

# ============================================================
# FIGURE 6 — Probability Distributions
# ============================================================
print("[Figure 6] Probability Distributions...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    'Figure 6. Predicted Pathogenicity Probability Distributions',
    fontsize=12, fontweight='bold', y=1.02
)

for ax, y_s, p_s, title in [
    (axes[0], y_val.values,  val_proba,  f'(A) Validation Set (n={len(y_val):,})'),
    (axes[1], y_test.values, test_proba, f'(B) Test Set (n={len(y_test):,})'),
]:
    ax.hist(p_s[y_s==0], bins=40, alpha=0.70,
            color=COLORS['benign'], label='Benign',
            edgecolor='white', density=True)
    ax.hist(p_s[y_s==1], bins=40, alpha=0.70,
            color=COLORS['pathogenic'], label='Pathogenic',
            edgecolor='white', density=True)
    ax.axvline(OPT_THRESH, color='black', linestyle='--', lw=2,
               label=f'Threshold ({OPT_THRESH:.3f})')
    ax.set_xlabel('Predicted Pathogenic Probability')
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig6_probability_distributions.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig6_probability_distributions.png")

# ============================================================
# FIGURE 7 — Overfitting Analysis
# ============================================================
print("[Figure 7] Overfitting Analysis...")

tr_auc  = roc_auc_score(y_train.values, train_proba)
va_auc  = roc_auc_score(y_val.values,   val_proba)
te_auc  = roc_auc_score(y_test.values,  test_proba)
gap_tvl = round(tr_auc - va_auc, 4)
gap_tte = round(tr_auc - te_auc, 4)

splits      = ['Train', 'Validation', 'Test']
roc_vals    = [tr_auc, va_auc, te_auc]
pr_vals     = [
    average_precision_score(y_train.values, train_proba),
    average_precision_score(y_val.values,   val_proba),
    average_precision_score(y_test.values,  test_proba),
]
f1_vals = [
    f1_score(y_train.values, (train_proba>=OPT_THRESH).astype(int), zero_division=0),
    f1_score(y_val.values,   (val_proba  >=OPT_THRESH).astype(int), zero_division=0),
    f1_score(y_test.values,  (test_proba >=OPT_THRESH).astype(int), zero_division=0),
]
mcc_vals = [
    matthews_corrcoef(y_train.values, (train_proba>=OPT_THRESH).astype(int)),
    matthews_corrcoef(y_val.values,   (val_proba  >=OPT_THRESH).astype(int)),
    matthews_corrcoef(y_test.values,  (test_proba >=OPT_THRESH).astype(int)),
]

status = ('Excellent' if max(gap_tvl, gap_tte) < 0.03
          else 'Acceptable' if max(gap_tvl, gap_tte) <= 0.05
          else 'Review regularization')

fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
fig.suptitle(
    f'Figure 7. Overfitting Analysis — Key Metrics Across Splits\n'
    f'Train-Val Gap: {gap_tvl:.4f}  |  Train-Test Gap: {gap_tte:.4f}  |  '
    f'Status: {status}',
    fontsize=11, fontweight='bold', y=1.02
)

colors_splits = [COLORS['train'], COLORS['val'], COLORS['test']]

for ax, vals, title, ylim_low in [
    (axes[0], roc_vals,  'ROC-AUC',  0.96),
    (axes[1], pr_vals,   'PR-AUC',   0.96),
    (axes[2], f1_vals,   'F1 Score', 0.94),
    (axes[3], mcc_vals,  'MCC',      0.90),
]:
    bars = ax.bar(splits, vals, color=colors_splits,
                  edgecolor='white', linewidth=0.5, alpha=0.88)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() - 0.005,
            f'{val:.4f}', ha='center', va='top',
            fontsize=10, fontweight='bold', color='white'
        )
    ax.set_title(title, fontweight='bold')
    ax.set_ylim([ylim_low, 1.01])
    ax.set_ylabel('Score')

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig7_overfitting_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig7_overfitting_analysis.png")

# ============================================================
# FIGURE 8 — Publication Metrics Table
# ============================================================
print("[Figure 8] Publication Metrics Table...")

rows = []
for split, y_s, p_s in [
    ('Train',           y_train.values, train_proba),
    ('Validation',      y_val.values,   val_proba),
    ('Test',            y_test.values,  test_proba),
    ('Test (Missense)', y_mis,          p_mis),
]:
    if len(y_s) == 0 or y_s.sum() == 0 or (y_s==0).sum() == 0:
        continue
    y_pred = (p_s >= OPT_THRESH).astype(int)
    rows.append([
        split,
        f'{roc_auc_score(y_s, p_s):.4f}',
        f'{average_precision_score(y_s, p_s):.4f}',
        f'{accuracy_score(y_s, y_pred):.4f}',
        f'{precision_score(y_s, y_pred, zero_division=0):.4f}',
        f'{recall_score(y_s, y_pred, zero_division=0):.4f}',
        f'{f1_score(y_s, y_pred, zero_division=0):.4f}',
        f'{matthews_corrcoef(y_s, y_pred):.4f}',
        f'{balanced_accuracy_score(y_s, y_pred):.4f}',
        f'{len(y_s):,}',
    ])

col_labels = [
    'Split','ROC-AUC','PR-AUC','Accuracy',
    'Precision','Recall','F1','MCC','Bal.Acc','N'
]

fig, ax = plt.subplots(figsize=(17, 3.2))
ax.axis('off')
table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C2C2A')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.28)

row_colors = ['#F1EFE8', 'white', '#E1F5EE', '#EAF3DE']
for i in range(1, len(rows)+1):
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(row_colors[min(i-1, len(row_colors)-1)])
        table[i, j].set_height(0.22)
        if 'Test' in rows[i-1][0] and 'Missense' not in rows[i-1][0]:
            table[i, j].set_text_props(fontweight='bold')

ax.set_title(
    'Table 1. XGBoost Classifier Performance — BRCA1 Variant Pathogenicity\n'
    f'Training pool: 6,009 variants | '
    f'Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}',
    fontsize=11, fontweight='bold', pad=15
)
plt.tight_layout()
plt.savefig('results/figures/manuscript/fig8_metrics_table.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig8_metrics_table.png")

# ============================================================
# FIGURE 9 — 5-Fold CV Per-Fold Performance
# ============================================================
print("[Figure 9] Cross-Validation Fold Performance...")

# Combine train+val for CV
X_tv_imp = pd.concat([X_train_imp, X_val_imp], ignore_index=True)
y_tv_arr = pd.concat([y_train, y_val], ignore_index=True).values

print(f"  CV dataset: {len(X_tv_imp):,} variants "
      f"(train {len(X_train_imp):,} + val {len(X_val_imp):,})")

scale_pw  = float((y_tv_arr==0).sum()) / float((y_tv_arr==1).sum())
cv_skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_aucs = []
fold_f1s  = []
fold_mccs = []
fold_prs  = []

for fold_num, (tr_idx, va_idx) in enumerate(
        cv_skf.split(X_tv_imp.values, y_tv_arr), 1):

    Xf_tr = X_tv_imp.values[tr_idx]
    Xf_va = X_tv_imp.values[va_idx]
    yf_tr = y_tv_arr[tr_idx]
    yf_va = y_tv_arr[va_idx]

    spw = float((yf_tr==0).sum()) / float((yf_tr==1).sum())
    fm  = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42,
        objective='binary:logistic', verbosity=0, n_jobs=-1
    )
    fm.fit(Xf_tr, yf_tr)
    p_va      = fm.predict_proba(Xf_va)[:, 1]
    y_pred_va = (p_va >= 0.5).astype(int)

    fold_aucs.append(roc_auc_score(yf_va, p_va))
    fold_prs.append(average_precision_score(yf_va, p_va))
    fold_f1s.append(f1_score(yf_va, y_pred_va, zero_division=0))
    fold_mccs.append(matthews_corrcoef(yf_va, y_pred_va))
    print(f"  Fold {fold_num}: AUC={fold_aucs[-1]:.4f} "
          f"F1={fold_f1s[-1]:.4f} MCC={fold_mccs[-1]:.4f}")

fold_nums = np.arange(1, 6)
fold_aucs = np.array(fold_aucs)
fold_prs  = np.array(fold_prs)
fold_f1s  = np.array(fold_f1s)
fold_mccs = np.array(fold_mccs)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    'Figure 9. 5-Fold Stratified Cross-Validation\n'
    f'ROC-AUC: {fold_aucs.mean():.4f}±{fold_aucs.std():.4f}  |  '
    f'PR-AUC: {fold_prs.mean():.4f}±{fold_prs.std():.4f}  |  '
    f'F1: {fold_f1s.mean():.4f}±{fold_f1s.std():.4f}  |  '
    f'MCC: {fold_mccs.mean():.4f}±{fold_mccs.std():.4f}',
    fontsize=11, fontweight='bold', y=1.02
)

for ax, vals, metric, color, ylim_low in [
    (axes[0], fold_aucs, 'ROC-AUC',  COLORS['train'],    0.990),
    (axes[1], fold_prs,  'PR-AUC',   COLORS['val'],      0.985),
    (axes[2], fold_f1s,  'F1 Score', COLORS['test'],     0.960),
    (axes[3], fold_mccs, 'MCC',      COLORS['missense'], 0.920),
]:
    bars = ax.bar(fold_nums, vals, color=color, alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(vals.mean(), color='black', linestyle='--', lw=1.5,
               label=f'Mean = {vals.mean():.4f}')
    ax.fill_between(
        [0.4, 5.6],
        vals.mean() - vals.std(),
        vals.mean() + vals.std(),
        alpha=0.15, color=color, label=f'±1 SD = {vals.std():.4f}'
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9
        )
    ax.set_xlabel('Fold')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} per Fold', fontweight='bold')
    ax.set_xticks(fold_nums)
    ax.set_xticklabels([f'Fold {i}' for i in fold_nums])
    ax.set_ylim([ylim_low, 1.01])
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/manuscript/fig9_cv_folds.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: fig9_cv_folds.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("ALL MANUSCRIPT FIGURES COMPLETE")
print("=" * 65)
print(f"\n  Dataset confirmed: {len(y_train)+len(y_val)+len(y_test):,} variants total")
print(f"    Train : {len(y_train):,}")
print(f"    Val   : {len(y_val):,}")
print(f"    Test  : {len(y_test):,}")
print(f"\n  Figures saved to: results/figures/manuscript/")
print()
files = sorted(os.listdir('results/figures/manuscript/'))
for f in files:
    fpath = f'results/figures/manuscript/{f}'
    size  = os.path.getsize(fpath) // 1024
    print(f"    {f:<55} {size:>5} KB")

print("""
  To view in VS Code:
    Ctrl+Shift+E → results/figures/manuscript/ → click any .png
""")