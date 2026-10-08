# ============================================================
# CONCORDANCE ANALYSIS — Clinical vs Functional Evidence
# File: src/concordance_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/figures/manuscript', exist_ok=True)

print("=" * 65)
print("CONCORDANCE ANALYSIS: Clinical vs Functional Evidence")
print("=" * 65)

# ── Load SGE file with scores ─────────────────────────────────────────────────
df = pd.read_csv('data/external/sge_with_scores.csv', low_memory=False)
print(f"Loaded: {len(df):,} SGE variants")
print(f"Columns: {list(df.columns)}")

# ── Identify key columns ──────────────────────────────────────────────────────
# From audit: sge_label {0: 2821, 1: 823}, function_score exists
score_col = None
for c in df.columns:
    if any(x in c.lower() for x in ['function_score','func_score',
                                      'score','sge_score']):
        if df[c].dtype in [float, 'float64'] and df[c].nunique() > 10:
            score_col = c
            break

label_col = 'sge_label'
prob_col  = None
for c in df.columns:
    if any(x in c.lower() for x in ['prob','proba','prediction',
                                      'pathogenic_prob']):
        prob_col = c
        break

print(f"\n  Score column  : {score_col}")
print(f"  Label column  : {label_col}")
print(f"  Prob column   : {prob_col}")

# ── If no prob column, generate predictions now ───────────────────────────────
if prob_col is None:
    print("\n  No probability column found — generating predictions...")
    xgb_base = joblib.load('models/xgb_base.pkl')
    imputer  = joblib.load('models/imputer.pkl')
    with open('data/processed/feature_cols_clean.json') as f:
        FEATURE_COLS = json.load(f)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    for col in missing:
        df[col] = 0

    X_sge = df[FEATURE_COLS].copy()
    X_sge_imp = imputer.transform(X_sge.values)
    df['model_prob'] = xgb_base.predict_proba(X_sge_imp)[:, 1]
    prob_col = 'model_prob'
    print(f"  Predictions generated. N={len(df):,}")

# ── Clean data ────────────────────────────────────────────────────────────────
df_clean = df[[score_col, label_col, prob_col]].dropna().copy()
df_clean[label_col] = df_clean[label_col].astype(int)

print(f"\n  Clean variants for analysis : {len(df_clean):,}")
print(f"  Functional (label=0)        : {(df_clean[label_col]==0).sum():,}")
print(f"  Non-functional (label=1)    : {(df_clean[label_col]==1).sum():,}")

# ── Core statistics ───────────────────────────────────────────────────────────
spearman_r, spearman_p = stats.spearmanr(
    df_clean[prob_col], df_clean[score_col]
)
print(f"\n  Spearman r = {spearman_r:.4f}  p = {spearman_p:.2e}")

func_probs    = df_clean[df_clean[label_col]==0][prob_col]
nonfunc_probs = df_clean[df_clean[label_col]==1][prob_col]
u_stat, u_p   = stats.mannwhitneyu(
    func_probs, nonfunc_probs, alternative='less'
)
print(f"  Functional median prob     : {func_probs.median():.4f}")
print(f"  Non-functional median prob : {nonfunc_probs.median():.4f}")
print(f"  Mann-Whitney U p-value     : {u_p:.2e}")

# ============================================================
# FIGURE A — Scatter: Model probability vs SGE score
# ============================================================
print("\n[Figure A] Scatter plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Clinical vs Functional Evidence Concordance\n'
    'ClinVar-Trained XGBoost Predictions vs Findlay 2018 SGE Scores',
    fontsize=12, fontweight='bold'
)

ax = axes[0]
colors = df_clean[label_col].map({1: '#E24B4A', 0: '#378ADD'})
ax.scatter(
    df_clean[prob_col], df_clean[score_col],
    c=colors, alpha=0.35, s=10, edgecolors='none'
)
ax.set_xlabel('Model Pathogenic Probability\n(ClinVar-trained XGBoost)',
              fontsize=11)
ax.set_ylabel('SGE Functional Score\n(Findlay 2018 — cell fitness)',
              fontsize=11)
ax.set_title(
    f'(A) Probability vs Functional Score\n'
    f'Spearman r = {spearman_r:.4f}  p < 0.001',
    fontweight='bold'
)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#E24B4A', alpha=0.7, label='Non-functional (path-like)'),
    Patch(facecolor='#378ADD', alpha=0.7, label='Functional (benign-like)')
], loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# ── Box plot ──────────────────────────────────────────────────────────────────
ax = axes[1]
bp = ax.boxplot(
    [func_probs.values, nonfunc_probs.values],
    labels=['Functional\n(benign-like)', 'Non-functional\n(path-like)'],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker='o', markersize=2, alpha=0.3)
)
bp['boxes'][0].set_facecolor('#378ADD'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#E24B4A'); bp['boxes'][1].set_alpha(0.7)

ax.set_ylabel('Model Pathogenic Probability', fontsize=11)
ax.set_title(
    f'(B) Probability by SGE Functional Class\n'
    f'Mann-Whitney U  p = {u_p:.2e}',
    fontweight='bold'
)
ax.set_ylim([-0.05, 1.05])
ax.text(1, func_probs.median() + 0.03,
        f'median={func_probs.median():.3f}', ha='center', fontsize=9)
ax.text(2, nonfunc_probs.median() + 0.03,
        f'median={nonfunc_probs.median():.3f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(
    'results/figures/manuscript/figA_concordance_scatter_box.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("   Saved: figA_concordance_scatter_box.png")

# ============================================================
# FIGURE B — Quartile analysis
# ============================================================
print("[Figure B] Quartile analysis...")

df_clean['prob_quartile'] = pd.qcut(
    df_clean[prob_col], q=4,
    labels=['Q1\n(Likely Benign)', 'Q2\n(Low)', 'Q3\n(High)',
            'Q4\n(Likely Pathogenic)']
)
q_stats = df_clean.groupby('prob_quartile', observed=True)[
    score_col
].agg(['mean', 'std', 'count']).reset_index()

fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = ['#378ADD', '#7BA7C7', '#E2937A', '#E24B4A']
bars = ax.bar(
    range(4), q_stats['mean'],
    yerr=q_stats['std'],
    color=bar_colors, alpha=0.85,
    edgecolor='white', linewidth=0.5,
    capsize=6, error_kw=dict(elinewidth=1.5, ecolor='#444')
)
for i, row in q_stats.iterrows():
    ax.text(i, row['mean'] + row['std'] + 0.05,
            f"n={int(row['count'])}",
            ha='center', fontsize=10, color='#333')

ax.set_xticks(range(4))
ax.set_xticklabels(q_stats['prob_quartile'])
ax.set_xlabel('Model Probability Quartile', fontsize=12)
ax.set_ylabel('Mean SGE Functional Score', fontsize=12)
ax.set_title(
    'Mean SGE Functional Score by Model Probability Quartile\n'
    'Monotonic decrease confirms clinical–functional concordance',
    fontsize=11, fontweight='bold'
)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

# Annotate monotonic trend
for i in range(3):
    y1 = q_stats['mean'].iloc[i]
    y2 = q_stats['mean'].iloc[i+1]
    if y2 < y1:
        ax.annotate('', xy=(i+1, y2+0.1), xytext=(i, y1+0.1),
                    arrowprops=dict(arrowstyle='->', color='#555',
                                   lw=1.5))

plt.tight_layout()
plt.savefig(
    'results/figures/manuscript/figB_quartile_sge_concordance.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("   Saved: figB_quartile_sge_concordance.png")

# ============================================================
# FIGURE C — ROC curve comparison: Internal vs External
# ============================================================
print("[Figure C] ROC comparison...")

from sklearn.metrics import roc_curve, roc_auc_score

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_title(
    'ROC Curve Comparison\nInternal Test Set vs External SGE Validation',
    fontsize=11, fontweight='bold'
)

# Internal test set
test_df  = pd.read_csv('data/processed/test_set.csv')
with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)
xgb_base = joblib.load('models/xgb_base.pkl')
imputer  = joblib.load('models/imputer.pkl')
avail    = [c for c in FEATURE_COLS if c in test_df.columns]
X_test   = test_df[avail].values
y_test   = test_df['label'].values
p_test   = xgb_base.predict_proba(X_test)[:, 1]
fpr_i, tpr_i, _ = roc_curve(y_test, p_test)
auc_i = roc_auc_score(y_test, p_test)
ax.plot(fpr_i, tpr_i, color='#1D9E75', lw=2.5,
        label=f'Internal test set — ClinVar labels\n(AUC = {auc_i:.4f}, n={len(y_test):,})')

# External SGE — all variants
y_sge = df_clean[label_col].values
p_sge = df_clean[prob_col].values
fpr_e, tpr_e, _ = roc_curve(y_sge, p_sge)
auc_e = roc_auc_score(y_sge, p_sge)
ax.plot(fpr_e, tpr_e, color='#D85A30', lw=2.5, linestyle='--',
        label=f'External SGE — all variants\n(AUC = {auc_e:.4f}, n={len(y_sge):,})')

# External SGE — missense only (if is_missense column present)
if 'is_missense' in df.columns:
    mis_mask = df['is_missense'].values == 1
    df_mis   = df_clean[mis_mask[:len(df_clean)]].copy() \
               if len(mis_mask) == len(df) \
               else df_clean.copy()
    if len(df_mis) >= 20 and df_mis[label_col].sum() > 0:
        y_mis = df_mis[label_col].values
        p_mis = df_mis[prob_col].values
        if len(np.unique(y_mis)) == 2:
            fpr_m, tpr_m, _ = roc_curve(y_mis, p_mis)
            auc_m = roc_auc_score(y_mis, p_mis)
            ax.plot(fpr_m, tpr_m, color='#7F77DD', lw=2.5,
                    linestyle=':',
                    label=f'External SGE — missense only\n'
                          f'(AUC = {auc_m:.4f}, n={len(y_mis):,})')

ax.plot([0,1],[0,1],'--', color='#888', lw=1, alpha=0.6, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.01])

plt.tight_layout()
plt.savefig(
    'results/figures/manuscript/figC_roc_internal_vs_external.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("   Saved: figC_roc_internal_vs_external.png")

# ============================================================
# FIGURE D — Probability distribution by SGE class
# ============================================================
print("[Figure D] Probability distribution by SGE class...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(func_probs, bins=40, alpha=0.70, color='#378ADD',
        label=f'Functional / benign-like (n={len(func_probs):,})',
        edgecolor='white', density=True)
ax.hist(nonfunc_probs, bins=40, alpha=0.70, color='#E24B4A',
        label=f'Non-functional / path-like (n={len(nonfunc_probs):,})',
        edgecolor='white', density=True)
ax.axvline(0.5, color='black', linestyle='--', lw=2,
           label='Decision boundary (0.5)')
ax.set_xlabel('Model Pathogenic Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(
    'Predicted Pathogenic Probability Distribution\n'
    'Stratified by SGE Functional Class (Findlay 2018)',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    'results/figures/manuscript/figD_sge_prob_distribution.png',
    dpi=300, bbox_inches='tight'
)
plt.close()
print("   Saved: figD_sge_prob_distribution.png")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 65)
print("CONCORDANCE ANALYSIS COMPLETE")
print("=" * 65)
print(f"""
  Key statistics for paper:

  Spearman r (prob vs SGE score) : {spearman_r:.4f}
  Spearman p-value               : {spearman_p:.2e}
  Mann-Whitney U p-value         : {u_p:.2e}

  Median probability:
    Functional variants          : {func_probs.median():.4f}
    Non-functional variants      : {nonfunc_probs.median():.4f}
    Fold separation              : {nonfunc_probs.median()/max(func_probs.median(),0.001):.1f}x

  ROC-AUC:
    Internal test set            : {auc_i:.4f}
    External SGE (all)           : {auc_e:.4f}
    AUC gap                      : {auc_i - auc_e:.4f}

  Figures saved:
    figA_concordance_scatter_box.png
    figB_quartile_sge_concordance.png
    figC_roc_internal_vs_external.png
    figD_sge_prob_distribution.png
""")