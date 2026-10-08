# ============================================================
# EXTERNAL VALIDATION V2 — SGE with dbNSFP scores injected
# File: src/external_validation_v2.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from scipy.stats import spearmanr, pearsonr

print("=" * 65)
print("EXTERNAL VALIDATION V2 — SGE + dbNSFP Scores")
print("=" * 65)

# ── Load model ────────────────────────────────────────────────────────────────
xgb_base = joblib.load('models/xgb_base.pkl')
imputer  = joblib.load('models/imputer.pkl')

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

print(f"  Features : {len(FEATURE_COLS)}")
print(f"  ✓ Train-fitted imputer — will NOT be refit")

# ── Load SGE raw file ─────────────────────────────────────────────────────────
SGE_PATH = 'data/external/41586_2018_461_MOESM3_ESM.xlsx'
df_raw   = pd.read_excel(SGE_PATH, sheet_name='Sheet1',
                          header=None, skiprows=2)

cols = [
    'gene','chromosome','position_hg19','reference','alt',
    'transcript_ID','transcript_position','transcript_ref','transcript_alt',
    'transcript_variant','aa_pos','aa_ref','aa_alt','protein_variant',
    'consequence','function_score_mean','func_class','p_nonfunctional',
    'experiment',
    'raw_negative','raw_library','raw_d5r1','raw_d11r1','raw_d5r2','raw_d11r2',
    'neg_freq','library_freq','d5_freq_r1','d11_freq_r1','d5_freq_r2','d11_freq_r2',
    'd5_lib_r1','loess_fit_r1','d11_lib_raw_r1','d11_lib_loess_r1',
    'd5_lib_r2','loess_fit_r2','d11_lib_raw_r2','d11_lib_loess_r2',
    'function_score_r1','function_score_r2',
    'rna_r1','rna_r2','rna_r1_freq','rna_r2_freq',
    'rna_score_1','rna_score_2','mean_rna_score',
    'CADD_score','phyloP_mammalian','polyphen2','sift',
    'aGVGD_diff','aGVGD_class',
    'clinvar','clinvar_simple','gnomAD_AF','bravo_AF','flossies_AF',
    'WT_HAP1_score_r1','WT_HAP1_score_r2','WT_HAP1_score_mean',
    'extra1','extra2','extra3'
]
df_raw.columns = cols[:len(df_raw.columns)]

df = df_raw[
    (df_raw['gene'] == 'BRCA1') &
    (df_raw['func_class'].isin(['FUNC','LOF']))
].copy()

df['position_hg19']       = pd.to_numeric(df['position_hg19'],      errors='coerce')
df['function_score_mean'] = pd.to_numeric(df['function_score_mean'], errors='coerce')
df['aa_pos']              = pd.to_numeric(df['aa_pos'],              errors='coerce')
df['CADD_score']          = pd.to_numeric(df['CADD_score'],          errors='coerce')
df['phyloP_mammalian']    = pd.to_numeric(df['phyloP_mammalian'],    errors='coerce')
df['polyphen2']           = pd.to_numeric(df['polyphen2'],           errors='coerce')
df['sift']                = pd.to_numeric(df['sift'],                errors='coerce')
df['gnomAD_AF']           = pd.to_numeric(df['gnomAD_AF'],           errors='coerce')
df = df.dropna(subset=['position_hg19','function_score_mean']).copy()
df['sge_label'] = (df['func_class'] == 'LOF').astype(int)

HG38_OFFSET     = 1_847_983
df['position']  = df['position_hg19'] + HG38_OFFSET

print(f"\n  SGE variants: {len(df):,}")

# ── Load fetched dbNSFP scores ────────────────────────────────────────────────
scores = pd.read_csv('data/external/sge_with_scores.csv', low_memory=False)

# Merge on position + ref + alt
score_cols = [
    'BayesDel_addAF','BayesDel_noAF','REVEL_score',
    'AlphaMissense_score','DANN_score','MetaRNN_score',
    'VEST4_score','ClinPred_score'
]

# Build merge key
scores['pos_key'] = (
    scores['position_hg38'].astype(str) + ':' +
    scores['reference'].astype(str) + ':' +
    scores['alt'].astype(str)
)
df['pos_key'] = (
    df['position'].astype(str) + ':' +
    df['reference'].astype(str) + ':' +
    df['alt'].astype(str)
)

available_score_cols = [c for c in score_cols if c in scores.columns]
df = df.merge(
    scores[['pos_key'] + available_score_cols].drop_duplicates('pos_key'),
    on='pos_key', how='left'
)

print(f"\n  Score coverage after merge:")
for col in available_score_cols:
    n   = df[col].notna().sum()
    pct = n/len(df)*100
    print(f"    {col:<25} {n:>5}/{len(df):>5} ({pct:5.1f}%)")

# ── Feature engineering ───────────────────────────────────────────────────────
print(f"\n  Engineering features...")

BRCA1_START = 43_044_295
BRCA1_END   = 43_125_483
BRCA1_LEN   = BRCA1_END - BRCA1_START

df['normalized_position']   = ((df['position'] - BRCA1_START) / BRCA1_LEN).clip(0,1)
df['variant_type_encoded']  = 0
df['ref_len']               = 1
df['alt_len']               = 1
df['indel_length']          = 0
df['is_frameshift']         = 0
df['allele_length_ratio']   = 1.0

df['cons_deletion']         = 0
df['cons_duplication']      = 0
df['cons_frameshift']       = 0
df['cons_insertion']        = 0
df['cons_nonsense']         = (df['consequence'] == 'Nonsense').astype(int)
df['cons_snv']              = 1
df['cons_splice']           = df['consequence'].isin(
    ['Splice region','Canonical splice']).astype(int)
df['cons_other']            = 0

df['has_protein_change']    = df['aa_pos'].notna().astype(int)
df['is_missense']           = (df['consequence'] == 'Missense').astype(int)
df['is_nonsense']           = (df['consequence'] == 'Nonsense').astype(int)
df['is_frameshift_protein'] = 0
df['is_synonymous']         = (df['consequence'] == 'Synonymous').astype(int)
df['is_splice_protein']     = df['consequence'].isin(
    ['Splice region','Canonical splice']).astype(int)
df['aa_position']           = df['aa_pos']

def annotate_domain(aa):
    if pd.isna(aa): return 0,0,0,1,0
    aa = float(aa)
    ring = int(1    <= aa <= 109)
    brct = int(1642 <= aa <= 1863)
    cc   = int(1391 <= aa <= 1424)
    dis  = int((110 <= aa <= 1390) or (1425 <= aa <= 1641))
    return ring, brct, cc, dis, 1

dom = df['aa_position'].apply(annotate_domain)
df['in_RING_domain']        = [d[0] for d in dom]
df['in_BRCT_domain']        = [d[1] for d in dom]
df['in_coiled_coil']        = [d[2] for d in dom]
df['in_disordered_region']  = [d[3] for d in dom]
df['domain_known']          = [d[4] for d in dom]
df['in_pathogenic_hotspot'] = (
    (df['in_RING_domain']==1) | (df['in_BRCT_domain']==1)
).astype(int)

df['is_likely_lof']              = df['is_nonsense'].copy()
df['is_canonical_splice_site']   = (
    df['consequence'] == 'Canonical splice').astype(int)
df['in_exon11']                  = (
    (df['position'] >= 43_082_434) &
    (df['position'] <= 43_091_032)
).astype(int)

df['review_strength'] = 1
df['num_submitters']  = 1

# Scores from SGE file
df['CADD_phred']     = df['CADD_score']
df['phyloP17way']    = df['phyloP_mammalian']
df['Polyphen2_HDIV'] = df['polyphen2']
df['Polyphen2_HVAR'] = df['polyphen2']
df['SIFT_score']     = df['sift']

# Scores not in SGE file and not fetched
for col in ['SpliceAI_DS_max','phastCons17way','GERP_RS','MPC_score']:
    if col not in df.columns:
        df[col] = np.nan

# ── Align to model feature set ────────────────────────────────────────────────
for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0

X_sge     = df[FEATURE_COLS].copy()
y_sge     = df['sge_label'].values
X_sge_imp = imputer.transform(X_sge.values)

print(f"  ✓ Imputed with train-fitted imputer")
print(f"  ✓ Shape: {X_sge_imp.shape}")

# ── Predict ───────────────────────────────────────────────────────────────────
proba = xgb_base.predict_proba(X_sge_imp)[:, 1]
pred  = (proba >= 0.5).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
def full_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    return {
        'ROC_AUC'     : round(roc_auc_score(y_true, y_prob), 4),
        'PR_AUC'      : round(average_precision_score(y_true, y_prob), 4),
        'Accuracy'    : round(accuracy_score(y_true, y_pred), 4),
        'Precision'   : round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'      : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1'          : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC'         : round(matthews_corrcoef(y_true, y_pred), 4),
        'Balanced_Acc': round(balanced_accuracy_score(y_true, y_pred), 4),
    }

metrics_all = full_metrics(y_sge, proba)

# Missense only
mis_mask = df['consequence'].values == 'Missense'
y_mis    = y_sge[mis_mask]
p_mis    = proba[mis_mask]
metrics_mis = full_metrics(y_mis, p_mis) \
    if y_mis.sum() > 0 and (y_mis==0).sum() > 0 else None

# Correlation with SGE score
r_sp, p_sp = spearmanr(proba, -df['function_score_mean'].values)
r_pe, p_pe = pearsonr( proba, -df['function_score_mean'].values)

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"RESULTS V2 — SGE with dbNSFP Scores (n={len(y_sge):,})")
print(f"{'='*65}")

print(f"\n  Overall (P={y_sge.sum():,}, B={(y_sge==0).sum():,}):")
for k,v in metrics_all.items():
    bar = '█' * int(v*30)
    print(f"    {k:<15} : {v:.4f}  {bar}")

if metrics_mis:
    print(f"\n  Missense-only (n={mis_mask.sum():,}, "
          f"P={y_mis.sum():,}, B={(y_mis==0).sum():,}):")
    for k,v in metrics_mis.items():
        print(f"    {k:<15} : {v:.4f}")

print(f"\n  Spearman ρ : {r_sp:.4f}  (p={p_sp:.2e})")
print(f"  Pearson r  : {r_pe:.4f}  (p={p_pe:.2e})")

# Consequence-stratified
print(f"\n  By consequence type:")
print(f"  {'Consequence':<20} {'N':>6} {'P':>5} {'B':>5} {'AUC':>8}")
print(f"  {'─'*20} {'─'*6} {'─'*5} {'─'*5} {'─'*8}")
for cons in ['Missense','Canonical splice','Splice region',
             'Synonymous','Nonsense','Intronic']:
    mask = df['consequence'].values == cons
    n    = mask.sum()
    if n < 10: continue
    y_c  = y_sge[mask]
    p_c  = proba[mask]
    if y_c.sum()==0 or (y_c==0).sum()==0:
        print(f"  {cons:<20} {n:>6} {y_c.sum():>5} {(y_c==0).sum():>5}  single class")
        continue
    auc_c = roc_auc_score(y_c, p_c)
    print(f"  {cons:<20} {n:>6} {y_c.sum():>5} {(y_c==0).sum():>5} {auc_c:>8.4f}")

# ── Figures ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'External Validation V2 — Findlay 2018 SGE + dbNSFP Scores\n'
    f'BRCA1 Pathogenicity Model | n={len(y_sge):,} | '
    f'ROC-AUC={metrics_all["ROC_AUC"]}',
    fontsize=13, fontweight='bold'
)

ax = axes[0,0]
fpr, tpr, _ = roc_curve(y_sge, proba)
ax.plot(fpr, tpr, color='#1D9E75', lw=2.5,
        label=f'All variants (AUC={metrics_all["ROC_AUC"]})')
if metrics_mis:
    fpr_m, tpr_m, _ = roc_curve(y_mis, p_mis)
    ax.plot(fpr_m, tpr_m, color='#D85A30', lw=2, linestyle='--',
            label=f'Missense (AUC={metrics_mis["ROC_AUC"]})')
ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5,label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — SGE V2')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0,1]
prec_c, rec_c, _ = precision_recall_curve(y_sge, proba)
ax.plot(rec_c, prec_c, color='#7F77DD', lw=2.5,
        label=f'PR-AUC={metrics_all["PR_AUC"]}')
ax.axhline(y_sge.mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline ({y_sge.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('PR Curve — SGE V2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[0,2]
cm = confusion_matrix(y_sge, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred FUNC','Pred LOF'],
            yticklabels=['True FUNC','True LOF'], ax=ax)
ax.set_title('Confusion Matrix — SGE V2')

ax = axes[1,0]
ax.hist(proba[y_sge==0], bins=40, alpha=0.7, color='#2196F3',
        label='FUNC (benign-like)', edgecolor='white', density=True)
ax.hist(proba[y_sge==1], bins=40, alpha=0.7, color='#F44336',
        label='LOF (path-like)', edgecolor='white', density=True)
ax.axvline(0.5, color='black', linestyle='--', lw=1.5)
ax.set_xlabel('Predicted Pathogenic Probability')
ax.set_ylabel('Density')
ax.set_title('Probability Distribution by SGE Class')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1,1]
sc = ax.scatter(df['function_score_mean'].values, proba,
                c=y_sge, cmap='RdBu_r', alpha=0.25, s=6)
ax.axvline(-1.0, color='gray', linestyle='--', lw=1, alpha=0.7)
ax.axhline(0.5,  color='black',linestyle='--', lw=1, alpha=0.7)
ax.set_xlabel('SGE Function Score')
ax.set_ylabel('Predicted Pathogenic Probability')
ax.set_title(f'SGE Score vs Model Prediction\nSpearman ρ={r_sp:.4f}')
ax.grid(True, alpha=0.2)
plt.colorbar(sc, ax=ax, label='True label')

ax = axes[1,2]
cons_rows = []
for cons in ['Missense','Canonical splice','Splice region',
             'Synonymous','Nonsense','Intronic']:
    mask = df['consequence'].values == cons
    n    = mask.sum()
    if n < 10: continue
    y_c  = y_sge[mask]
    p_c  = proba[mask]
    if y_c.sum()==0 or (y_c==0).sum()==0: continue
    cons_rows.append({'consequence':cons, 'n':n,
                      'auc':roc_auc_score(y_c,p_c)})
if cons_rows:
    cons_df = pd.DataFrame(cons_rows).sort_values('auc',ascending=True)
    colors  = ['#F44336' if a>=0.85 else '#FF9800' if a>=0.70
               else '#2196F3' for a in cons_df['auc']]
    bars    = ax.barh(cons_df['consequence'], cons_df['auc'],
                      color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(0.85, color='green', linestyle='--', lw=1.5, alpha=0.7)
    ax.axvline(0.5,  color='gray',  linestyle='--', lw=1,   alpha=0.5)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('AUC by Consequence — SGE V2')
    ax.set_xlim([0,1.05])
    for bar, (_, row) in zip(bars, cons_df.iterrows()):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                f'n={row["n"]}', va='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/figures/external_validation_sge_v2.png',
            dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  results/figures/external_validation_sge_v2.png saved")

# ── Comparison table ──────────────────────────────────────────────────────────
print(f"""
{'='*65}
COMPARISON: V1 (no scores) vs V2 (with dbNSFP scores)
{'='*65}

  {'Metric':<16} {'V1 (no scores)':>16} {'V2 (with scores)':>18} {'Change':>8}
  {'─'*16} {'─'*16} {'─'*18} {'─'*8}
  {'Overall AUC':<16} {'0.7430':>16} {metrics_all['ROC_AUC']:>18.4f} {metrics_all['ROC_AUC']-0.7430:>+8.4f}
  {'Overall F1':<16} {'0.2836':>16} {metrics_all['F1']:>18.4f} {metrics_all['F1']-0.2836:>+8.4f}
  {'Overall MCC':<16} {'0.3645':>16} {metrics_all['MCC']:>18.4f} {metrics_all['MCC']-0.3645:>+8.4f}
  {'Missense AUC':<16} {'0.5002':>16} {metrics_mis['ROC_AUC'] if metrics_mis else 'N/A':>18} {'':>8}
  {'Spearman rho':<16} {'0.2702':>16} {r_sp:>18.4f} {r_sp-0.2702:>+8.4f}

  Internal test set (ClinVar holdout): ROC-AUC = 0.9973

  {'='*65}
  FINAL EXTERNAL VALIDATION SUMMARY
  {'='*65}
  Dataset      : Findlay et al. 2018 SGE (Nature 562:217-222)
  N variants   : {len(y_sge):,}  (FUNC={( y_sge==0).sum():,} LOF={y_sge.sum():,})
  dbNSFP fetch : BayesDel={df['BayesDel_addAF'].notna().sum():,} ({df['BayesDel_addAF'].notna().mean()*100:.1f}%)
               : REVEL={df['REVEL_score'].notna().sum():,} ({df['REVEL_score'].notna().mean()*100:.1f}%)
               : AlphaMissense={df['AlphaMissense_score'].notna().sum():,} ({df['AlphaMissense_score'].notna().mean()*100:.1f}%)
""")

# Save
pd.DataFrame([{
    'Dataset'    : 'Findlay_SGE_V2_with_dbNSFP',
    'N'          : len(y_sge),
    'N_LOF'      : int(y_sge.sum()),
    'N_FUNC'     : int((y_sge==0).sum()),
    **metrics_all,
    'Missense_AUC': metrics_mis['ROC_AUC'] if metrics_mis else None,
    'Missense_MCC': metrics_mis['MCC']      if metrics_mis else None,
    'Spearman_rho': round(r_sp,4),
    'Pearson_r'   : round(r_pe,4),
}]).to_csv('results/metrics/external_validation_sge_v2.csv', index=False)
print("  results/metrics/external_validation_sge_v2.csv saved")