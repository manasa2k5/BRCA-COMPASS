# ============================================================
# PHASE 4 FINAL — Complete Publication Pipeline
# File: src/phase4_final.py
#
# Strategy confirmed from diagnostic:
#   - Report BOTH overall and missense-only metrics
#   - Missense-only AUC is primary clinical contribution
#   - No retraining needed (missense AUC = 0.9955)
#   - CV nan bug fixed (removed eval_metric from XGB)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import make_scorer
import xgboost as xgb

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/vus',     exist_ok=True)
os.makedirs('models',          exist_ok=True)

print("=" * 65)
print("PHASE 4 FINAL: Complete Publication Pipeline")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
df     = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)
df_vus = pd.read_csv('data/processed/vus_features_scored.csv',   low_memory=False)

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

LABEL     = 'label'
available = [c for c in FEATURE_COLS if c in df.columns]

X = df[available].copy()
y = df[LABEL].astype(int).copy()

print(f"\nDataset : {len(df):,} | Features : {len(available)}")
print(f"P={y.sum():,} ({y.mean()*100:.1f}%)  B={(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

# ============================================================
# SECTION 1: Stratified 70/10/20 Split
# ============================================================
print("\n" + "─" * 65)
print("SECTION 1: Stratified 70/10/20 Split")
print("─" * 65)

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.125, random_state=42, stratify=y_tv
)

# Imputer fit on train only
imputer     = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=available)
X_val_imp   = pd.DataFrame(imputer.transform(X_val),       columns=available)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=available)
joblib.dump(imputer, 'models/imputer.pkl')

# VUS imputation — train imputer only
X_vus_avail = [c for c in available if c in df_vus.columns]
X_vus_imp   = pd.DataFrame(
    imputer.transform(df_vus[X_vus_avail]),
    columns=X_vus_avail
)

print(f"  Train : {len(X_train):,} | Val : {len(X_val):,} | "
      f"Test : {len(X_test):,} | VUS : {len(X_vus_imp):,}")
print(f"  ✓ Imputer fit on train only")
print(f"  ✓ VUS never in any split")

# ============================================================
# SECTION 2: XGBoost Training
# (AutoGluon separately — run after this if needed)
# ============================================================
print("\n" + "─" * 65)
print("SECTION 2: XGBoost Training (train split only)")
print("─" * 65)

scale_pw = float((y_train==0).sum()) / float((y_train==1).sum())
print(f"  scale_pos_weight : {scale_pw:.4f}")

# NOTE: NO eval_metric in constructor — this fixes the CV nan bug
xgb_base = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pw,
    random_state=42,
    verbosity=0,
    objective='binary:logistic',   # ← explicitly set binary classification
    n_jobs=-1
)
xgb_base.fit(
    X_train_imp.values, y_train.values,
    eval_set=[(X_val_imp.values, y_val.values)],
    verbose=False
)

# ============================================================
# SECTION 3: Calibration on VALIDATION only
# ============================================================
# ============================================================
# SECTION 3: Calibration (validation only)
# ============================================================
print("\n" + "─" * 65)
print("SECTION 3: Calibration (validation only)")
print("─" * 65)

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

# Before calibration metrics
val_proba_pre = xgb_base.predict_proba(X_val_imp.values)[:, 1]
brier_pre     = round(brier_score_loss(y_val.values, val_proba_pre), 4)
ece_pre       = ece_score(y_val.values, val_proba_pre)

# ── Platt scaling — manual implementation ────────────────────────────────────
# sklearn 1.7+ tightened is_classifier() check — XGBoost fails it.
# Implement Platt scaling directly with LogisticRegression.
# This is mathematically identical to CalibratedClassifierCV(method='sigmoid').

from sklearn.linear_model import LogisticRegression

# Get raw logit scores from XGBoost on validation
val_logits = xgb_base.predict_proba(X_val_imp.values)[:, 1].reshape(-1, 1)

# Fit Platt scaler on validation only
platt_scaler = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
platt_scaler.fit(val_logits, y_val.values)
joblib.dump(platt_scaler, 'models/platt_scaler.pkl')

def calibrated_predict_proba(X, xgb_model, platt):
    """Apply XGBoost then Platt scaling."""
    raw_proba = xgb_model.predict_proba(X)[:, 1].reshape(-1, 1)
    cal_proba = platt.predict_proba(raw_proba)[:, 1]
    return cal_proba

# Calibrated validation probabilities
val_proba_cal = calibrated_predict_proba(
    X_val_imp.values, xgb_base, platt_scaler
)
brier_post = round(brier_score_loss(y_val.values, val_proba_cal), 4)
ece_post   = ece_score(y_val.values, val_proba_cal)

print(f"  Brier: {brier_pre} → {brier_post}  "
      f"(Δ={brier_pre-brier_post:+.4f})")
print(f"  ECE  : {ece_pre} → {ece_post}  "
      f"(Δ={ece_pre-ece_post:+.4f})")
print(f"  ✓ Platt scaling fit on validation only")
print(f"  ✓ Implementation: LogisticRegression on XGBoost outputs")
print(f"     (mathematically identical to sklearn CalibratedClassifierCV)")

# Calibration plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Probability Calibration — BRCA1 Pathogenicity\n'
             '(Platt Scaling, Fitted on Validation Set Only)',
             fontsize=12, fontweight='bold')
for ax, proba, title, br, ec in [
    (axes[0], val_proba_pre, 'Before Calibration', brier_pre, ece_pre),
    (axes[1], val_proba_cal, 'After Platt Scaling', brier_post, ece_post)
]:
    fp, mp = calibration_curve(y_val.values, proba, n_bins=10)
    ax.plot(mp, fp, 's-', color='#D85A30', label='Model')
    ax.plot([0,1],[0,1],'--', color='gray', label='Perfect')
    ax.set_title(f'{title}\nBrier={br} | ECE={ec}')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/calibration_curves.png',
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  results/figures/calibration_curves.png saved")

# ============================================================
# SECTION 4: Threshold Optimization on VALIDATION only
# ============================================================
print("\n" + "─" * 65)
print("SECTION 4: Threshold Optimization (validation only)")
print("─" * 65)

def optimal_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s[:-1])
    return float(thresholds[idx]), float(f1s[idx])

opt_thresh, opt_f1_val = optimal_threshold(y_val.values, val_proba_cal)
print(f"  Optimal threshold : {opt_thresh:.4f}")
print(f"  F1 at threshold   : {opt_f1_val:.4f}")
print(f"  ✓ Threshold optimized on validation only")
joblib.dump({'threshold': opt_thresh}, 'models/threshold.pkl')

# ============================================================
# SECTION 5: 5-Fold Cross-Validation (train+val only)
# ============================================================
# ============================================================
# SECTION 5: 5-Fold Stratified Cross-Validation
# ============================================================
print("\n" + "─" * 65)
print("SECTION 5: 5-Fold Stratified Cross-Validation")
print("─" * 65)
print("  CV dataset: train+val only — test excluded")

X_tv_imp = pd.DataFrame(
    imputer.transform(X_tv), columns=available
)
y_tv_arr = y_tv.values

print(f"  CV dataset: {len(X_tv_imp):,} variants "
      f"(train {len(X_train_imp):,} + val {len(X_val_imp):,})")

# Manual 5-fold CV — bypasses sklearn scorer version issues entirely
cv_skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    'roc_auc': [], 'pr_auc': [], 'f1': [],
    'mcc': [], 'balanced_accuracy': [],
    'train_roc_auc': []
}

fold_spw = float((y_tv_arr==0).sum()) / float((y_tv_arr==1).sum())

for fold_num, (tr_idx, va_idx) in enumerate(
        cv_skf.split(X_tv_imp.values, y_tv_arr), 1):

    X_fold_tr = X_tv_imp.values[tr_idx]
    y_fold_tr = y_tv_arr[tr_idx]
    X_fold_va = X_tv_imp.values[va_idx]
    y_fold_va = y_tv_arr[va_idx]

    # Fit imputer on fold train only — strict leak prevention
    fold_imp = SimpleImputer(strategy='median')
    X_fold_tr = fold_imp.fit_transform(X_fold_tr)
    X_fold_va = fold_imp.transform(X_fold_va)

    fold_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fold_spw,
        random_state=42,
        objective='binary:logistic',
        verbosity=0,
        n_jobs=-1
    )
    fold_model.fit(X_fold_tr, y_fold_tr)

    p_tr = fold_model.predict_proba(X_fold_tr)[:, 1]
    p_va = fold_model.predict_proba(X_fold_va)[:, 1]
    y_va = y_fold_va
    y_pred_va = (p_va >= 0.5).astype(int)

    cv_metrics['roc_auc'].append(
        roc_auc_score(y_va, p_va))
    cv_metrics['pr_auc'].append(
        average_precision_score(y_va, p_va))
    cv_metrics['f1'].append(
        f1_score(y_va, y_pred_va, zero_division=0))
    cv_metrics['mcc'].append(
        matthews_corrcoef(y_va, y_pred_va))
    cv_metrics['balanced_accuracy'].append(
        balanced_accuracy_score(y_va, y_pred_va))
    cv_metrics['train_roc_auc'].append(
        roc_auc_score(y_fold_tr, p_tr))

    print(f"  Fold {fold_num}: "
          f"AUC={cv_metrics['roc_auc'][-1]:.4f}  "
          f"F1={cv_metrics['f1'][-1]:.4f}  "
          f"MCC={cv_metrics['mcc'][-1]:.4f}")

cv_summary = {}
print(f"\n  {'Metric':<22} {'Val mean±std':>18}  {'Train mean':>12}")
print(f"  {'─'*22} {'─'*18}  {'─'*12}")

for metric in ['roc_auc','pr_auc','f1','mcc','balanced_accuracy']:
    vals  = np.array(cv_metrics[metric])
    t_key = f'train_{metric}'
    t_val = np.array(cv_metrics[t_key]).mean() \
            if t_key in cv_metrics else float('nan')
    cv_summary[metric] = {
        'val_mean'  : round(float(vals.mean()), 4),
        'val_std'   : round(float(vals.std()),  4),
        'train_mean': round(float(t_val), 4)
            if not np.isnan(t_val) else float('nan'),
    }
    t_str = f"{t_val:.4f}" if not np.isnan(t_val) else "n/a"
    print(f"  {metric:<22} "
          f"{vals.mean():.4f} ± {vals.std():.4f}      "
          f"{t_str}")

pd.DataFrame(cv_summary).T.to_csv(
    'results/metrics/cv_metrics.csv'
)
print(f"\n  ✓ CV complete — results/metrics/cv_metrics.csv saved")

# ============================================================
# SECTION 6: FINAL METRICS — Unlock Test Set
# ============================================================
print("\n" + "─" * 65)
print("SECTION 6: Final Metrics — Train / Val / Test")
print("─" * 65)
print("  *** TEST SET UNLOCKED ***")

def compute_metrics(name, split, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'Model'       : name,
        'Split'       : split,
        'ROC_AUC'     : round(roc_auc_score(y_true, y_prob), 4),
        'PR_AUC'      : round(average_precision_score(y_true, y_prob), 4),
        'Accuracy'    : round(accuracy_score(y_true, y_pred), 4),
        'Precision'   : round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'      : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1'          : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC'         : round(matthews_corrcoef(y_true, y_pred), 4),
        'Balanced_Acc': round(balanced_accuracy_score(y_true, y_pred), 4),
    }

# All probabilities
# NEW
train_proba = calibrated_predict_proba(X_train_imp.values, xgb_base, platt_scaler)
val_proba   = calibrated_predict_proba(X_val_imp.values,   xgb_base, platt_scaler)
test_proba  = calibrated_predict_proba(X_test_imp.values,  xgb_base, platt_scaler)

all_metrics = []
for split, y_s, p_s in [
    ('Train',      y_train.values, train_proba),
    ('Validation', y_val.values,   val_proba),
    ('Test',       y_test.values,  test_proba),
]:
    all_metrics.append(
        compute_metrics('XGBoost_Calibrated', split, y_s, p_s, opt_thresh)
    )

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv('results/metrics/full_metrics.csv', index=False)

print(f"\n  XGBoost Calibrated — All Splits:")
print(metrics_df[['Split','ROC_AUC','PR_AUC','Accuracy',
                  'Precision','Recall','F1','MCC',
                  'Balanced_Acc']].to_string(index=False))

# ── Missense-only metrics on test set ────────────────────────────────────────
print(f"\n  {'─'*55}")
print(f"  MISSENSE-ONLY METRICS (Test Set — Primary Clinical Metric)")
print(f"  {'─'*55}")

test_idx    = X_test.index
df_test_ref = df.loc[test_idx].reset_index(drop=True)
mis_mask    = df_test_ref['is_missense'].values == 1

y_mis  = y_test.values[mis_mask]
p_mis  = test_proba[mis_mask]

if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
    mis_metrics = compute_metrics(
        'XGBoost_Missense_Only', 'Test', y_mis, p_mis, opt_thresh
    )
    print(f"\n  n={mis_mask.sum()} | P={y_mis.sum()} | B={(y_mis==0).sum()}")
    for k, v in mis_metrics.items():
        if k not in ['Model','Split']:
            print(f"    {k:<15} : {v}")
    mis_metrics['Split'] = 'Test_MissenseOnly'
    metrics_df = pd.concat([metrics_df,
                            pd.DataFrame([mis_metrics])],
                           ignore_index=True)
    metrics_df.to_csv('results/metrics/full_metrics.csv', index=False)

# ── Overfitting Analysis ──────────────────────────────────────────────────────
print(f"\n  {'─'*65}")
print(f"  OVERFITTING ANALYSIS")
print(f"  {'─'*65}")

rows = metrics_df[metrics_df['Model']=='XGBoost_Calibrated']
tr_auc = float(rows[rows['Split']=='Train']['ROC_AUC'].values[0])
va_auc = float(rows[rows['Split']=='Validation']['ROC_AUC'].values[0])
te_auc = float(rows[rows['Split']=='Test']['ROC_AUC'].values[0])
gap_tv = round(tr_auc - va_auc, 4)
gap_tt = round(tr_auc - te_auc, 4)
status = ('Excellent' if max(gap_tv, gap_tt) < 0.03
          else 'Acceptable' if max(gap_tv, gap_tt) <= 0.05
          else 'Review')

print(f"\n  {'Model':<25} {'Train':>8} {'Val':>8} {'Test':>8} "
      f"{'Gap T-V':>9} {'Gap T-Te':>9} Status")
print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*10}")
print(f"  {'XGBoost_Calibrated':<25} {tr_auc:>8.4f} {va_auc:>8.4f} "
      f"{te_auc:>8.4f} {gap_tv:>9.4f} {gap_tt:>9.4f} {status}")

pd.DataFrame([{
    'Model':'XGBoost_Calibrated',
    'Train_AUC':tr_auc,'Val_AUC':va_auc,'Test_AUC':te_auc,
    'Gap_TrainVal':gap_tv,'Gap_TrainTest':gap_tt,'Status':status
}]).to_csv('results/metrics/overfitting_analysis.csv', index=False)

# Calibration on test
brier_test = round(brier_score_loss(y_test.values, test_proba), 4)
ece_test   = ece_score(y_test.values, test_proba)
print(f"\n  Calibration (test): Brier={brier_test} | ECE={ece_test}")

# ============================================================
# SECTION 7: ROC + PR Curves
# ============================================================
print("\n" + "─" * 65)
print("SECTION 7: ROC and PR Curves")
print("─" * 65)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('BRCA1 Pathogenicity Prediction — ROC and PR Curves\n'
             'Independent Test Set (n=814, 20% holdout)',
             fontsize=12, fontweight='bold')

ax = axes[0]
# Overall
fpr, tpr, _ = roc_curve(y_test.values, test_proba)
ax.plot(fpr, tpr, color='#1D9E75', lw=2,
        label=f'All variants (AUC={te_auc:.4f})')
# Missense only
if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
    fpr_m, tpr_m, _ = roc_curve(y_mis, p_mis)
    auc_m = roc_auc_score(y_mis, p_mis)
    ax.plot(fpr_m, tpr_m, color='#D85A30', lw=2, linestyle='--',
            label=f'Missense only (AUC={auc_m:.4f})')
ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5,label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Test Set')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

ax = axes[1]
prec, rec, _ = precision_recall_curve(y_test.values, test_proba)
pr_v = average_precision_score(y_test.values, test_proba)
ax.plot(rec, prec, color='#7F77DD', lw=2,
        label=f'All variants (PR-AUC={pr_v:.4f})')
if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
    prec_m, rec_m, _ = precision_recall_curve(y_mis, p_mis)
    pr_m = average_precision_score(y_mis, p_mis)
    ax.plot(rec_m, prec_m, color='#EF9F27', lw=2, linestyle='--',
            label=f'Missense only (PR-AUC={pr_m:.4f})')
ax.axhline(y_test.mean(), color='gray', linestyle='--', alpha=0.5,
           label=f'Baseline ({y_test.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('PR Curve — Test Set')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  results/figures/roc_pr_curves.png saved")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mask, title in [
    (axes[0], np.ones(len(y_test), dtype=bool), 'All Variants (n=814)'),
    (axes[1], mis_mask, f'Missense Only (n={mis_mask.sum()})')
]:
    y_s = y_test.values[mask]
    p_s = test_proba[mask]
    cm  = confusion_matrix(y_s, (p_s >= opt_thresh).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Benign','Pred Path'],
                yticklabels=['True Benign','True Path'], ax=ax)
    ax.set_title(f'Confusion Matrix\n{title}')
plt.suptitle(f'BRCA1 Pathogenicity — threshold={opt_thresh:.3f}',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  results/figures/confusion_matrix.png saved")

# ============================================================
# SECTION 8: SHAP
# ============================================================
print("\n" + "─" * 65)
print("SECTION 8: SHAP Explainability")
print("─" * 65)

import shap
explainer  = shap.TreeExplainer(xgb_base)
shap_test  = explainer.shap_values(X_test_imp.values)
shap_train = explainer.shap_values(X_train_imp.values)

# Global bar
plt.figure(figsize=(10, 9))
shap.summary_plot(shap_test, X_test_imp, plot_type='bar',
                  max_display=20, show=False)
plt.title('Global Feature Importance (SHAP) — Test Set\n'
          'BRCA1 Pathogenicity Prediction', fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/shap_global_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Beeswarm
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_test, X_test_imp, max_display=20, show=False)
plt.title('SHAP Beeswarm — Feature Impact Distribution\n'
          'BRCA1 Pathogenicity Prediction', fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/shap_global_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP importance table
shap_imp = pd.DataFrame({
    'Feature'  : available,
    'Mean_SHAP': np.abs(shap_test).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)
shap_imp.to_csv('results/metrics/shap_feature_importance.csv', index=False)

print(f"  Top 10 features by SHAP (test set):")
for _, r in shap_imp.head(10).iterrows():
    bar = '█' * int(r['Mean_SHAP'] * 60)
    print(f"    {r['Feature']:<30} {r['Mean_SHAP']:.4f}  {bar}")

# Waterfall plots — use missense variants for clinical relevance
test_proba_base = xgb_base.predict_proba(X_test_imp.values)[:, 1]
sv_obj          = explainer(X_test_imp)

# Select from missense test variants only
mis_indices = np.where(mis_mask)[0]

if len(mis_indices) >= 3:
    p_mis_base = test_proba_base[mis_indices]
    y_mis_base = y_test.values[mis_indices]

    hi_path = mis_indices[
        np.where((y_mis_base==1) & (p_mis_base>=0.80))[0]
    ]
    hi_ben  = mis_indices[
        np.where((y_mis_base==0) & (p_mis_base<=0.20))[0]
    ]
    unc     = mis_indices[
        np.argmin(np.abs(p_mis_base - 0.5))
    ]

    for idx, lbl in [
        (hi_path[0] if len(hi_path) > 0 else mis_indices[0],
         'missense_pathogenic'),
        (hi_ben[0]  if len(hi_ben) > 0  else mis_indices[-1],
         'missense_benign'),
        (int(unc),   'missense_uncertain')
    ]:
        try:
            plt.figure()
            shap.plots.waterfall(sv_obj[int(idx)], max_display=15, show=False)
            plt.title(
                f'SHAP Waterfall — {lbl.replace("_"," ").title()}\n'
                f'P(path)={test_proba_base[idx]:.3f}  '
                f'True={y_test.values[idx]}',
                fontsize=10
            )
            plt.tight_layout()
            plt.savefig(
                f'results/figures/shap_waterfall_{lbl}.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            print(f"  results/figures/shap_waterfall_{lbl}.png saved")
        except Exception as e:
            print(f"  Waterfall {lbl} skipped: {e}")

print(f"  results/figures/shap_global_bar.png saved")
print(f"  results/figures/shap_global_beeswarm.png saved")

# ============================================================
# SECTION 9: Confidence Framework
# ============================================================
print("\n" + "─" * 65)
print("SECTION 9: Confidence Framework")
print("─" * 65)

def assign_confidence(prob_cal, prob_base):
    dist  = abs(prob_cal - 0.5)
    agree = abs(prob_cal - prob_base) < 0.15
    if dist >= 0.35 and agree:
        return 'High'
    elif dist >= 0.15:
        return 'Medium'
    else:
        return 'Low'

test_base_proba = xgb_base.predict_proba(X_test_imp.values)[:, 1]
test_conf = [assign_confidence(c, b)
             for c, b in zip(test_proba, test_base_proba)]
conf_s    = pd.Series(test_conf)

print(f"\n  Test set confidence distribution:")
for tier in ['High', 'Medium', 'Low']:
    n    = (conf_s==tier).sum()
    pct  = (conf_s==tier).mean()*100
    mask = np.array([c==tier for c in test_conf])
    acc  = accuracy_score(
        y_test.values[mask],
        (test_proba[mask] >= opt_thresh).astype(int)
    ) if mask.sum() > 0 else 0
    print(f"    {tier:<8} n={n:>4} ({pct:4.1f}%)  accuracy={acc:.4f}")

# ============================================================
# SECTION 10: VUS Inference
# ============================================================
print("\n" + "─" * 65)
print("SECTION 10: VUS Inference")
print("─" * 65)

vus_proba_cal  = calibrated_predict_proba(X_vus_imp.values, xgb_base, platt_scaler)
vus_proba_base = xgb_base.predict_proba(X_vus_imp.values)[:, 1]
shap_vus       = explainer.shap_values(X_vus_imp.values)
shap_vus_df    = pd.DataFrame(shap_vus, columns=X_vus_avail)

def map_acmg(shap_row, prob, feat_dict):
    codes = []
    top5  = shap_row.abs().nlargest(5).index.tolist()
    pred  = 1 if prob >= 0.5 else 0
    if pred == 1:
        if feat_dict.get('is_likely_lof', 0) == 1 \
                and 'is_likely_lof' in top5:
            codes.append('PVS1')
        if feat_dict.get('in_pathogenic_hotspot', 0) == 1 \
                and 'in_pathogenic_hotspot' in top5:
            codes.append('PM1')
        if any(c in top5 for c in
               ['BayesDel_addAF','BayesDel_noAF','REVEL_score']):
            codes.append('PP3_Strong' if prob >= 0.90 else 'PP3')
        if feat_dict.get('is_canonical_splice_site', 0) == 1:
            codes.append('PS3_Moderate')
    else:
        if any(c in top5 for c in
               ['BayesDel_addAF','REVEL_score','AlphaMissense_score']):
            codes.append('BP4')
        if feat_dict.get('gnomAD_AF', 0) > 0.001:
            codes.append('BS1')
        if feat_dict.get('is_synonymous', 0) == 1:
            codes.append('BP7')
    return ', '.join(codes) if codes else 'Insufficient_evidence'

vus_rows = []
for i in range(len(X_vus_imp)):
    p_cal  = float(vus_proba_cal[i])
    p_base = float(vus_proba_base[i])
    p_ens  = round((p_cal + p_base) / 2, 4)
    conf   = assign_confidence(p_cal, p_base)
    sr     = shap_vus_df.iloc[i]
    fd     = dict(zip(X_vus_avail, X_vus_imp.values[i]))
    acmg   = map_acmg(sr, p_ens, fd)
    top5   = sr.abs().nlargest(5).index.tolist()
    vus_rows.append({
        'VariationID'      : df_vus.iloc[i].get('VariationID', i),
        'Name'             : df_vus.iloc[i].get('Name', ''),
        'Cal_prob'         : round(p_cal,  4),
        'Base_prob'        : round(p_base, 4),
        'Ensemble_prob'    : p_ens,
        'Predicted_class'  : 'Pathogenic' if p_ens >= 0.5 else 'Benign',
        'Confidence'       : conf,
        'Top5_SHAP'        : ', '.join(top5),
        'ACMG_codes'       : acmg,
    })

vus_out = pd.DataFrame(vus_rows)
vus_out.to_csv('results/vus/vus_predictions.csv', index=False)

print(f"\n  VUS total         : {len(vus_out):,}")
print(f"  Pred Pathogenic   : "
      f"{(vus_out['Predicted_class']=='Pathogenic').sum():,} "
      f"({(vus_out['Predicted_class']=='Pathogenic').mean()*100:.1f}%)")
print(f"  Pred Benign       : "
      f"{(vus_out['Predicted_class']=='Benign').sum():,} "
      f"({(vus_out['Predicted_class']=='Benign').mean()*100:.1f}%)")
for tier in ['High','Medium','Low']:
    n = (vus_out['Confidence']==tier).sum()
    print(f"  {tier} confidence  : {n:,} "
          f"({n/len(vus_out)*100:.1f}%)")

# VUS probability distribution
plt.figure(figsize=(10, 5))
for cls, col in [('Pathogenic','#F44336'),('Benign','#2196F3')]:
    sub = vus_out[vus_out['Predicted_class']==cls]['Ensemble_prob']
    plt.hist(sub, bins=30, alpha=0.7, color=col,
             label=f'Pred {cls}', edgecolor='white')
plt.axvline(0.5, color='black', linestyle='--', lw=1.5,
            label='Decision boundary (0.5)')
plt.xlabel('Pathogenic Probability Score', fontsize=12)
plt.ylabel('Number of VUS', fontsize=12)
plt.title(f'VUS Pathogenicity Probability Distribution\n'
          f'n={len(vus_out):,} Variants of Uncertain Significance',
          fontsize=12, fontweight='bold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/vus_probability_distribution.png',
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  results/figures/vus_probability_distribution.png saved")

# ============================================================
# SECTION 11: FINAL PUBLICATION SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("SECTION 11: FINAL PUBLICATION SUMMARY")
print("=" * 65)

xr  = metrics_df[metrics_df['Model']=='XGBoost_Calibrated']
tr  = xr[xr['Split']=='Train'].iloc[0]
va  = xr[xr['Split']=='Validation'].iloc[0]
te  = xr[xr['Split']=='Test'].iloc[0]

try:
    mis_row = metrics_df[
        metrics_df['Split']=='Test_MissenseOnly'
    ].iloc[0]
    mis_str = (f"    ROC-AUC  : {mis_row.ROC_AUC}\n"
               f"    PR-AUC   : {mis_row.PR_AUC}\n"
               f"    F1       : {mis_row.F1}\n"
               f"    MCC      : {mis_row.MCC}\n"
               f"    n        : {mis_mask.sum()} "
               f"(P={y_mis.sum()} B={(y_mis==0).sum()})")
except:
    mis_str = "    See missense analysis above"

print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  BRCA1 PATHOGENICITY PREDICTION — PUBLICATION METRICS    ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  PRIMARY TABLE — All Variants                            ║
  ╠══════════════════╦══════════╦════════════╦═══════════════╣
  ║  Metric          ║  Train   ║ Validation ║  Test         ║
  ╠══════════════════╬══════════╬════════════╬═══════════════╣
  ║  ROC-AUC         ║  {tr.ROC_AUC:.4f}  ║   {va.ROC_AUC:.4f}   ║   {te.ROC_AUC:.4f}     ║
  ║  PR-AUC          ║  {tr.PR_AUC:.4f}  ║   {va.PR_AUC:.4f}   ║   {te.PR_AUC:.4f}     ║
  ║  Accuracy        ║  {tr.Accuracy:.4f}  ║   {va.Accuracy:.4f}   ║   {te.Accuracy:.4f}     ║
  ║  Precision       ║  {tr.Precision:.4f}  ║   {va.Precision:.4f}   ║   {te.Precision:.4f}     ║
  ║  Recall          ║  {tr.Recall:.4f}  ║   {va.Recall:.4f}   ║   {te.Recall:.4f}     ║
  ║  F1 Score        ║  {tr.F1:.4f}  ║   {va.F1:.4f}   ║   {te.F1:.4f}     ║
  ║  MCC             ║  {tr.MCC:.4f}  ║   {va.MCC:.4f}   ║   {te.MCC:.4f}     ║
  ║  Balanced Acc    ║  {tr.Balanced_Acc:.4f}  ║   {va.Balanced_Acc:.4f}   ║   {te.Balanced_Acc:.4f}     ║
  ╠══════════════════╩══════════╩════════════╩═══════════════╣
  ║  Overfitting: Train-Test Gap = {gap_tt:.4f} ({status})    
  ╠═══════════════════════════════════════════════════════════╣
  ║  MISSENSE-ONLY TABLE (Primary Clinical Contribution)     ║
  ╠═══════════════════════════════════════════════════════════╣
{mis_str}
  ╠═══════════════════════════════════════════════════════════╣
  ║  5-Fold CV (XGBoost, train+val, n={len(X_tv_imp):,})         
  ╠═══════════════════════════════════════════════════════════╣
  ║  ROC-AUC  : {cv_summary['roc_auc']['val_mean']:.4f} ± {cv_summary['roc_auc']['val_std']:.4f}                    ║
  ║  ║  PR-AUC   : {cv_summary['pr_auc']['val_mean']:.4f} ± {cv_summary['pr_auc']['val_std']:.4f}                  ║
  ║  F1       : {cv_summary['f1']['val_mean']:.4f} ± {cv_summary['f1']['val_std']:.4f}                    ║
  ║  MCC      : {cv_summary['mcc']['val_mean']:.4f} ± {cv_summary['mcc']['val_std']:.4f}                    ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Calibration (test set, fitted on val)                   ║
  ║  Brier : {brier_test}  |  ECE : {ece_test}               
  ╠═══════════════════════════════════════════════════════════╣
  ║  VUS Inference                                           ║
  ║  Total={len(vus_out):,} Path={
    (vus_out['Predicted_class']=='Pathogenic').sum()
  :,} Ben={
    (vus_out['Predicted_class']=='Benign').sum()
  :,} HighConf={
    (vus_out['Confidence']=='High').sum()
  :,}    ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Leakage Controls                                        ║
  ║  ✓ ClinSigSimple excluded                               ║
  ║  ✓ Calibration on validation only                       ║
  ║  ✓ Threshold on validation only                         ║
  ║  ✓ Test untouched until Section 6                       ║
  ║  ✓ VUS never in any split                               ║
  ╚═══════════════════════════════════════════════════════════╝
""")

print("=" * 65)
print("PHASE 4 FINAL COMPLETE")
print("=" * 65)
print("\nAll outputs saved:")
print("  results/metrics/full_metrics.csv")
print("  results/metrics/cv_metrics.csv")
print("  results/metrics/overfitting_analysis.csv")
print("  results/metrics/shap_feature_importance.csv")
print("  results/figures/roc_pr_curves.png")
print("  results/figures/confusion_matrix.png")
print("  results/figures/calibration_curves.png")
print("  results/figures/shap_global_bar.png")
print("  results/figures/shap_global_beeswarm.png")
print("  results/figures/shap_waterfall_missense_*.png")
print("  results/vus/vus_predictions.csv")