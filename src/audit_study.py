# ============================================================
# AUDIT STUDY — Feature Ablation + dbNSFP Experiment + CV Fix
# File: src/audit_study.py
# ============================================================

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    make_scorer
)
import xgboost as xgb

print("=" * 65)
print("AUDIT STUDY: Ablation + dbNSFP + CV Fix")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)

with open('data/processed/feature_cols_clean.json') as f:
    ALL_FEATURES = json.load(f)

LABEL     = 'label'
available = [c for c in ALL_FEATURES if c in df.columns]

X = df[available].copy()
y = df[LABEL].astype(int).copy()

# ── Fixed 70/10/20 split — same seed as production ───────────────────────────
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.125, random_state=42, stratify=y_tv
)

X_tv_all  = pd.concat([X_train, X_val], ignore_index=True)
y_tv_all  = pd.concat([y_train, y_val], ignore_index=True)

scale_pw  = float((y_train==0).sum()) / float((y_train==1).sum())

def run_experiment(name, feature_list, X_train, y_train,
                   X_val, y_val, X_test, y_test,
                   X_tv, y_tv, run_cv=True):
    """
    Train XGBoost on given feature subset.
    Report train / val / test metrics + 5-fold CV.
    Imputer fitted on train only.
    """
    print(f"\n{'─'*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  Features: {len(feature_list)}")
    print(f"  Feature list: {feature_list[:5]}{'...' if len(feature_list)>5 else ''}")

    # Filter to available features only
    feat = [f for f in feature_list if f in X_train.columns]
    missing = [f for f in feature_list if f not in X_train.columns]
    if missing:
        print(f"  Missing (skipped): {missing}")
    if len(feat) == 0:
        print(f"  ERROR: No features available")
        return None
    print(f"  Available: {len(feat)}")

    # Impute — fit on train only
    imp = SimpleImputer(strategy='median')
    Xtr = pd.DataFrame(imp.fit_transform(X_train[feat]), columns=feat)
    Xva = pd.DataFrame(imp.transform(X_val[feat]),       columns=feat)
    Xte = pd.DataFrame(imp.transform(X_test[feat]),      columns=feat)
    Xtv = pd.DataFrame(imp.transform(X_tv[feat]),        columns=feat)

    # Check class balance for this feature subset
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    spw   = float(n_neg) / float(n_pos)

    # Train
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=42,
        objective='binary:logistic',
        verbosity=0,
        n_jobs=-1
    )
    model.fit(Xtr.values, y_train.values)

    # Metrics
    def metrics(y_true, y_prob):
        thresh = 0.5
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'ROC_AUC' : round(roc_auc_score(y_true, y_prob), 4),
            'PR_AUC'  : round(average_precision_score(y_true, y_prob), 4),
            'F1'      : round(f1_score(y_true, y_pred, zero_division=0), 4),
            'MCC'     : round(matthews_corrcoef(y_true, y_pred), 4),
        }

    tr_m = metrics(y_train.values, model.predict_proba(Xtr.values)[:, 1])
    va_m = metrics(y_val.values,   model.predict_proba(Xva.values)[:, 1])
    te_m = metrics(y_test.values,  model.predict_proba(Xte.values)[:, 1])

    print(f"\n  {'Split':<12} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>9} {'MCC':>9}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for split, m in [('Train', tr_m), ('Validation', va_m), ('Test', te_m)]:
        print(f"  {split:<12} {m['ROC_AUC']:>9.4f} {m['PR_AUC']:>9.4f} "
              f"{m['F1']:>9.4f} {m['MCC']:>9.4f}")

    gap = round(tr_m['ROC_AUC'] - te_m['ROC_AUC'], 4)
    print(f"\n  Train-Test Gap : {gap}  "
          f"({'Excellent' if gap < 0.03 else 'Acceptable' if gap <= 0.05 else 'Review'})")

    # 5-Fold CV — fixed scorer to avoid nan bug
    if run_cv and len(feat) >= 2:
        cv_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=float((y_tv==0).sum())/float((y_tv==1).sum()),
            random_state=42,
            objective='binary:logistic',
            verbosity=0,
            n_jobs=-1
        )
        cv_scoring = {
            'roc_auc' : make_scorer(roc_auc_score,
                            needs_proba=True,
                            response_method='predict_proba'),
            'pr_auc'  : make_scorer(average_precision_score,
                            needs_proba=True,
                            response_method='predict_proba'),
            'f1'      : 'f1',
            'mcc'     : make_scorer(matthews_corrcoef),
        }
        cv_res = cross_validate(
            cv_model, Xtv.values, y_tv.values,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=cv_scoring,
            n_jobs=-1,
            return_train_score=False
        )
        print(f"\n  5-Fold CV (train+val, n={len(Xtv):,}):")
        for metric in ['roc_auc','pr_auc','f1','mcc']:
            vals = cv_res[f'test_{metric}']
            print(f"    {metric:<15} {vals.mean():.4f} ± {vals.std():.4f}")

    # Missense-only test metrics
    test_idx   = X_test.index
    df_ref     = df.loc[test_idx].reset_index(drop=True)
    mis_mask   = df_ref['is_missense'].values == 1
    n_mis      = mis_mask.sum()
    if n_mis >= 10:
        y_mis = y_test.values[mis_mask]
        p_mis = model.predict_proba(Xte.values)[:, 1][mis_mask]
        if y_mis.sum() > 0 and (y_mis==0).sum() > 0:
            m_mis = metrics(y_mis, p_mis)
            print(f"\n  Missense-only test (n={n_mis}, "
                  f"P={y_mis.sum()}, B={(y_mis==0).sum()}):")
            for k, v in m_mis.items():
                print(f"    {k:<12} : {v}")

    return {'name': name, 'n_features': len(feat),
            'train': tr_m, 'val': va_m, 'test': te_m, 'gap': gap}


# ============================================================
# EXPERIMENT 0: Full model baseline (reproduces production run)
# ============================================================
result_full = run_experiment(
    name         = 'FULL MODEL (baseline)',
    feature_list = available,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all
)

# ============================================================
# EXPERIMENT 1: Remove LOF + consequence proxies
# These are the features suspected of near-determinism
# ============================================================
LOF_PROXY_FEATURES = [
    'is_likely_lof',
    'cons_snv',
    'cons_splice',
    'cons_frameshift',
    'cons_nonsense',       # also remove nonsense — same concern
    'is_frameshift_protein',
    'is_nonsense',
    'is_frameshift',
]

ablated_features = [f for f in available if f not in LOF_PROXY_FEATURES]

result_ablated = run_experiment(
    name         = 'ABLATED (no LOF/consequence proxies)',
    feature_list = ablated_features,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all
)

# ============================================================
# EXPERIMENT 2: Remove num_submitters (generalizability concern)
# ============================================================
no_submitters = [f for f in available if f != 'num_submitters']

result_no_sub = run_experiment(
    name         = 'NO num_submitters',
    feature_list = no_submitters,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all,
    run_cv=False
)

# ============================================================
# EXPERIMENT 3: Remove LOF + consequence + num_submitters
# Most conservative ablation
# ============================================================
conservative_features = [
    f for f in available
    if f not in LOF_PROXY_FEATURES + ['num_submitters']
]

result_conservative = run_experiment(
    name         = 'CONSERVATIVE (no LOF + no num_submitters)',
    feature_list = conservative_features,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all
)

# ============================================================
# EXPERIMENT 4: dbNSFP scores ONLY
# Pure in-silico prediction scores — no engineered features
# ============================================================
DBNSFP_ONLY = [
    'REVEL_score',
    'CADD_phred',
    'AlphaMissense_score',
    'BayesDel_addAF',
    'BayesDel_noAF',
    'SIFT_score',
    'Polyphen2_HDIV',
    'Polyphen2_HVAR',
    'SpliceAI_DS_max',
    'phyloP17way',
    'phastCons17way',
    'GERP_RS',
    'DANN_score',
    'gnomAD_AF',
]

result_dbnsfp = run_experiment(
    name         = 'dbNSFP SCORES ONLY',
    feature_list = DBNSFP_ONLY,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all
)

# ============================================================
# EXPERIMENT 5: ClinVar features ONLY (no dbNSFP)
# Tests contribution of external scores
# ============================================================
CLINVAR_ONLY = [
    'variant_type_encoded', 'indel_length', 'is_frameshift',
    'allele_length_ratio', 'position', 'normalized_position',
    'in_exon11', 'cons_deletion', 'cons_duplication',
    'cons_other', 'has_protein_change', 'is_missense',
    'is_synonymous', 'is_splice_protein',
    'aa_position', 'in_RING_domain', 'in_BRCT_domain',
    'in_coiled_coil', 'in_disordered_region',
    'in_pathogenic_hotspot', 'domain_known',
    'is_canonical_splice_site', 'review_strength',
    'num_submitters'
]

result_clinvar = run_experiment(
    name         = 'ClinVar FEATURES ONLY (no dbNSFP)',
    feature_list = CLINVAR_ONLY,
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_tv=X_tv_all,   y_tv=y_tv_all,
    run_cv=False
)

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 65)
print("ABLATION SUMMARY TABLE")
print("=" * 65)

results = [r for r in [
    result_full, result_ablated, result_no_sub,
    result_conservative, result_dbnsfp, result_clinvar
] if r is not None]

print(f"\n  {'Experiment':<45} {'N feat':>7} {'Test AUC':>10} "
      f"{'Test F1':>9} {'Test MCC':>9} {'Gap':>8}")
print(f"  {'─'*45} {'─'*7} {'─'*10} {'─'*9} {'─'*9} {'─'*8}")

for r in results:
    print(f"  {r['name']:<45} {r['n_features']:>7} "
          f"{r['test']['ROC_AUC']:>10.4f} "
          f"{r['test']['F1']:>9.4f} "
          f"{r['test']['MCC']:>9.4f} "
          f"{r['gap']:>8.4f}")

print(f"""
  INTERPRETATION GUIDE:
  ─────────────────────────────────────────────────────────
  If ABLATED AUC > 0.95:
    → LOF features are helpful but not solely responsible
    → Performance is robust and defensible

  If ABLATED AUC drops to 0.80–0.94:
    → LOF features drive most performance
    → Missense-only model is the honest contribution

  If dbNSFP-only AUC > 0.90:
    → In-silico scores alone are strong enough
    → Feature engineering adds incremental value

  If ClinVar-only AUC is within 0.02 of full model:
    → dbNSFP scores not contributing much
    → Coverage too sparse (17% CADD, 7% REVEL)
  ─────────────────────────────────────────────────────────
""")
print("→ Paste full output — this determines paper framing")