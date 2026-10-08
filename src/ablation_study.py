# ============================================================
# ABLATION STUDY — 6,009 variant expanded dataset
# File: src/ablation_study.py
# ============================================================

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score
)
import xgboost as xgb

print("=" * 65)
print("ABLATION STUDY — BRCA1 Expanded Dataset (6,009 variants)")
print("=" * 65)

df = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)

with open('data/processed/feature_cols_clean.json') as f:
    ALL_FEATURES = json.load(f)

LABEL     = 'label'
available = [c for c in ALL_FEATURES if c in df.columns]

X = df[available].copy()
y = df[LABEL].astype(int).copy()

# Fixed splits — same seed as production
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.125, random_state=42, stratify=y_tv
)
X_tv_all = pd.concat([X_train, X_val], ignore_index=True)
y_tv_all = pd.concat([y_train, y_val], ignore_index=True)

scale_pw = float((y_train==0).sum()) / float((y_train==1).sum())

def run_experiment(name, feature_list):
    feat = [f for f in feature_list if f in X_train.columns]
    if len(feat) == 0:
        print(f"\n  ERROR: No features for {name}")
        return None

    imp = SimpleImputer(strategy='median')
    Xtr = pd.DataFrame(imp.fit_transform(X_train[feat]), columns=feat)
    Xva = pd.DataFrame(imp.transform(X_val[feat]),       columns=feat)
    Xte = pd.DataFrame(imp.transform(X_test[feat]),      columns=feat)
    Xtv = pd.DataFrame(imp.transform(X_tv_all[feat]),    columns=feat)

    spw = float((y_train==0).sum()) / float((y_train==1).sum())

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42,
        objective='binary:logistic', verbosity=0, n_jobs=-1
    )
    model.fit(Xtr.values, y_train.values)

    def metrics(y_true, y_prob, thresh=0.5):
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'ROC_AUC' : round(roc_auc_score(y_true, y_prob), 4),
            'PR_AUC'  : round(average_precision_score(y_true, y_prob), 4),
            'Accuracy': round(accuracy_score(y_true, y_pred), 4),
            'F1'      : round(f1_score(y_true, y_pred, zero_division=0), 4),
            'MCC'     : round(matthews_corrcoef(y_true, y_pred), 4),
        }

    # Threshold from validation
    from sklearn.metrics import precision_recall_curve
    val_prob = model.predict_proba(Xva.values)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val.values, val_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    thresh = float(thresholds[np.argmax(f1s[:-1])])

    tr_m = metrics(y_train.values, model.predict_proba(Xtr.values)[:,1], thresh)
    va_m = metrics(y_val.values,   model.predict_proba(Xva.values)[:,1], thresh)
    te_m = metrics(y_test.values,  model.predict_proba(Xte.values)[:,1], thresh)

    # 5-fold CV
    cv_aucs = []
    cv_skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in cv_skf.split(Xtv.values, y_tv_all.values):
        fold_imp = SimpleImputer(strategy='median')
        Xf_tr = fold_imp.fit_transform(Xtv.values[tr_idx])
        Xf_va = fold_imp.transform(Xtv.values[va_idx])
        yf_tr = y_tv_all.values[tr_idx]
        yf_va = y_tv_all.values[va_idx]
        fm = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float((yf_tr==0).sum())/float((yf_tr==1).sum()),
            random_state=42, objective='binary:logistic',
            verbosity=0, n_jobs=-1
        )
        fm.fit(Xf_tr, yf_tr)
        cv_aucs.append(roc_auc_score(yf_va, fm.predict_proba(Xf_va)[:,1]))

    cv_mean = round(np.mean(cv_aucs), 4)
    cv_std  = round(np.std(cv_aucs),  4)

    # Missense-only
    test_idx  = X_test.index
    df_ref    = df.loc[test_idx].reset_index(drop=True)
    mis_mask  = df_ref['is_missense'].values == 1
    y_mis     = y_test.values[mis_mask]
    p_mis     = model.predict_proba(Xte.values)[:,1][mis_mask]
    mis_auc   = round(roc_auc_score(y_mis, p_mis), 4) \
                if y_mis.sum() > 0 and (y_mis==0).sum() > 0 else None
    mis_mcc   = round(matthews_corrcoef(y_mis, (p_mis>=thresh).astype(int)), 4) \
                if mis_auc else None

    gap = round(tr_m['ROC_AUC'] - te_m['ROC_AUC'], 4)

    return {
        'name'      : name,
        'n_features': len(feat),
        'train'     : tr_m,
        'val'       : va_m,
        'test'      : te_m,
        'cv_mean'   : cv_mean,
        'cv_std'    : cv_std,
        'gap'       : gap,
        'mis_auc'   : mis_auc,
        'mis_mcc'   : mis_mcc,
        'mis_n'     : int(mis_mask.sum()),
        'thresh'    : round(thresh, 4),
    }


# ── EXPERIMENT 0: Full model ──────────────────────────────────────────────────
r0 = run_experiment('FULL MODEL (48 features)', available)

# ── EXPERIMENT A: Remove LOF + consequence proxies ────────────────────────────
REMOVE_A = [
    'is_likely_lof', 'cons_snv', 'cons_splice', 'cons_frameshift',
]
feat_A = [f for f in available if f not in REMOVE_A]
rA = run_experiment('EXPERIMENT A: No LOF/consequence proxies', feat_A)

# ── EXPERIMENT B: dbNSFP biological scores only ───────────────────────────────
DBNSFP_ONLY = [
    'REVEL_score', 'CADD_phred', 'AlphaMissense_score',
    'BayesDel_addAF', 'BayesDel_noAF',
    'SIFT_score', 'Polyphen2_HDIV', 'Polyphen2_HVAR',
    'SpliceAI_DS_max', 'GERP_RS',
    'phyloP17way', 'phastCons17way',
    'DANN_score', 'gnomAD_AF'
]
rB = run_experiment('EXPERIMENT B: dbNSFP scores only', DBNSFP_ONLY)

# ── EXPERIMENT C: ClinVar engineered only (no dbNSFP) ────────────────────────
CLINVAR_ONLY = [f for f in available if f not in DBNSFP_ONLY]
rC = run_experiment('EXPERIMENT C: ClinVar engineered only', CLINVAR_ONLY)

# ── EXPERIMENT D: Most conservative (no LOF + no num_submitters) ─────────────
REMOVE_D = REMOVE_A + ['num_submitters', 'cons_nonsense', 'is_nonsense',
                        'is_frameshift_protein', 'is_frameshift']
feat_D = [f for f in available if f not in REMOVE_D]
rD = run_experiment('EXPERIMENT D: Conservative (no LOF, no num_sub)', feat_D)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("ABLATION RESULTS — FULL OUTPUT")
print("=" * 65)

for r in [r0, rA, rB, rC, rD]:
    if r is None:
        continue
    print(f"\n{'─'*65}")
    print(f"  {r['name']}")
    print(f"  Features: {r['n_features']}  |  Threshold: {r['thresh']}")
    print(f"{'─'*65}")
    print(f"  {'Split':<12} {'ROC-AUC':>9} {'PR-AUC':>9} "
          f"{'Accuracy':>10} {'F1':>9} {'MCC':>9}")
    print(f"  {'─'*12} {'─'*9} {'─'*9} {'─'*10} {'─'*9} {'─'*9}")
    for split, m in [('Train', r['train']),
                     ('Validation', r['val']),
                     ('Test', r['test'])]:
        print(f"  {split:<12} {m['ROC_AUC']:>9.4f} {m['PR_AUC']:>9.4f} "
              f"{m['Accuracy']:>10.4f} {m['F1']:>9.4f} {m['MCC']:>9.4f}")
    print(f"\n  CV ROC-AUC : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
    print(f"  Train-Test Gap : {r['gap']}  "
          f"({'Excellent' if r['gap']<0.03 else 'Acceptable'})")
    if r['mis_auc']:
        print(f"  Missense-only  : AUC={r['mis_auc']}  MCC={r['mis_mcc']}  "
              f"n={r['mis_n']}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("SUMMARY COMPARISON TABLE")
print("=" * 65)
print(f"\n  {'Experiment':<45} {'N':>4} {'Test AUC':>9} "
      f"{'Test F1':>8} {'Test MCC':>9} {'CV AUC':>10} {'Gap':>7}")
print(f"  {'─'*45} {'─'*4} {'─'*9} {'─'*8} {'─'*9} {'─'*10} {'─'*7}")
for r in [r0, rA, rB, rC, rD]:
    if r is None:
        continue
    print(f"  {r['name'][:45]:<45} {r['n_features']:>4} "
          f"{r['test']['ROC_AUC']:>9.4f} "
          f"{r['test']['F1']:>8.4f} "
          f"{r['test']['MCC']:>9.4f} "
          f"{r['cv_mean']:>9.4f}±{r['cv_std']:.4f} "
          f"{r['gap']:>7.4f}")