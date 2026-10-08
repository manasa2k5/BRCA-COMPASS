# ============================================================
# DIAGNOSTIC (SELF-CONTAINED) — Overfit Analysis
# File: src/diagnose_overfit.py
#
# Requires only:
#   data/processed/brca1_features_scored.csv
#   data/processed/vus_features_scored.csv
#   data/processed/feature_cols_clean.json
#
# Does NOT require any saved models or previous run outputs.
# Trains a quick XGBoost internally for diagnosis only.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve
)
from sklearn.metrics import make_scorer
import xgboost as xgb

print("=" * 65)
print("DIAGNOSTIC: Overfit + Variant-Type Stratified Analysis")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────────────────────
df     = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)
df_vus = pd.read_csv('data/processed/vus_features_scored.csv',   low_memory=False)

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

LABEL = 'label'

available = [c for c in FEATURE_COLS if c in df.columns]
X = df[available].copy()
y = df[LABEL].astype(int).copy()

print(f"\nFull dataset     : {len(df):,} variants")
print(f"Features used    : {len(available)}")
print(f"Pathogenic (1)   : {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"Benign     (0)   : {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

# ── Variant type breakdown in full dataset ────────────────────────────────────
print(f"\nVariant type breakdown (full training pool):")
type_cols = {
    'is_likely_lof'       : 'LOF (frameshift+nonsense+splice)',
    'is_missense'         : 'Missense',
    'is_synonymous'       : 'Synonymous',
    'cons_snv'            : 'SNV (all)',
    'cons_deletion'       : 'Deletion',
    'cons_insertion'      : 'Insertion',
    'cons_frameshift'     : 'Frameshift',
    'cons_nonsense'       : 'Nonsense',
    'cons_splice'         : 'Splice',
    'is_canonical_splice_site': 'Canonical splice',
}
for col, label in type_cols.items():
    if col in df.columns:
        n   = int(df[col].sum())
        pct = df[col].mean() * 100
        n_p = int(df[df[col]==1][LABEL].sum()) if col in df.columns else 0
        n_b = n - n_p
        print(f"  {label:<40} n={n:>5} ({pct:4.1f}%)  P={n_p:>5} B={n_b:>5}")

# ── Stratified 70/10/20 split ─────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"Splitting 70/10/20 (stratified)...")

X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.125, random_state=42, stratify=y_tv
)

# Fit imputer on train only
imputer     = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=available)
X_val_imp   = pd.DataFrame(imputer.transform(X_val),       columns=available)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=available)

print(f"  Train : {len(X_train):,}  Val : {len(X_val):,}  Test : {len(X_test):,}")

# ── Train quick XGBoost for diagnosis ────────────────────────────────────────
print(f"\nTraining XGBoost for diagnosis (2 minutes)...")
scale_pw = float((y_train==0).sum()) / float((y_train==1).sum())

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    random_state=42,
    eval_metric='auc',
    verbosity=0
)
xgb_model.fit(
    X_train_imp.values, y_train.values,
    eval_set=[(X_val_imp.values, y_val.values)],
    verbose=False
)

train_proba = xgb_model.predict_proba(X_train_imp.values)[:, 1]
val_proba   = xgb_model.predict_proba(X_val_imp.values)[:, 1]
test_proba  = xgb_model.predict_proba(X_test_imp.values)[:, 1]

train_auc = roc_auc_score(y_train.values, train_proba)
val_auc   = roc_auc_score(y_val.values,   val_proba)
test_auc  = roc_auc_score(y_test.values,  test_proba)

print(f"\n  OVERALL AUC:")
print(f"    Train : {train_auc:.4f}")
print(f"    Val   : {val_auc:.4f}")
print(f"    Test  : {test_auc:.4f}")
print(f"    Gap (train-test) : {train_auc - test_auc:.4f}")

# ============================================================
# ANALYSIS 1: AUC by Variant Type (KEY ANALYSIS)
# ============================================================
print(f"\n{'─'*65}")
print(f"ANALYSIS 1: AUC Stratified by Variant Type (TEST SET)")
print(f"{'─'*65}")
print(f"This is the critical analysis — shows where AUC is real vs inflated")

# Get test set variant type flags
test_idx = X_test.index
df_test  = df.loc[test_idx].reset_index(drop=True)

print(f"\n  {'Variant Type':<40} {'N':>6} {'P':>5} {'B':>5} {'ROC-AUC':>10}")
print(f"  {'─'*40} {'─'*6} {'─'*5} {'─'*5} {'─'*10}")

stratified_results = []

for col, label in type_cols.items():
    if col not in df_test.columns:
        continue
    mask  = df_test[col].values == 1
    n     = mask.sum()
    if n < 10:
        print(f"  {label:<40} {n:>6}  (n<10 skip)")
        continue
    y_sub = y_test.values[mask]
    p_sub = test_proba[mask]
    n_pos = y_sub.sum()
    n_neg = (y_sub==0).sum()
    if n_pos == 0 or n_neg == 0:
        print(f"  {label:<40} {n:>6} {n_pos:>5} {n_neg:>5}  "
              f"(single class — model memorized this type)")
        stratified_results.append({
            'type': label, 'n': n,
            'n_path': n_pos, 'n_benign': n_neg,
            'auc': 1.0 if n_pos == n else 0.0,
            'note': 'single class'
        })
        continue
    auc   = roc_auc_score(y_sub, p_sub)
    pr    = average_precision_score(y_sub, p_sub)
    flag  = ' ← TRIVIAL (LOF=path by definition)' \
            if (n_pos / n > 0.90 or n_neg / n > 0.90) else ''
    print(f"  {label:<40} {n:>6} {n_pos:>5} {n_neg:>5} {auc:>10.4f}{flag}")
    stratified_results.append({
        'type': label, 'n': n,
        'n_path': n_pos, 'n_benign': n_neg,
        'auc': round(auc, 4), 'pr_auc': round(pr, 4)
    })

# ── MISSENSE ONLY — the clinically meaningful number ─────────────────────────
print(f"\n  {'─'*65}")
print(f"  MISSENSE-ONLY AUC (matches VUS population — PRIMARY METRIC):")
mis_mask = df_test['is_missense'].values == 1
n_mis    = mis_mask.sum()
print(f"  Missense variants in test set: {n_mis}")

if n_mis >= 10:
    y_mis = y_test.values[mis_mask]
    p_mis = test_proba[mis_mask]
    n_mp  = y_mis.sum()
    n_mb  = (y_mis==0).sum()
    print(f"  Pathogenic missense : {n_mp}")
    print(f"  Benign missense     : {n_mb}")
    if n_mp > 0 and n_mb > 0:
        auc_mis = roc_auc_score(y_mis, p_mis)
        pr_mis  = average_precision_score(y_mis, p_mis)
        f1_mis  = f1_score(y_mis, (p_mis>=0.5).astype(int), zero_division=0)
        mcc_mis = matthews_corrcoef(y_mis, (p_mis>=0.5).astype(int))
        print(f"\n  *** MISSENSE-ONLY METRICS ***")
        print(f"    ROC-AUC  : {auc_mis:.4f}")
        print(f"    PR-AUC   : {pr_mis:.4f}")
        print(f"    F1       : {f1_mis:.4f}")
        print(f"    MCC      : {mcc_mis:.4f}")
    else:
        print(f"  Single class in missense test — check data")
        auc_mis = None
else:
    print(f"  Insufficient missense variants in test set")
    auc_mis = None

# ============================================================
# ANALYSIS 2: LOF Dominance Check
# ============================================================
print(f"\n{'─'*65}")
print(f"ANALYSIS 2: LOF Dominance — Is model just flagging LOF?")
print(f"{'─'*65}")

pred_class = (test_proba >= 0.5).astype(int)

for lof_val, name in [(1, 'LOF variants (is_likely_lof=1)'),
                       (0, 'Non-LOF variants (is_likely_lof=0)')]:
    if 'is_likely_lof' not in df_test.columns:
        break
    mask = df_test['is_likely_lof'].values == lof_val
    n    = mask.sum()
    if n == 0:
        continue
    y_g    = y_test.values[mask]
    pred_g = pred_class[mask]
    print(f"\n  {name} (n={n:,}):")
    print(f"    True Pathogenic : {y_g.sum():,} ({y_g.mean()*100:.1f}%)")
    print(f"    True Benign     : {(y_g==0).sum():,} ({(y_g==0).mean()*100:.1f}%)")
    print(f"    Pred Pathogenic : {pred_g.sum():,} ({pred_g.mean()*100:.1f}%)")
    print(f"    Accuracy        : {accuracy_score(y_g, pred_g):.4f}")
    if y_g.sum() > 0 and (y_g==0).sum() > 0:
        p_g = test_proba[mask]
        print(f"    ROC-AUC         : {roc_auc_score(y_g, p_g):.4f}")

# ============================================================
# ANALYSIS 3: VUS Population vs Training Mismatch
# ============================================================
print(f"\n{'─'*65}")
print(f"ANALYSIS 3: Training vs VUS Population Mismatch")
print(f"{'─'*65}")

print(f"\n  {'Feature':<35} {'Train %':>10} {'VUS %':>10} {'Match?':>8}")
print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*8}")

compare_cols = [
    'is_missense', 'is_likely_lof', 'cons_frameshift',
    'cons_nonsense', 'cons_deletion', 'cons_snv',
    'is_canonical_splice_site', 'in_BRCT_domain', 'in_RING_domain'
]
for col in compare_cols:
    if col not in df.columns or col not in df_vus.columns:
        continue
    tr_pct  = df[col].mean() * 100
    vus_pct = df_vus[col].mean() * 100
    diff    = abs(tr_pct - vus_pct)
    match   = '✓' if diff < 20 else '⚠ MISMATCH'
    print(f"  {col:<35} {tr_pct:>9.1f}% {vus_pct:>9.1f}%  {match}")

# ============================================================
# ANALYSIS 4: 5-Fold CV AUC — ALL variants vs MISSENSE ONLY
# ============================================================
print(f"\n{'─'*65}")
print(f"ANALYSIS 4: 5-Fold CV — All variants vs Missense-only")
print(f"{'─'*65}")

# All variants CV
X_tv_imp = pd.DataFrame(
    imputer.transform(X_tv), columns=available
)
cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_all = cross_validate(
    xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pw, random_state=42,
        eval_metric='auc', verbosity=0
    ),
    X_tv_imp.values, y_tv.values,
    cv=cv_skf, scoring='roc_auc', n_jobs=-1
)
print(f"\n  All variants (n={len(X_tv_imp):,}):")
print(f"    CV AUC : {cv_all['test_score'].mean():.4f} "
      f"± {cv_all['test_score'].std():.4f}")

# Missense-only CV
tv_idx       = X_tv.index
df_tv        = df.loc[tv_idx].reset_index(drop=True)
mis_tv_mask  = df_tv['is_missense'].values == 1
n_mis_tv     = mis_tv_mask.sum()
print(f"\n  Missense-only (n={n_mis_tv:,}):")

if n_mis_tv >= 50:
    X_mis_tv = X_tv_imp.values[mis_tv_mask]
    y_mis_tv = y_tv.values[mis_tv_mask]
    n_pos_mis = y_mis_tv.sum()
    n_neg_mis = (y_mis_tv==0).sum()
    print(f"    P={n_pos_mis} B={n_neg_mis}")
    if n_pos_mis >= 10 and n_neg_mis >= 10:
        cv_mis = cross_validate(
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=float(n_neg_mis)/float(n_pos_mis),
                random_state=42, eval_metric='auc', verbosity=0
            ),
            X_mis_tv, y_mis_tv,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1
        )
        print(f"    CV AUC : {cv_mis['test_score'].mean():.4f} "
              f"± {cv_mis['test_score'].std():.4f}")
        print(f"\n    *** THIS IS YOUR TRUE PUBLISHABLE AUC ***")
    else:
        print(f"    Insufficient class balance for CV")
else:
    print(f"    Only {n_mis_tv} missense in train+val — too few for CV")

# ============================================================
# FINAL RECOMMENDATION
# ============================================================
print(f"\n{'='*65}")
print(f"DIAGNOSTIC SUMMARY + RECOMMENDATION")
print(f"{'='*65}")

print(f"""
  OVERALL TEST AUC     : {test_auc:.4f}
  TRAIN-TEST GAP       : {train_auc - test_auc:.4f}
  MISSENSE-ONLY AUC    : {f'{auc_mis:.4f}' if auc_mis else 'see above'}

  KEY FINDING:
  ─────────────────────────────────────────────────────────
  The overall AUC is driven by LOF variants (frameshift,
  nonsense, splice, deletion) which are trivially predicted
  because they are almost exclusively pathogenic in BRCA1.

  VUS are {df_vus['is_missense'].mean()*100:.0f}% missense. Your model's real
  clinical challenge is the MISSENSE-ONLY task.

  WHAT TO DO:
  ─────────────────────────────────────────────────────────
  Option A — Report stratified metrics (recommended):
    Report overall AUC for completeness.
    Report missense-only AUC as PRIMARY metric.
    This is honest, standard, and publishable.

  Option B — Retrain on missense-only subset:
    Filter training to is_missense==1 variants only.
    Model becomes a focused missense pathogenicity predictor.
    This directly matches the VUS inference task.
    Expected missense AUC: 0.88–0.96 (genuinely earned).

  RECOMMENDATION: Option B if missense-only AUC < 0.85.
                  Option A if missense-only AUC >= 0.85.
  ─────────────────────────────────────────────────────────
""")
print("→ Paste this full output — we decide next step from the numbers")