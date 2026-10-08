# ============================================================
# PHASE 3 — STEP 2 (FINAL): Update feature_cols.json
# File: src/update_features.py
# ============================================================

import pandas as pd
import numpy as np
import json

print("=" * 60)
print("PHASE 3 STEP 2: Updating Feature Column List")
print("=" * 60)

BASE_CLINVAR_FEATURES = [
    'variant_type_encoded', 'indel_length', 'is_frameshift',
    'allele_length_ratio', 'position', 'normalized_position',
    'in_exon11', 'cons_deletion', 'cons_duplication', 'cons_frameshift',
    'cons_insertion', 'cons_nonsense', 'cons_snv', 'cons_splice',
    'cons_other', 'has_protein_change', 'is_missense', 'is_nonsense',
    'is_frameshift_protein', 'is_synonymous', 'is_splice_protein',
    'aa_position', 'in_RING_domain', 'in_BRCT_domain', 'in_coiled_coil',
    'in_disordered_region', 'in_pathogenic_hotspot', 'domain_known',
    'is_likely_lof', 'is_canonical_splice_site',
    'review_strength', 'num_submitters'
]

# All external scores now with correct column names
EXTERNAL_SCORE_COLS = [
    'BayesDel_addAF',
    'BayesDel_noAF',
    'REVEL_score',
    'CADD_phred',
    'SIFT_score',
    'Polyphen2_HDIV',
    'Polyphen2_HVAR',
    'SpliceAI_DS_max',
    'phyloP17way',
    'phastCons17way',
    'GERP_RS',
    'AlphaMissense_score',
    'DANN_score',
    'gnomAD_AF'
]

df = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)
print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

final_features = BASE_CLINVAR_FEATURES.copy()
added = []

print(f"\n  {'Column':<28} {'Coverage':>12}  {'Status'}")
print(f"  {'─'*28} {'─'*12}  {'─'*10}")

for col in EXTERNAL_SCORE_COLS:
    if col in df.columns:
        n   = df[col].notna().sum()
        pct = n / len(df) * 100
        if n > 0:
            final_features.append(col)
            added.append(col)
            status = '✓ ADDED'
        else:
            status = '✗ empty'
        print(f"  {col:<28} {n:>5}/{len(df):>5} ({pct:4.1f}%)  {status}")
    else:
        print(f"  {col:<28} {'not found':>12}  ✗ missing")

print(f"\n  Base ClinVar features : {len(BASE_CLINVAR_FEATURES)}")
print(f"  External added        : {len(added)}")
print(f"  Total features        : {len(final_features)}")
print(f"  Added columns         : {added}")

with open('data/processed/feature_cols.json', 'w') as f:
    json.dump(final_features, f, indent=2)

print(f"\n  ✓ data/processed/feature_cols.json saved")

# ── AUC expectation based on what we have ────────────────────────────────────
has_bayesdel = any('BayesDel' in c for c in added)
has_revel    = 'REVEL_score' in added
has_cadd     = 'CADD_phred' in added
has_alpha    = 'AlphaMissense_score' in added

if has_bayesdel and has_revel:
    auc_expect = "~0.97+"
elif has_revel or has_cadd:
    auc_expect = "~0.93–0.96"
elif has_alpha:
    auc_expect = "~0.92–0.95"
else:
    auc_expect = "~0.88–0.92 (ClinVar only)"

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  FINAL FEATURE SET                                   │
  │                                                      │
  │  ClinVar features     : {len(BASE_CLINVAR_FEATURES):<3}                           │
  │  External scores      : {len(added):<3}                           │
  │  Total                : {len(final_features):<3}                           │
  │                                                      │
  │  BayesDel available   : {'YES' if has_bayesdel else 'NO — only for missense SNVs'}                  │
  │  REVEL available      : {'YES' if has_revel else 'NO'}                           │
  │  CADD available       : {'YES' if has_cadd else 'NO'}                           │
  │  AlphaMissense avail  : {'YES' if has_alpha else 'NO'}                           │
  │                                                      │
  │  Expected AUC         : {auc_expect:<28} │
  │                                                      │
  │  Note: Low coverage on training is expected.         │
  │  Frameshift/nonsense/splice variants do NOT get      │
  │  REVEL/BayesDel scores (these tools only score       │
  │  missense SNVs by design). The model will use        │
  │  NaN → imputed median for non-missense variants,     │
  │  which is scientifically valid and standard in       │
  │  published pathogenicity prediction pipelines.       │
  └──────────────────────────────────────────────────────┘
""")
print("→ Phase 3 complete. Proceed to Phase 4: Train/Test Split")