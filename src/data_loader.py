# ============================================================
# PHASE 1 (EXPANDED) — Data Loading, Filtering, EDA
# File: src/data_loader.py
#
# Expansion strategy (confirmed from audit):
#   Tier 0 (current): expert panel + multi-submitter → 4,741
#   Tier 1 (new):     + single submitter WITH criteria → 7,475
#   Tier 1 + consistency filter → ~6,500–7,000 (clean)
#   + BRCA Exchange ENIGMA variants → ~8,500–9,500
#
# Label noise controls for new variants:
#   1. ClinSigSimple must be {0, 1} only
#   2. No variant with conflicting submissions
#   3. LastEvaluated >= 2015 (post-ACMG guidelines)
#   4. Consistency check: all submitters agree on P or B
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/external',  exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/vus',     exist_ok=True)
os.makedirs('models/calibrated', exist_ok=True)

print("=" * 65)
print("PHASE 1 (EXPANDED): Data Loading and Filtering")
print("=" * 65)

# ── Load raw ClinVar ──────────────────────────────────────────────────────────
RAW_PATH = 'data/raw/variant_summary.txt'
if os.path.exists(RAW_PATH):
    df = pd.read_csv(RAW_PATH, sep='\t', low_memory=False)
elif os.path.exists(RAW_PATH + '.gz'):
    df = pd.read_csv(RAW_PATH + '.gz', sep='\t',
                     low_memory=False, compression='gzip')
else:
    raise FileNotFoundError("variant_summary.txt not found in data/raw/")

print(f"\n[1] Total variants loaded       : {len(df):,}")

# ── BRCA1 + GRCh38 + type filter (unchanged) ─────────────────────────────────
df = df[df['GeneSymbol'] == 'BRCA1'].copy()
print(f"[2] After BRCA1 filter          : {len(df):,}")

df = df[df['Assembly'] == 'GRCh38'].copy()
print(f"[3] After GRCh38 filter         : {len(df):,}")

allowed_types = [
    'single nucleotide variant',
    'Deletion', 'Insertion', 'Duplication', 'Indel'
]
df = df[df['Type'].isin(allowed_types)].copy()
print(f"[4] After variant type filter   : {len(df):,}")

# ── EXPANDED review status tiers ─────────────────────────────────────────────
TIER_0 = [  # original production
    'practice guideline',
    'reviewed by expert panel',
    'criteria provided, multiple submitters, no conflicts'
]
TIER_1 = TIER_0 + [
    'criteria provided, single submitter'
]

df_tier0 = df[df['ReviewStatus'].isin(TIER_0)].copy()
df_tier1 = df[df['ReviewStatus'].isin(TIER_1)].copy()

print(f"\n[5] Tier 0 (expert panel only)  : {len(df_tier0):,}")
print(f"    Tier 1 (+ single submitter) : {len(df_tier1):,}")

# ── Remove conflicts ──────────────────────────────────────────────────────────
df_tier1 = df_tier1[
    ~df_tier1['ClinicalSignificance'].str.contains(
        'conflict', case=False, na=False
    )
].copy()
print(f"[6] After conflict removal      : {len(df_tier1):,}")

# ── NOISE CONTROL 1: ClinSigSimple must be 0 or 1 ────────────────────────────
# ClinSigSimple: 1=pathogenic, 0=benign, -1=VUS, others=mixed
if 'ClinSigSimple' in df_tier1.columns:
    before = len(df_tier1)
    df_tier1 = df_tier1[
        df_tier1['ClinSigSimple'].isin([0, 1, '0', '1'])
    ].copy()
    print(f"[7] After ClinSigSimple filter  : {len(df_tier1):,}  "
          f"(removed {before - len(df_tier1):,} ambiguous)")
else:
    print(f"[7] ClinSigSimple column not found — skipping")

# ── NOISE CONTROL 2: LastEvaluated >= 2015 for single-submitter only ─────────
# Only apply date filter to the NEW variants (single submitter tier)
# Expert panel variants are always valid regardless of date
if 'LastEvaluated' in df_tier1.columns:
    df_tier1['LastEvaluated'] = pd.to_datetime(
        df_tier1['LastEvaluated'], errors='coerce'
    )
    single_sub_mask = (
        df_tier1['ReviewStatus'] == 'criteria provided, single submitter'
    )
    old_single_sub_mask = (
        single_sub_mask &
        df_tier1['LastEvaluated'].notna() &
        (df_tier1['LastEvaluated'].dt.year < 2015)
    )
    n_removed_old = old_single_sub_mask.sum()
    df_tier1 = df_tier1[~old_single_sub_mask].copy()
    print(f"[8] After date filter (single sub >= 2015): {len(df_tier1):,}  "
          f"(removed {n_removed_old:,} pre-2015 single-submitter)")
else:
    print(f"[8] LastEvaluated column not found — skipping date filter")

# ── NOISE CONTROL 3: Consistency check for single submitters ─────────────────
# For single-submitter variants, verify that no OTHER submission
# for the same VariationID has a conflicting classification.
# A variant classified P by one submitter and B by another = unreliable.
print(f"\n[9] Consistency check for single-submitter variants...")

single_sub_ids = df_tier1[
    df_tier1['ReviewStatus'] == 'criteria provided, single submitter'
]['VariationID'].unique()

# Among all BRCA1 variants (any review status), find VariationIDs
# that have BOTH pathogenic AND benign submissions
all_brca1_clinvar = df[df['ReviewStatus'].isin(TIER_1)].copy()

PATH_TERMS   = ['Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic']
BENIGN_TERMS = ['Benign', 'Likely benign', 'Benign/Likely benign']

def has_conflict_across_submitters(group):
    """Return True if variant has both P and B among all submissions."""
    sigs = set(group['ClinicalSignificance'].values)
    has_p = any(s in PATH_TERMS for s in sigs)
    has_b = any(s in BENIGN_TERMS for s in sigs)
    return has_p and has_b

# Group by VariationID across all submissions
print(f"    Checking {len(single_sub_ids):,} single-submitter VariationIDs...")
conflict_ids = set()
for vid, grp in all_brca1_clinvar.groupby('VariationID'):
    if vid in single_sub_ids:
        if has_conflict_across_submitters(grp):
            conflict_ids.add(vid)

print(f"    VariationIDs with cross-submitter conflicts: {len(conflict_ids):,}")

# Remove conflicted single-submitter variants
before_consistency = len(df_tier1)
df_tier1 = df_tier1[
    ~(
        (df_tier1['ReviewStatus'] == 'criteria provided, single submitter') &
        (df_tier1['VariationID'].isin(conflict_ids))
    )
].copy()
print(f"    After consistency filter: {len(df_tier1):,}  "
      f"(removed {before_consistency - len(df_tier1):,})")

# ── Separate VUS ──────────────────────────────────────────────────────────────
df_vus = df_tier1[
    df_tier1['ClinicalSignificance'].str.contains(
        'Uncertain significance', case=False, na=False
    )
].copy()
print(f"\n[10] VUS variants (held out)    : {len(df_vus):,}")

df_tier1 = df_tier1[
    ~df_tier1['ClinicalSignificance'].str.contains(
        'Uncertain significance', case=False, na=False
    )
].copy()

# ── Assign binary labels ──────────────────────────────────────────────────────
def assign_label(sig):
    if sig in PATH_TERMS:
        return 1
    elif sig in BENIGN_TERMS:
        return 0
    return np.nan

df_tier1['label'] = df_tier1['ClinicalSignificance'].apply(assign_label)
df_train_pool = df_tier1.dropna(subset=['label']).copy()
df_train_pool['label'] = df_train_pool['label'].astype(int)

print(f"\n[11] Training pool (P+B)        : {len(df_train_pool):,}")
vc = df_train_pool['label'].value_counts()
print(f"     Pathogenic (1) : {vc.get(1, 0):,}")
print(f"     Benign     (0) : {vc.get(0, 0):,}")
print(f"     Imbalance ratio (B:P): {vc.get(0,1)/vc.get(1,1):.3f}")

# ── Remove duplicates ─────────────────────────────────────────────────────────
before_dedup = len(df_train_pool)
df_train_pool = df_train_pool.drop_duplicates(
    subset=['Chromosome', 'Start', 'Stop',
            'ReferenceAllele', 'AlternateAllele']
).copy()
print(f"\n[12] After deduplication        : {len(df_train_pool):,}  "
      f"(removed {before_dedup - len(df_train_pool):,} duplicates)")

# ── Review tier breakdown of final training pool ──────────────────────────────
print(f"\n[13] Review tier breakdown (final training pool):")
tier_breakdown = df_train_pool['ReviewStatus'].value_counts()
for status, count in tier_breakdown.items():
    tier = 'HIGH CONFIDENCE' if status in TIER_0 else 'SINGLE SUBMITTER'
    pct  = count / len(df_train_pool) * 100
    print(f"     [{tier}] {status[:50]}: {count:,} ({pct:.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
df_train_pool.to_csv('data/processed/brca1_filtered.csv', index=False)
df_vus.to_csv('data/processed/vus_set.csv', index=False)

print(f"\n[14] Files saved:")
print(f"     data/processed/brca1_filtered.csv  ({len(df_train_pool):,} rows)")
print(f"     data/processed/vus_set.csv         ({len(df_vus):,} rows)")

# ── EDA ───────────────────────────────────────────────────────────────────────
print("\n[15] Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BRCA1 ClinVar Expanded Dataset — Phase 1 EDA',
             fontsize=14, fontweight='bold')

ax = axes[0, 0]
vc.plot(kind='bar', ax=ax, color=['#2196F3', '#F44336'], edgecolor='black')
ax.set_title('Class Distribution')
ax.set_xticklabels(['Benign (0)', 'Pathogenic (1)'], rotation=0)
ax.set_ylabel('Count')
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{int(bar.get_height()):,}', ha='center', fontsize=10)

ax = axes[0, 1]
tier_breakdown.plot(kind='barh', ax=ax, color='#FF9800', edgecolor='black')
ax.set_title('Review Status (Training Pool)')
ax.set_xlabel('Count')

ax = axes[0, 2]
df_train_pool['Type'].value_counts().plot(
    kind='bar', ax=ax, color='#4CAF50', edgecolor='black')
ax.set_title('Variant Type Distribution')
ax.tick_params(axis='x', rotation=30)

ax = axes[1, 0]
cs = df_train_pool['ClinicalSignificance'].value_counts()
cs.plot(kind='barh', ax=ax, color='#9C27B0', edgecolor='black')
ax.set_title('Clinical Significance Breakdown')

ax = axes[1, 1]
if 'LastEvaluated' in df_train_pool.columns:
    years = df_train_pool['LastEvaluated'].dt.year.dropna()
    years.hist(bins=20, ax=ax, color='#607D8B', edgecolor='white')
    ax.set_title('Last Evaluated Year Distribution')
    ax.set_xlabel('Year')
else:
    ax.set_visible(False)

ax = axes[1, 2]
cross_tab = pd.crosstab(
    df_train_pool['Type'], df_train_pool['label']
)
cross_tab.columns = ['Benign', 'Pathogenic']
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('Variant Type vs Class')

plt.tight_layout()
plt.savefig('results/figures/phase1_eda_expanded.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("     results/figures/phase1_eda_expanded.png saved")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 1 (EXPANDED) COMPLETE")
print("=" * 65)
print(f"  Training variants (P+B)  : {len(df_train_pool):,}")
print(f"  VUS variants (held out)  : {len(df_vus):,}")
print(f"  Pathogenic               : {vc.get(1,0):,}")
print(f"  Benign                   : {vc.get(0,0):,}")
print(f"  Class imbalance (B:P)    : {vc.get(0,1)/vc.get(1,1):.3f}")
print(f"\n  Quality tiers present:")
hc_n   = (df_train_pool['ReviewStatus'].isin(TIER_0)).sum()
tier1_n = len(df_train_pool) - hc_n
print(f"    High confidence (3+ stars) : {hc_n:,} ({hc_n/len(df_train_pool)*100:.1f}%)")
print(f"    Single submitter (2 stars) : {tier1_n:,} ({tier1_n/len(df_train_pool)*100:.1f}%)")
print(f"\n  ClinVar features only AUC  : ~0.88–0.92 (expected)")
print(f"  With dbNSFP scores AUC     : ~0.97+")
print("=" * 65)
print("\n→ Proceed to Phase 2: Feature Engineering")
print("→ NOTE: Re-run Phase 2, Phase 3 (dbNSFP fetch), and Phase 4")
print("        with the expanded dataset for updated metrics.")