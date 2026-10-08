# ============================================================
# PHASE 2 — Feature Engineering (ClinVar Features Only)
# File: src/feature_engineer.py
# VERSION: Fixed for your ClinVar column structure
# Protein info extracted entirely from 'Name' (HGVS) column
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 2: Feature Engineering (ClinVar Features)")
print("=" * 60)

# ── Load filtered data ────────────────────────────────────────────────────────
df     = pd.read_csv('data/processed/brca1_filtered.csv', low_memory=False)
df_vus = pd.read_csv('data/processed/vus_set.csv', low_memory=False)

print(f"\nLoaded training pool : {len(df):,} variants")
print(f"Loaded VUS pool      : {len(df_vus):,} variants")
print(f"Columns available    : {len(df.columns)}")

# ── Quick look at Name column to confirm HGVS format ─────────────────────────
print(f"\nSample Name values:")
for v in df['Name'].dropna().head(5).values:
    print(f"  {v}")

# ============================================================
# HELPER: Extract protein notation from HGVS Name string
# Handles formats like:
#   NM_007294.3:c.5266dupC(p.Gln1756ProfsTer25)
#   NM_007294.4:c.181T>G (p.Cys61Gly)
#   NM_007294.3:c.68_69del (p.Glu23Valfs*17)
# ============================================================

def extract_hgvsp_from_name(name):
    """Extract p. notation from HGVS Name string."""
    if pd.isna(name):
        return None
    match = re.search(r'\(?(p\.[^\s\)]+)\)?', str(name))
    if match:
        return match.group(1)
    return None

def extract_cdna_from_name(name):
    """Extract c. notation from HGVS Name string."""
    if pd.isna(name):
        return None
    match = re.search(r'(c\.[^\s\(]+)', str(name))
    if match:
        return match.group(1)
    return None

# ============================================================
# CORE FEATURE ENGINEERING FUNCTION
# Applied identically to training pool AND VUS — zero leakage
# ============================================================

def engineer_features(df_input, dataset_name='dataset'):
    df = df_input.copy()
    print(f"\n{'─'*55}")
    print(f"Engineering: {dataset_name} ({len(df):,} variants)")
    print(f"{'─'*55}")

    # ── FEATURE 1: Variant Type Encoding ─────────────────────────────────────
    type_map = {
        'single nucleotide variant' : 0,
        'Deletion'                  : 1,
        'Insertion'                 : 2,
        'Duplication'               : 3,
        'Indel'                     : 4
    }
    df['variant_type_encoded'] = df['Type'].map(type_map).fillna(5).astype(int)
    print(f"  [F1]  variant_type_encoded          : done")

    # ── FEATURE 2: Genomic Position ───────────────────────────────────────────
    df['position'] = pd.to_numeric(df['Start'], errors='coerce')
    print(f"  [F2]  position                      : done")

    # ── FEATURE 3: Variant Length + Frameshift Flag ───────────────────────────
    df['ref_len']      = df['ReferenceAllele'].fillna('').apply(len)
    df['alt_len']      = df['AlternateAllele'].fillna('').apply(len)
    df['indel_length'] = (df['alt_len'] - df['ref_len']).abs()
    df['is_frameshift'] = (
        (df['indel_length'] > 0) & (df['indel_length'] % 3 != 0)
    ).astype(int)
    print(f"  [F3]  indel_length, is_frameshift    : done")

    # ── FEATURE 4: Allele Length Ratio ───────────────────────────────────────
    df['allele_length_ratio'] = df.apply(
        lambda r: r['alt_len'] / r['ref_len'] if r['ref_len'] > 0 else 1.0,
        axis=1
    ).clip(0, 10)
    print(f"  [F4]  allele_length_ratio            : done")

    # ── FEATURE 5: Extract p. and c. from Name column ────────────────────────
    df['hgvsp'] = df['Name'].apply(extract_hgvsp_from_name)
    df['hgvsc'] = df['Name'].apply(extract_cdna_from_name)
    hgvsp_found = df['hgvsp'].notna().sum()
    hgvsc_found = df['hgvsc'].notna().sum()
    print(f"  [F5]  HGVS extraction               : done")
    print(f"         p. notation found in : {hgvsp_found:,} / {len(df):,} variants")
    print(f"         c. notation found in : {hgvsc_found:,} / {len(df):,} variants")

    # ── FEATURE 6: Consequence from Name column ───────────────────────────────
    def extract_consequence(name):
        if pd.isna(name):
            return 'unknown'
        n = str(name).lower()
        # Check protein annotation first (more reliable)
        if 'fs' in n or 'frameshift' in n:
            return 'frameshift'
        elif re.search(r'p\.[a-z]{3}\d+ter', n) or re.search(r'p\.[a-z]{3}\d+\*', n):
            return 'nonsense'
        elif re.search(r'c\.\d+[\+\-]\d', n) or 'splice' in n:
            return 'splice'
        elif 'del' in n and 'ins' not in n and 'deletion' not in n.split('p.')[0] if 'p.' in n else 'del' in n:
            return 'deletion'
        elif 'ins' in n and 'del' not in n:
            return 'insertion'
        elif 'dup' in n:
            return 'duplication'
        elif '>' in n:
            return 'snv'
        else:
            return 'other'

    df['consequence'] = df['Name'].apply(extract_consequence)

    # One-hot encode consequence — ensure all categories present
    all_cons_cats = [
        'cons_deletion', 'cons_duplication', 'cons_frameshift',
        'cons_insertion', 'cons_nonsense', 'cons_other',
        'cons_snv', 'cons_splice', 'cons_unknown'
    ]
    consequence_dummies = pd.get_dummies(df['consequence'], prefix='cons')
    for col in all_cons_cats:
        if col not in consequence_dummies.columns:
            consequence_dummies[col] = 0
    consequence_dummies = consequence_dummies[all_cons_cats]
    df = pd.concat([
        df.reset_index(drop=True),
        consequence_dummies.reset_index(drop=True)
    ], axis=1)
    print(f"  [F6]  consequence (one-hot)          : done")
    print(f"         Distribution: {df['consequence'].value_counts().to_dict()}")

    # ── FEATURE 7: Protein Change Features from hgvsp ────────────────────────
    def parse_hgvsp(hgvsp):
        out = {
            'has_protein_change'    : 0,
            'is_missense'           : 0,
            'is_nonsense'           : 0,
            'is_frameshift_protein' : 0,
            'is_synonymous'         : 0,
            'is_splice_protein'     : 0,
            'aa_position'           : np.nan
        }
        if pd.isna(hgvsp) or str(hgvsp).strip() in ['-', '', 'nan', 'None']:
            return out

        p = str(hgvsp).strip()
        out['has_protein_change'] = 1
        p_lower = p.lower()

        if 'fs' in p_lower:
            out['is_frameshift_protein'] = 1
        elif p.endswith('*') or re.search(r'ter', p_lower):
            out['is_nonsense'] = 1
        elif '=' in p:
            out['is_synonymous'] = 1
        elif 'splice' in p_lower:
            out['is_splice_protein'] = 1
        elif re.search(r'p\.[A-Za-z]{1,3}\d+[A-Za-z]{1,3}', p):
            out['is_missense'] = 1

        # Extract amino acid position number
        pos_match = re.search(r'\d+', p)
        if pos_match:
            try:
                out['aa_position'] = int(pos_match.group())
            except:
                pass

        return out

    protein_features = df['hgvsp'].apply(parse_hgvsp)
    protein_df = pd.DataFrame(list(protein_features))
    df = pd.concat([
        df.reset_index(drop=True),
        protein_df.reset_index(drop=True)
    ], axis=1)
    print(f"  [F7]  protein change features       : done")
    print(f"         Missense       : {df['is_missense'].sum():,}")
    print(f"         Nonsense       : {df['is_nonsense'].sum():,}")
    print(f"         Frameshift     : {df['is_frameshift_protein'].sum():,}")
    print(f"         Synonymous     : {df['is_synonymous'].sum():,}")
    print(f"         Has p. change  : {df['has_protein_change'].sum():,}")

    # ── FEATURE 8: BRCA1 Domain Annotation ───────────────────────────────────
    # Source: UniProt P38398, ClinGen ENIGMA VCEP 2024
    # RING domain  : aa 1   – 109
    # Coiled coil  : aa 1391 – 1424
    # BRCT domain  : aa 1642 – 1863
    # Disordered   : linker regions between domains
    def annotate_domain(aa_pos):
        if pd.isna(aa_pos):
            return {
                'in_RING_domain'      : 0,
                'in_BRCT_domain'      : 0,
                'in_coiled_coil'      : 0,
                'in_disordered_region': 0,
                'domain_known'        : 0
            }
        aa = float(aa_pos)
        in_ring = int(1    <= aa <= 109)
        in_brct = int(1642 <= aa <= 1863)
        in_cc   = int(1391 <= aa <= 1424)
        in_dis  = int(
            (110 <= aa <= 1390) or (1425 <= aa <= 1641)
        )
        return {
            'in_RING_domain'      : in_ring,
            'in_BRCT_domain'      : in_brct,
            'in_coiled_coil'      : in_cc,
            'in_disordered_region': in_dis,
            'domain_known'        : 1
        }

    domain_feats = df['aa_position'].apply(annotate_domain)
    domain_df    = pd.DataFrame(list(domain_feats))
    df = pd.concat([
        df.reset_index(drop=True),
        domain_df.reset_index(drop=True)
    ], axis=1)
    print(f"  [F8]  domain annotation             : done")
    print(f"         In RING  : {df['in_RING_domain'].sum():,}")
    print(f"         In BRCT  : {df['in_BRCT_domain'].sum():,}")
    print(f"         In CC    : {df['in_coiled_coil'].sum():,}")

    # ── FEATURE 9: Pathogenic Hotspot Flag ───────────────────────────────────
    # RING + BRCT are the primary pathogenic missense clusters
    # per ENIGMA BRCA1/2 VCEP classification criteria
    df['in_pathogenic_hotspot'] = (
        (df['in_RING_domain'] == 1) | (df['in_BRCT_domain'] == 1)
    ).astype(int)
    print(f"  [F9]  in_pathogenic_hotspot         : done "
          f"({df['in_pathogenic_hotspot'].sum():,} variants in hotspot)")

    # ── FEATURE 10: Loss-of-Function (LOF) Flag ──────────────────────────────
    # PVS1-equivalent: nonsense + frameshift + canonical splice
    lof_source_cols = [
        'is_nonsense', 'is_frameshift_protein',
        'cons_frameshift', 'cons_nonsense',
        'is_frameshift'
    ]
    available_lof = [c for c in lof_source_cols if c in df.columns]
    df['is_likely_lof'] = (df[available_lof].sum(axis=1) > 0).astype(int)
    print(f"  [F10] is_likely_lof                 : done "
          f"({df['is_likely_lof'].sum():,} likely LOF variants)")

    # ── FEATURE 11: Canonical Splice Site (from c. notation) ─────────────────
    # c.NNNN+1, c.NNNN+2, c.NNNN-1, c.NNNN-2
    def is_canonical_splice(name):
        if pd.isna(name):
            return 0
        return int(bool(re.search(r'c\.\d+[+\-][12][^0-9]', str(name))))

    df['is_canonical_splice_site'] = df['Name'].apply(is_canonical_splice)
    print(f"  [F11] is_canonical_splice_site      : done "
          f"({df['is_canonical_splice_site'].sum():,} canonical splice variants)")

    # ── FEATURE 12: Normalized Genomic Position ───────────────────────────────
    # BRCA1 on chr17 GRCh38: 43,044,295 – 43,125,483
    BRCA1_START = 43_044_295
    BRCA1_END   = 43_125_483
    BRCA1_LEN   = BRCA1_END - BRCA1_START
    df['normalized_position'] = (
        (df['position'] - BRCA1_START) / BRCA1_LEN
    ).clip(0, 1)
    print(f"  [F12] normalized_position           : done")

    # ── FEATURE 13: Exon 11 Flag ─────────────────────────────────────────────
    # Exon 11 is the largest exon — contains ~60% of coding sequence
    # GRCh38 coordinates: 43,082,434 – 43,091,032
    EXON11_START = 43_082_434
    EXON11_END   = 43_091_032
    df['in_exon11'] = (
        (df['position'] >= EXON11_START) &
        (df['position'] <= EXON11_END)
    ).astype(int)
    print(f"  [F13] in_exon11                     : done "
          f"({df['in_exon11'].sum():,} variants in exon 11)")

    # ── FEATURE 14: Review Strength ───────────────────────────────────────────
    review_map = {
        'practice guideline'                                   : 4,
        'reviewed by expert panel'                             : 3,
        'criteria provided, multiple submitters, no conflicts' : 2
    }
    df['review_strength'] = (
        df['ReviewStatus'].map(review_map).fillna(1).astype(int)
    )
    print(f"  [F14] review_strength               : done")

    # ── FEATURE 15: Number of Submitters ─────────────────────────────────────
    if 'NumberSubmitters' in df.columns:
        df['num_submitters'] = pd.to_numeric(
            df['NumberSubmitters'], errors='coerce'
        ).fillna(1).astype(int)
    else:
        df['num_submitters'] = df['review_strength'].copy()
    print(f"  [F15] num_submitters                : done")

    print(f"\n  ✓ Feature engineering complete for {dataset_name}")
    return df


# ── Apply to both datasets ────────────────────────────────────────────────────
df_eng     = engineer_features(df,     'Training Pool')
df_vus_eng = engineer_features(df_vus, 'VUS Pool')


# ── Define final feature column list ─────────────────────────────────────────
FEATURE_COLS = [
    # Variant type
    'variant_type_encoded',
    'indel_length',
    'is_frameshift',
    'allele_length_ratio',

    # Position
    'position',
    'normalized_position',
    'in_exon11',

    # Consequence (one-hot)
    'cons_deletion',
    'cons_duplication',
    'cons_frameshift',
    'cons_insertion',
    'cons_nonsense',
    'cons_snv',
    'cons_splice',
    'cons_other',

    # Protein change
    'has_protein_change',
    'is_missense',
    'is_nonsense',
    'is_frameshift_protein',
    'is_synonymous',
    'is_splice_protein',
    'aa_position',

    # Domain
    'in_RING_domain',
    'in_BRCT_domain',
    'in_coiled_coil',
    'in_disordered_region',
    'in_pathogenic_hotspot',
    'domain_known',

    # Functional flags
    'is_likely_lof',
    'is_canonical_splice_site',

    # Review metadata
    'review_strength',
    'num_submitters'
]

TARGET_COL = 'label'

# Check which features are present
available_features = [c for c in FEATURE_COLS if c in df_eng.columns]
missing_features   = [c for c in FEATURE_COLS if c not in df_eng.columns]

print(f"\n{'='*55}")
print(f"[Feature Audit]")
print(f"  Expected features  : {len(FEATURE_COLS)}")
print(f"  Available features : {len(available_features)}")
print(f"  Missing features   : {len(missing_features)}")
if missing_features:
    print(f"  Missing list       : {missing_features}")


# ── Save engineered datasets ──────────────────────────────────────────────────
df_eng.to_csv('data/processed/brca1_features.csv', index=False)
df_vus_eng.to_csv('data/processed/vus_features.csv', index=False)

# Save feature list as JSON — used by all future phases
with open('data/processed/feature_cols.json', 'w') as f:
    json.dump(available_features, f, indent=2)

print(f"\n[Saved]")
print(f"  data/processed/brca1_features.csv  ({len(df_eng):,} rows)")
print(f"  data/processed/vus_features.csv    ({len(df_vus_eng):,} rows)")
print(f"  data/processed/feature_cols.json   ({len(available_features)} features)")


# ── EDA: Feature Distributions by Class ──────────────────────────────────────
print("\n[EDA] Generating plots...")

plot_features = [
    ('variant_type_encoded',     'Variant Type Encoded'),
    ('in_pathogenic_hotspot',    'In Pathogenic Hotspot'),
    ('is_likely_lof',            'Is Likely LOF'),
    ('is_missense',              'Is Missense'),
    ('is_nonsense',              'Is Nonsense'),
    ('in_BRCT_domain',           'In BRCT Domain'),
    ('in_RING_domain',           'In RING Domain'),
    ('is_canonical_splice_site', 'Canonical Splice Site'),
    ('normalized_position',      'Normalized Position'),
]

# Filter to only features that exist
plot_features = [(f, t) for f, t in plot_features if f in df_eng.columns]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('BRCA1 Phase 2 — Feature Distributions by Class',
             fontsize=13, fontweight='bold')

for ax, (feat, title) in zip(axes.flat, plot_features):
    for label_val, label_name, color in [
        (0, 'Benign',     '#2196F3'),
        (1, 'Pathogenic', '#F44336')
    ]:
        subset = df_eng[df_eng['label'] == label_val][feat].dropna()
        ax.hist(subset, bins=20, alpha=0.6,
                label=label_name, color=color, edgecolor='white')
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('Count', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/phase2_feature_distributions.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  results/figures/phase2_feature_distributions.png  saved")

# ── Feature Correlation with Label ───────────────────────────────────────────
corr_data  = df_eng[available_features + ['label']].fillna(0)
label_corr = (
    corr_data.corr()['label']
    .drop('label')
    .abs()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 8))
label_corr.head(20).plot(kind='barh', color='#7B1FA2', edgecolor='black')
plt.xlabel('Absolute Correlation with Pathogenicity Label')
plt.title('Top 20 ClinVar Features — Correlation with Label\n'
          '(Phase 3 dbNSFP scores will significantly extend this list)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/figures/phase2_feature_correlation.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  results/figures/phase2_feature_correlation.png    saved")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — Summary")
print("=" * 60)
print(f"  Features engineered        : {len(available_features)}")
print(f"  Training variants          : {len(df_eng):,}")
print(f"  VUS variants               : {len(df_vus_eng):,}")

print(f"\n  Top 10 features by label correlation:")
for feat, corr_val in label_corr.head(10).items():
    bar = '█' * int(corr_val * 40)
    print(f"    {feat:<35} {corr_val:.4f}  {bar}")

print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  EXPECTED PERFORMANCE AT THIS STAGE                 │
  │                                                     │
  │  ClinVar features only  →  AUC ~0.88 – 0.92        │
  │                                                     │
  │  Phase 3 adds dbNSFP scores:                        │
  │    BayesDel, REVEL, CADD, SpliceAI,                 │
  │    SIFT, PolyPhen2, gnomAD AF, conservation         │
  │                                                     │
  │  With dbNSFP scores     →  AUC ~0.97+              │
  │                                                     │
  │  dbNSFP is the single most impactful feature        │
  │  addition in the entire pipeline.                   │
  └─────────────────────────────────────────────────────┘
""")
print("=" * 60)
print("→ Proceed to Phase 3: dbNSFP Download and Score Integration")