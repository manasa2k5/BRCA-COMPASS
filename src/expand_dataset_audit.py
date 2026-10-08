# ============================================================
# DATASET EXPANSION AUDIT
# File: src/expand_dataset_audit.py
#
# Analyzes every possible relaxation of the current filters
# and reports exact variant counts at each step.
# Does NOT modify any existing files.
# Read-only audit only.
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("DATASET EXPANSION AUDIT — BRCA1 ClinVar")
print("=" * 65)

# ── Load raw ClinVar ──────────────────────────────────────────────────────────
import os
RAW_PATH = 'data/raw/variant_summary.txt'
if os.path.exists(RAW_PATH):
    df_raw = pd.read_csv(RAW_PATH, sep='\t', low_memory=False)
elif os.path.exists(RAW_PATH + '.gz'):
    df_raw = pd.read_csv(RAW_PATH + '.gz', sep='\t',
                         low_memory=False, compression='gzip')

print(f"\nTotal ClinVar variants loaded : {len(df_raw):,}")

# ── Step 1: BRCA1 only ───────────────────────────────────────────────────────
df = df_raw[df_raw['GeneSymbol'] == 'BRCA1'].copy()
print(f"\nAfter BRCA1 filter            : {len(df):,}")

# ── Step 2: GRCh38 only ──────────────────────────────────────────────────────
df38 = df[df['Assembly'] == 'GRCh38'].copy()
print(f"After GRCh38 filter           : {len(df38):,}")

# ── Step 3: Allowed variant types ────────────────────────────────────────────
allowed_types = [
    'single nucleotide variant',
    'Deletion', 'Insertion', 'Duplication', 'Indel'
]
df_type = df38[df38['Type'].isin(allowed_types)].copy()
print(f"After variant type filter     : {len(df_type):,}")

# ── Current production filter ─────────────────────────────────────────────────
HIGH_CONF = [
    'practice guideline',
    'reviewed by expert panel',
    'criteria provided, multiple submitters, no conflicts'
]
df_hc = df_type[df_type['ReviewStatus'].isin(HIGH_CONF)].copy()
print(f"After HIGH-CONFIDENCE filter  : {len(df_hc):,}  ← current baseline")

# ── Label definitions ─────────────────────────────────────────────────────────
PATH_TERMS = [
    'Pathogenic', 'Likely pathogenic',
    'Pathogenic/Likely pathogenic'
]
BENIGN_TERMS = [
    'Benign', 'Likely benign', 'Benign/Likely benign'
]
VUS_TERMS    = ['Uncertain significance']
CONFLICT_STR = 'conflict'

def count_classes(df_sub, label):
    """Count P, B, VUS, conflict in a dataframe."""
    cs = df_sub['ClinicalSignificance']
    n_path    = cs.isin(PATH_TERMS).sum()
    n_benign  = cs.isin(BENIGN_TERMS).sum()
    n_vus     = cs.str.contains('Uncertain significance',
                                case=False, na=False).sum()
    n_conflict= cs.str.contains(CONFLICT_STR,
                                case=False, na=False).sum()
    n_other   = len(df_sub) - n_path - n_benign - n_vus - n_conflict
    trainable = n_path + n_benign
    print(f"\n  [{label}]")
    print(f"    Pathogenic + Likely P : {n_path:>6,}")
    print(f"    Benign + Likely B     : {n_benign:>6,}")
    print(f"    VUS                   : {n_vus:>6,}  (held out)")
    print(f"    Conflicting           : {n_conflict:>6,}  (excluded)")
    print(f"    Other/Mixed           : {n_other:>6,}")
    print(f"    ─────────────────────────────")
    print(f"    TRAINABLE (P+B)       : {trainable:>6,}")
    return trainable

print("\n" + "=" * 65)
print("CURRENT PRODUCTION BASELINE")
print("=" * 65)
current_trainable = count_classes(df_hc, "Current: HIGH-CONFIDENCE only")

# ── Review status breakdown ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FULL REVIEW STATUS BREAKDOWN (BRCA1, GRCh38, allowed types)")
print("=" * 65)
rs_counts = df_type['ReviewStatus'].value_counts()
print(f"\n  {'Review Status':<55} {'Count':>8}")
print(f"  {'─'*55} {'─'*8}")
for rs, n in rs_counts.items():
    flag = ' ← IN PRODUCTION' if rs in HIGH_CONF else ''
    print(f"  {rs:<55} {n:>8,}{flag}")

# ── RELAXATION ANALYSIS ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RELAXATION STEPS — INCREMENTAL VARIANT COUNTS")
print("=" * 65)

results = {}

# ── TIER 1: Add single-submitter WITH criteria ────────────────────────────────
TIER1 = HIGH_CONF + [
    'criteria provided, single submitter'
]
df_t1 = df_type[df_type['ReviewStatus'].isin(TIER1)].copy()
# Remove conflicting
df_t1 = df_t1[~df_t1['ClinicalSignificance'].str.contains(
    CONFLICT_STR, case=False, na=False)].copy()
t1_train = count_classes(df_t1,
    "TIER 1: + single submitter WITH criteria")
results['Tier1_SingleSubmitter'] = t1_train

# ── TIER 2: Add no_assertion_criteria_provided ───────────────────────────────
TIER2 = TIER1 + [
    'no assertion criteria provided'
]
df_t2 = df_type[df_type['ReviewStatus'].isin(TIER2)].copy()
df_t2 = df_t2[~df_t2['ClinicalSignificance'].str.contains(
    CONFLICT_STR, case=False, na=False)].copy()
t2_train = count_classes(df_t2,
    "TIER 2: + no assertion criteria provided")
results['Tier2_NoAssertionCriteria'] = t2_train

# ── TIER 3: Add conflicting interpretations as AMBIGUOUS ─────────────────────
# Not for training — just count
df_conflict = df_type[
    df_type['ClinicalSignificance'].str.contains(
        CONFLICT_STR, case=False, na=False)
].copy()
print(f"\n  [Conflicting interpretations — NEVER for training]")
print(f"    Total conflicting in BRCA1: {len(df_conflict):,}")
print(f"    These cannot be used without expert re-review")

# ── TIER 4: Phenotype filter — breast cancer specific ────────────────────────
print("\n" + "─" * 65)
print("BREAST CANCER SPECIFICITY CHECK")
print("─" * 65)

breast_terms = [
    'breast', 'BRCA', 'hereditary breast',
    'breast-ovarian', 'breast/ovarian'
]
breast_pattern = '|'.join(breast_terms)

# Check current production set phenotype coverage
if 'PhenotypeList' in df_hc.columns:
    has_breast = df_hc['PhenotypeList'].str.contains(
        breast_pattern, case=False, na=False
    )
    print(f"\n  Current production set ({len(df_hc):,} variants):")
    print(f"    With breast cancer phenotype : "
          f"{has_breast.sum():,} ({has_breast.mean()*100:.1f}%)")
    print(f"    Without breast phenotype     : "
          f"{(~has_breast).sum():,} ({(~has_breast).mean()*100:.1f}%)")
    print(f"    Note: All are BRCA1 — breast relevance is implicit")

# ── TIER 5: External databases ────────────────────────────────────────────────
print("\n" + "─" * 65)
print("EXTERNAL DATABASE ESTIMATES")
print("─" * 65)
print(f"""
  BRCA Exchange (https://brcaexchange.org):
    Curated BRCA1/2 variants with ENIGMA classifications
    Estimated BRCA1 classified variants: ~3,000–5,000
    Overlap with ClinVar: ~60–70%
    Net new variants: ~1,000–2,000

  ENIGMA direct (https://enigmaconsortium.org):
    Expert panel classifications
    Largely already included in ClinVar expert panel tier
    Net new variants: ~200–500

  LOVD (https://lovd.nl):
    Leiden Open Variation Database
    BRCA1 entries: ~8,000 (many VUS)
    High-confidence classified: ~1,500–2,500
    Net new variants: ~500–1,500

  ClinGen Variant Curation Interface:
    ENIGMA VCEP curated variants
    Largely overlaps with ClinVar expert panel
    Net new variants: ~100–300
""")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("=" * 65)
print("EXPANSION SUMMARY — WHAT IS ACHIEVABLE")
print("=" * 65)

print(f"""
  CURRENT BASELINE (production)
  ─────────────────────────────────────────────────────
  Trainable variants (P+B): {current_trainable:,}

  RELAXATION OPTIONS (ClinVar only, in order of safety)
  ─────────────────────────────────────────────────────
  Option A — Add single-submitter WITH criteria:
    Trainable: {results.get('Tier1_SingleSubmitter', 0):,}
    Net new  : {results.get('Tier1_SingleSubmitter', 0) - current_trainable:,}
    Risk     : Moderate — some label noise from single submitters
    Safety   : Accept if variant has consistent classification
               history. Still has submission criteria.

  Option B — Also add no-assertion-criteria:
    Trainable: {results.get('Tier2_NoAssertionCriteria', 0):,}
    Net new  : {results.get('Tier2_NoAssertionCriteria', 0) - current_trainable:,}
    Risk     : Higher — no standardized criteria used
    Safety   : Acceptable ONLY with ClinSig consistency check
               (same classification from multiple sources)

  Option C — External databases (BRCA Exchange + LOVD):
    Estimated net new: 1,500–3,500
    Risk     : Low for BRCA Exchange (ENIGMA curated)
               Moderate for LOVD (variable quality)
    Safety   : Merge on chr:pos:ref:alt, keep only variants
               with concordant ClinVar + external classification

  REALISTIC PATH TO 10,000 VARIANTS
  ─────────────────────────────────────────────────────
""")

target   = 10000
current  = current_trainable
t1_net   = results.get('Tier1_SingleSubmitter', 0)
t2_net   = results.get('Tier2_NoAssertionCriteria', 0)
ext_est  = 2000  # conservative BRCA Exchange estimate

print(f"  Current production             : {current:>7,}")
print(f"  + Single submitter w/ criteria : {t1_net:>7,}  "
      f"(+{t1_net - current:,})")
print(f"  + No assertion criteria        : {t2_net:>7,}  "
      f"(+{t2_net - t1_net:,})")
print(f"  + BRCA Exchange (est.)         : {t2_net + ext_est:>7,}  "
      f"(+{ext_est:,} estimated)")
print(f"\n  Target                         : {target:>7,}")

gap = target - (t2_net + ext_est)
if gap <= 0:
    print(f"\n  ✓ 10,000 IS ACHIEVABLE from ClinVar + BRCA Exchange")
    print(f"    Recommended strategy: Option A + BRCA Exchange")
elif gap <= 1000:
    print(f"\n  ⚠ 10,000 IS BORDERLINE — gap of ~{gap:,}")
    print(f"    Need Option B (no-criteria) + BRCA Exchange")
else:
    print(f"\n  ✗ 10,000 NOT ACHIEVABLE from ClinVar alone")
    print(f"    Gap of {gap:,} variants requires LOVD or literature mining")

print(f"""
  RECOMMENDATION
  ─────────────────────────────────────────────────────
  Step 1 (safest): Add 'criteria provided, single submitter'
    Apply consistency filter: keep only variants where
    ClinicalSignificance is uniform (no mixed P+B from
    different submitters for same variant).

  Step 2 (if needed): Download BRCA Exchange JSON
    URL: https://brcaexchange.org/backend/downloads/
    Join on chr17:pos:ref:alt (GRCh38)
    Keep only ENIGMA-classified variants (highest quality)

  Step 3 (last resort): Add no-assertion-criteria
    Only after applying strict deduplication and
    ClinSig consistency check across all submitters.

  Label noise mitigation for all new variants:
    - Require ClinSigSimple in {{0, 1}} only (not -1)
    - Reject any variant with >1 unique classification
      across all submitters
    - Reject variants with LastEvaluated before 2015
      (pre-ACMG 2015 guidelines)
  ─────────────────────────────────────────────────────
""")
print("→ Paste output — exact numbers determine the recommendation")