# ============================================================
# PHASE 3 FINAL FIX + AutoGluon Feature Finalization
# File: src/fix_and_finalize_phase3.py
#
# Root cause confirmed:
#   bayesdel.add_af → dict → need .score
#   bayesdel.no_af  → dict → need .score
#   phylop.17way_primate → dict → need .score
#   phastcons.17way_primate → dict → need .score
#   gerp++.rs → confirmed direct float (was missing from fields)
#
# This script:
#   1. Re-fetches with corrected nested extraction
#   2. Saves final brca1_features_scored.csv
#   3. Updates feature_cols.json
#   4. Prepares train/test/vus splits for AutoGluon
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 3 FINAL FIX: Corrected Score Extraction")
print("=" * 60)

import myvariant
mv = myvariant.MyVariantInfo()

# ── Load original engineered files (pre-score) ────────────────────────────────
df_train = pd.read_csv('data/processed/brca1_features.csv', low_memory=False)
df_vus   = pd.read_csv('data/processed/vus_features.csv',   low_memory=False)

print(f"\nLoaded original engineered files:")
print(f"  Training : {len(df_train):,} rows")
print(f"  VUS      : {len(df_vus):,} rows")

# ── Build HGVS IDs ────────────────────────────────────────────────────────────
def build_hgvs_id(row):
    try:
        chrom = str(row['Chromosome']).replace('chr', '').strip()
        pos   = int(row['PositionVCF']) if pd.notna(row.get('PositionVCF')) else int(row['Start'])
        ref   = str(row['ReferenceAlleleVCF']).strip() if pd.notna(row.get('ReferenceAlleleVCF')) else str(row['ReferenceAllele']).strip()
        alt   = str(row['AlternateAlleleVCF']).strip() if pd.notna(row.get('AlternateAlleleVCF')) else str(row['AlternateAllele']).strip()
        if ref in ['-', '', 'nan', 'None'] or alt in ['-', '', 'nan', 'None']:
            return None
        if len(ref) == 1 and len(alt) == 1:
            return f"chr{chrom}:g.{pos}{ref}>{alt}"
        elif len(ref) > len(alt):
            del_start = pos + 1
            del_end   = pos + len(ref) - 1
            return f"chr{chrom}:g.{del_start}del" if del_start == del_end \
                   else f"chr{chrom}:g.{del_start}_{del_end}del"
        elif len(alt) > len(ref):
            ins_seq = alt[len(ref):]
            ins_pos = pos + len(ref) - 1
            return f"chr{chrom}:g.{ins_pos}_{ins_pos+1}ins{ins_seq}"
        else:
            return f"chr{chrom}:g.{pos}{ref}>{alt}"
    except:
        return None

df_train['hgvs_id'] = df_train.apply(build_hgvs_id, axis=1)
df_vus['hgvs_id']   = df_vus.apply(build_hgvs_id, axis=1)

# ============================================================
# CORRECTED EXTRACTION — all nested .score levels fixed
# ============================================================

def safe_num(val):
    """Convert to float, average if list."""
    if val is None:
        return np.nan
    if isinstance(val, list):
        try:
            nums = [float(v) for v in val if v is not None]
            return float(np.mean(nums)) if nums else np.nan
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan


def get_nested(obj, *keys):
    """
    Navigate nested dict safely.
    Handles lists by taking first element.
    Final value passed through safe_num.
    """
    for key in keys:
        if obj is None:
            return np.nan
        if isinstance(obj, list):
            obj = obj[0] if len(obj) > 0 else None
            if obj is None:
                return np.nan
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return np.nan
    return safe_num(obj)


def extract_scores_fixed(r):
    """
    Fully corrected extraction.
    All field paths confirmed from STEP 2 raw output.
    """
    empty_row = {k: np.nan for k in [
        'BayesDel_addAF', 'BayesDel_noAF',
        'REVEL_score', 'CADD_phred',
        'SIFT_score', 'Polyphen2_HDIV', 'Polyphen2_HVAR',
        'SpliceAI_DS_max',
        'phyloP17way', 'phastCons17way', 'GERP_RS',
        'AlphaMissense_score', 'DANN_score',
        'VEST4_score', 'MetaRNN_score',
        'ClinPred_score', 'MPC_score', 'gnomAD_AF'
    ]}

    if not isinstance(r, dict) or r.get('notfound'):
        empty_row['hgvs_id'] = r.get('query', '') if isinstance(r, dict) else ''
        return empty_row

    qid = r.get('query', r.get('_id', ''))
    d   = r.get('dbnsfp') or {}
    gex = r.get('gnomad_exome')  or {}
    ggn = r.get('gnomad_genome') or {}

    result = {'hgvs_id': qid}

    # ── BayesDel ──────────────────────────────────────────────────────────────
    # Confirmed structure: dbnsfp.bayesdel.add_af.score
    #                      dbnsfp.bayesdel.no_af.score
    bd = d.get('bayesdel') or {}
    result['BayesDel_addAF'] = get_nested(bd, 'add_af', 'score')
    result['BayesDel_noAF']  = get_nested(bd, 'no_af',  'score')

    # ── REVEL ─────────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.revel.score (list — take mean)
    rv = d.get('revel') or {}
    result['REVEL_score'] = get_nested(rv, 'score')

    # ── CADD ─────────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.cadd.phred (direct float)
    ca = d.get('cadd') or {}
    result['CADD_phred'] = get_nested(ca, 'phred')

    # ── SIFT ─────────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.sift.score
    sf = d.get('sift') or {}
    result['SIFT_score'] = get_nested(sf, 'score')

    # ── PolyPhen2 ─────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.polyphen2.hdiv.score
    pp = d.get('polyphen2') or {}
    result['Polyphen2_HDIV'] = get_nested(pp, 'hdiv', 'score')
    result['Polyphen2_HVAR'] = get_nested(pp, 'hvar', 'score')

    # ── SpliceAI ─────────────────────────────────────────────────────────────
    # Not in dbnsfp — lives at top-level r.spliceai
    sp = r.get('spliceai') or {}
    if isinstance(sp, dict):
        ds_vals = [
            safe_num(sp.get(k))
            for k in ['ds_ag', 'ds_al', 'ds_dg', 'ds_dl']
        ]
        ds_vals = [v for v in ds_vals if not np.isnan(v)]
        result['SpliceAI_DS_max'] = max(ds_vals) if ds_vals else np.nan
    else:
        result['SpliceAI_DS_max'] = np.nan

    # ── PhyloP ───────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.phylop.17way_primate.score  ← EXTRA .score level!
    phy = d.get('phylop') or {}
    result['phyloP17way'] = get_nested(phy, '17way_primate', 'score')
    # Fallback to 100way if 17way missing
    if np.isnan(result['phyloP17way']):
        result['phyloP17way'] = get_nested(phy, '100way_vertebrate', 'score')

    # ── PhastCons ────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.phastcons.17way_primate.score ← EXTRA .score level!
    phc = d.get('phastcons') or {}
    result['phastCons17way'] = get_nested(phc, '17way_primate', 'score')
    if np.isnan(result['phastCons17way']):
        result['phastCons17way'] = get_nested(phc, '100way_vertebrate', 'score')

    # ── GERP ─────────────────────────────────────────────────────────────────
    # Confirmed: dbnsfp['gerp++'].rs (direct float — was missing from fields)
    gp = d.get('gerp++') or {}
    result['GERP_RS'] = get_nested(gp, 'rs')
    # Also try gerp (without ++) for different dbNSFP versions
    if np.isnan(result['GERP_RS']):
        gp2 = d.get('gerp') or {}
        result['GERP_RS'] = get_nested(gp2, '91_mammals')

    # ── AlphaMissense ────────────────────────────────────────────────────────
    # Confirmed: dbnsfp.alphamissense.score (list — take mean)
    am = d.get('alphamissense') or {}
    result['AlphaMissense_score'] = get_nested(am, 'score')

    # ── DANN ─────────────────────────────────────────────────────────────────
    dn = d.get('dann') or {}
    result['DANN_score'] = get_nested(dn, 'score')

    # ── VEST4 ─────────────────────────────────────────────────────────────────
    vt = d.get('vest4') or {}
    result['VEST4_score'] = get_nested(vt, 'score')

    # ── MetaRNN ───────────────────────────────────────────────────────────────
    mr = d.get('metarnn') or {}
    result['MetaRNN_score'] = get_nested(mr, 'score')

    # ── ClinPred ─────────────────────────────────────────────────────────────
    cp = d.get('clinpred') or {}
    result['ClinPred_score'] = get_nested(cp, 'score')

    # ── MPC ───────────────────────────────────────────────────────────────────
    mpc = d.get('mpc') or {}
    result['MPC_score'] = get_nested(mpc, 'score')

    # ── gnomAD AF ─────────────────────────────────────────────────────────────
    gnomad = np.nan
    if isinstance(gex, dict):
        af_obj = gex.get('af') or {}
        gnomad = get_nested(af_obj, 'af') if isinstance(af_obj, dict) \
                 else safe_num(af_obj)
    if np.isnan(gnomad) and isinstance(ggn, dict):
        af_obj = ggn.get('af') or {}
        gnomad = get_nested(af_obj, 'af') if isinstance(af_obj, dict) \
                 else safe_num(af_obj)
    result['gnomAD_AF'] = gnomad

    return result


# ============================================================
# FETCH FUNCTION
# ============================================================

def fetch_final(df_input, label, batch_size=200):
    valid_df = df_input[df_input['hgvs_id'].notna()].copy()
    all_ids  = valid_df['hgvs_id'].tolist()
    total_b  = (len(all_ids) + batch_size - 1) // batch_size

    print(f"\nFetching {len(all_ids):,} — {label}...")

    FIELDS = ','.join([
        'dbnsfp.bayesdel',
        'dbnsfp.revel',
        'dbnsfp.cadd',
        'dbnsfp.sift',
        'dbnsfp.polyphen2',
        'dbnsfp.phylop',
        'dbnsfp.phastcons',
        'dbnsfp.gerp++',
        'dbnsfp.alphamissense',
        'dbnsfp.dann',
        'dbnsfp.vest4',
        'dbnsfp.metarnn',
        'dbnsfp.clinpred',
        'dbnsfp.mpc',
        'gnomad_exome.af',
        'gnomad_genome.af',
        'spliceai'
    ])

    all_raw = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        bn    = i // batch_size + 1
        print(f"  Batch {bn}/{total_b}...", end='\r')
        try:
            res = mv.getvariants(batch, fields=FIELDS, assembly='hg38')
            all_raw.extend(res)
        except Exception as e:
            print(f"\n  Batch {bn} error: {e}")
            for vid in batch:
                all_raw.append({'query': vid, 'notfound': True})

    print(f"\n  Done: {len(all_raw):,} results")

    # Parse with corrected extractor
    parsed    = [extract_scores_fixed(r) for r in all_raw]
    scores_df = pd.DataFrame(parsed)
    scores_df = scores_df.drop_duplicates(subset='hgvs_id', keep='first')

    # Coverage report
    score_cols = [
        'BayesDel_addAF', 'BayesDel_noAF', 'REVEL_score',
        'CADD_phred', 'SIFT_score', 'Polyphen2_HDIV',
        'SpliceAI_DS_max', 'phyloP17way', 'phastCons17way',
        'GERP_RS', 'AlphaMissense_score', 'DANN_score',
        'VEST4_score', 'MetaRNN_score', 'ClinPred_score',
        'MPC_score', 'gnomAD_AF'
    ]

    print(f"\n  {'Score':<25} {'Coverage':>14}  Bar")
    print(f"  {'─'*25} {'─'*14}  {'─'*20}")
    for col in score_cols:
        if col in scores_df.columns:
            n   = scores_df[col].notna().sum()
            pct = n / len(scores_df) * 100
            bar = '█' * int(pct / 5)
            status = '✓' if n > 0 else '✗'
            print(f"  {status} {col:<24} {n:>5}/{len(scores_df):>5} ({pct:5.1f}%)  {bar}")

    # Merge back — one-to-one
    merged = df_input.merge(scores_df, on='hgvs_id', how='left')
    if len(merged) != len(df_input):
        merged = merged.drop_duplicates(subset='VariationID', keep='first')
    assert len(merged) == len(df_input), \
        f"FATAL row mismatch: {len(merged)} != {len(df_input)}"
    print(f"  ✓ Row count verified: {len(merged):,}")

    return merged


# ── Execute ───────────────────────────────────────────────────────────────────
df_train_final = fetch_final(df_train, 'Training Pool')
df_vus_final   = fetch_final(df_vus,   'VUS Pool')

# ── Verify BayesDel is now populated ─────────────────────────────────────────
print("\n" + "─" * 55)
print("VERIFICATION: Sample BayesDel values")
print("─" * 55)
sample = df_train_final[df_train_final['BayesDel_addAF'].notna()].head(5)
if len(sample) > 0:
    print(sample[['Name', 'BayesDel_addAF', 'BayesDel_noAF',
                  'REVEL_score', 'phyloP17way', 'phastCons17way']].to_string())
else:
    print("⚠ BayesDel still empty — see diagnosis below")

# ── Save final scored files ───────────────────────────────────────────────────
df_train_final.to_csv('data/processed/brca1_features_scored.csv', index=False)
df_vus_final.to_csv('data/processed/vus_features_scored.csv',     index=False)

print(f"\n[Saved]")
print(f"  brca1_features_scored.csv : {len(df_train_final):,} rows, "
      f"{len(df_train_final.columns)} columns")
print(f"  vus_features_scored.csv   : {len(df_vus_final):,} rows")

# ── Update feature_cols.json ──────────────────────────────────────────────────
print("\n" + "─" * 55)
print("Updating feature_cols.json")
print("─" * 55)

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

ALL_EXTERNAL = [
    'BayesDel_addAF', 'BayesDel_noAF',
    'REVEL_score', 'CADD_phred',
    'SIFT_score', 'Polyphen2_HDIV', 'Polyphen2_HVAR',
    'SpliceAI_DS_max', 'phyloP17way', 'phastCons17way',
    'GERP_RS', 'AlphaMissense_score', 'DANN_score',
    'VEST4_score', 'MetaRNN_score', 'ClinPred_score',
    'MPC_score', 'gnomAD_AF'
]

final_features = BASE_CLINVAR_FEATURES.copy()
added          = []
zero_cols      = []

for col in ALL_EXTERNAL:
    if col in df_train_final.columns:
        n = df_train_final[col].notna().sum()
        if n > 0:
            final_features.append(col)
            added.append(col)
        else:
            zero_cols.append(col)

with open('data/processed/feature_cols.json', 'w') as f:
    json.dump(final_features, f, indent=2)

print(f"\n  ✓ Added to model     : {len(added)} external scores")
print(f"  Added               : {added}")
print(f"  Zero coverage (skip): {zero_cols}")
print(f"  Total features      : {len(final_features)}")

# ── Prepare AutoGluon-ready CSVs ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("Preparing AutoGluon-ready datasets")
print("─" * 55)

from sklearn.model_selection import train_test_split

available_features = [c for c in final_features if c in df_train_final.columns]
missing            = [c for c in final_features if c not in df_train_final.columns]
if missing:
    print(f"  ⚠ Missing from dataframe: {missing}")

# Select feature + label columns
df_model = df_train_final[available_features + ['label']].copy()
df_model['label'] = df_model['label'].astype(int)

print(f"  Model-ready rows     : {len(df_model):,}")
print(f"  Model-ready features : {len(available_features)}")
print(f"  Label distribution   : {df_model['label'].value_counts().to_dict()}")

# Stratified 80/20 split
train_df, test_df = train_test_split(
    df_model,
    test_size=0.20,
    random_state=42,
    stratify=df_model['label']
)

# VUS — features only, no label column
df_vus_model = df_vus_final[
    [c for c in available_features if c in df_vus_final.columns]
].copy()

# Save AutoGluon-ready CSVs
train_df.to_csv('data/processed/train_set.csv',    index=False)
test_df.to_csv('data/processed/test_set.csv',      index=False)
df_vus_model.to_csv('data/processed/vus_set_model.csv', index=False)

print(f"\n  data/processed/train_set.csv     : {len(train_df):,} rows")
print(f"  data/processed/test_set.csv      : {len(test_df):,} rows")
print(f"  data/processed/vus_set_model.csv : {len(df_vus_model):,} rows")

# ── Final decision gate ───────────────────────────────────────────────────────
has_bayesdel  = 'BayesDel_addAF' in added or 'BayesDel_noAF' in added
has_revel     = 'REVEL_score' in added
has_cadd      = 'CADD_phred' in added
has_alpha     = 'AlphaMissense_score' in added
has_phylop    = 'phyloP17way' in added
n_ext         = len(added)

print("\n" + "=" * 60)
print("PHASE 3 FINAL — DECISION GATE")
print("=" * 60)
print(f"  BayesDel      : {'✓' if has_bayesdel else '✗'}")
print(f"  REVEL         : {'✓' if has_revel    else '✗'}")
print(f"  CADD          : {'✓' if has_cadd     else '✗'}")
print(f"  AlphaMissense : {'✓' if has_alpha    else '✗'}")
print(f"  PhyloP        : {'✓' if has_phylop   else '✗'}")
print(f"  Total ext     : {n_ext}")
print(f"  Total features: {len(final_features)}")

if has_bayesdel and has_revel and has_cadd:
    print(f"\n  ✓ ALL KEY SCORES PRESENT")
    print(f"  Expected AUC with AutoGluon : ~0.97–0.99")
    print(f"  → PROCEED TO PHASE 4 (AutoGluon)")
elif has_revel and has_cadd and n_ext >= 6:
    print(f"\n  ⚠ BayesDel missing but {n_ext} other scores present")
    print(f"  Expected AUC with AutoGluon : ~0.95–0.97")
    print(f"  → ACCEPTABLE — PROCEED TO PHASE 4 (AutoGluon)")
else:
    print(f"\n  ✗ Insufficient external scores")
    print(f"  DO NOT PROCEED — paste output for diagnosis")

print("=" * 60)
print("\n→ Install AutoGluon:")
print("  pip install autogluon.tabular")
print("→ Then run: python src/phase4_autogluon.py")