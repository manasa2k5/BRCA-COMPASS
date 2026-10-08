# ============================================================
# FETCH dbNSFP SCORES FOR SGE VARIANTS
# File: src/fetch_sge_scores.py
#
# Fetches REVEL, BayesDel, AlphaMissense for the 1,917
# missense SGE variants — the missing scores that caused
# missense AUC = 0.50 in the first SGE run.
# ============================================================

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, matthews_corrcoef
import myvariant

print("=" * 65)
print("FETCHING dbNSFP SCORES FOR SGE VARIANTS")
print("=" * 65)

mv = myvariant.MyVariantInfo()

# ── Load SGE data ─────────────────────────────────────────────────────────────
df_raw = pd.read_excel(
    'data/external/41586_2018_461_MOESM3_ESM.xlsx',
    sheet_name='Sheet1', header=None, skiprows=2
)

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

df['position_hg19'] = pd.to_numeric(df['position_hg19'], errors='coerce')
df['sge_label']     = (df['func_class'] == 'LOF').astype(int)
HG38_OFFSET = 1_847_983
df['position_hg38'] = df['position_hg19'] + HG38_OFFSET
df = df.dropna(subset=['position_hg19']).copy()

print(f"  SGE variants: {len(df):,}")
print(f"  Missense    : {(df['consequence']=='Missense').sum():,}")

# ── Build hg38 HGVS IDs ──────────────────────────────────────────────────────
def build_hgvs(row):
    try:
        pos = int(row['position_hg38'])
        ref = str(row['reference']).strip().upper()
        alt = str(row['alt']).strip().upper()
        if len(ref)==1 and len(alt)==1 and ref != alt:
            return f"chr17:g.{pos}{ref}>{alt}"
    except:
        pass
    return None

df['hgvs_id'] = df.apply(build_hgvs, axis=1)
valid = df[df['hgvs_id'].notna()].copy()
print(f"  Valid HGVS IDs: {len(valid):,}")

# ── Fetch in batches ──────────────────────────────────────────────────────────
FIELDS = ','.join([
    'dbnsfp.bayesdel',
    'dbnsfp.revel',
    'dbnsfp.alphamissense',
    'dbnsfp.dann',
    'dbnsfp.metarnn',
    'dbnsfp.vest4',
    'dbnsfp.clinpred',
])

all_ids  = valid['hgvs_id'].tolist()
all_raw  = []
batch_sz = 200
total_b  = (len(all_ids) + batch_sz - 1) // batch_sz

print(f"\n  Fetching {len(all_ids):,} variants in {total_b} batches...")

for i in range(0, len(all_ids), batch_sz):
    batch = all_ids[i:i+batch_sz]
    bn    = i//batch_sz + 1
    print(f"  Batch {bn}/{total_b}...", end='\r')
    try:
        res = mv.getvariants(batch, fields=FIELDS, assembly='hg38')
        all_raw.extend(res)
    except Exception as e:
        print(f"\n  Batch {bn} error: {e}")
        for vid in batch:
            all_raw.append({'query': vid, 'notfound': True})

print(f"\n  Done: {len(all_raw):,} results")

# ── Parse ─────────────────────────────────────────────────────────────────────
def safe_num(val):
    if val is None: return np.nan
    if isinstance(val, list):
        try:
            nums = [float(v) for v in val if v is not None]
            return float(np.mean(nums)) if nums else np.nan
        except: return np.nan
    try: return float(val)
    except: return np.nan

def get_nested(obj, *keys):
    for key in keys:
        if obj is None: return np.nan
        if isinstance(obj, list):
            obj = obj[0] if obj else None
            if obj is None: return np.nan
        if isinstance(obj, dict): obj = obj.get(key)
        else: return np.nan
    return safe_num(obj)

rows = []
for r in all_raw:
    if not isinstance(r, dict) or r.get('notfound'):
        rows.append({'hgvs_id': r.get('query','') if isinstance(r,dict) else ''})
        continue
    d  = r.get('dbnsfp') or {}
    bd = d.get('bayesdel') or {}
    rv = d.get('revel')    or {}
    am = d.get('alphamissense') or {}
    dn = d.get('dann')     or {}
    mr = d.get('metarnn')  or {}
    vt = d.get('vest4')    or {}
    cp = d.get('clinpred') or {}
    rows.append({
        'hgvs_id'            : r.get('query', r.get('_id','')),
        'BayesDel_addAF'     : get_nested(bd, 'add_af', 'score'),
        'BayesDel_noAF'      : get_nested(bd, 'no_af',  'score'),
        'REVEL_score'        : get_nested(rv, 'score'),
        'AlphaMissense_score': get_nested(am, 'score'),
        'DANN_score'         : get_nested(dn, 'score'),
        'MetaRNN_score'      : get_nested(mr, 'score'),
        'VEST4_score'        : get_nested(vt, 'score'),
        'ClinPred_score'     : get_nested(cp, 'score'),
    })

scores_df = pd.DataFrame(rows).drop_duplicates('hgvs_id', keep='first')

# Coverage report
score_cols = ['BayesDel_addAF','REVEL_score','AlphaMissense_score',
              'DANN_score','MetaRNN_score']
print(f"\n  Score coverage on SGE variants:")
for col in score_cols:
    if col in scores_df.columns:
        n   = scores_df[col].notna().sum()
        pct = n/len(scores_df)*100
        print(f"    {col:<25} {n:>5}/{len(scores_df):>5} ({pct:5.1f}%)")

# Merge back
df_scored = valid.merge(scores_df, on='hgvs_id', how='left')

# Save
df_scored.to_csv('data/external/sge_with_scores.csv', index=False)
print(f"\n  data/external/sge_with_scores.csv saved ({len(df_scored):,} rows)")

# ── Re-evaluate with scores ───────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("RE-EVALUATING WITH dbNSFP SCORES")
print(f"{'─'*65}")

xgb_base = joblib.load('models/xgb_base.pkl')
imputer  = joblib.load('models/imputer.pkl')

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

# Rebuild full feature matrix with new scores
# Import the feature engineering logic from external_validation
exec(open('src/external_validation.py').read().split(
    '# ── Align and impute')[0].split(
    'print(f"\\n  Engineering features...")')[-1])

# Simpler: just update the score columns in the already-engineered matrix
# Load the SGE feature matrix from the previous run
# Re-run external_validation with scores injected

print("\n  Running updated evaluation...")
print("  → Run python src/external_validation_v2.py")
print("    (see instructions below)")

# Save score lookup for v2
scores_lookup = df_scored[['hgvs_id'] + [c for c in score_cols
                            if c in df_scored.columns]].copy()
scores_lookup.to_csv('data/external/sge_scores_lookup.csv', index=False)
print(f"  data/external/sge_scores_lookup.csv saved")

print("\n" + "=" * 65)
print("FETCH COMPLETE")
print("=" * 65)