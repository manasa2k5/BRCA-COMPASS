# ============================================================
# PHASE 3 — STEP 1 (FINAL): dbNSFP Score Fetching
# File: src/download_dbnsfp.py
# 
# Field paths confirmed from raw API inspection:
#   bayesdel       → dbnsfp.bayesdel.add_af / no_af
#   cadd           → dbnsfp.cadd.phred
#   revel          → dbnsfp.revel.score
#   sift           → dbnsfp.sift.score
#   polyphen2      → dbnsfp.polyphen2.hdiv.score
#   phylop         → dbnsfp.phylop.17way_primate
#   phastcons      → dbnsfp.phastcons.17way_primate
#   gerp           → dbnsfp['gerp++'].rs
#   alphamissense  → dbnsfp.alphamissense.score  (bonus — better than SIFT)
#   gnomad         → gnomad_exome.af.af
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHASE 3 STEP 1 (FINAL): dbNSFP Score Fetching")
print("=" * 60)

# ── Load ORIGINAL engineered files ────────────────────────────────────────────
df_train = pd.read_csv('data/processed/brca1_features.csv', low_memory=False)
df_vus   = pd.read_csv('data/processed/vus_features.csv',   low_memory=False)

print(f"\nLoaded:")
print(f"  Training : {len(df_train):,} variants")
print(f"  VUS      : {len(df_vus):,} variants")

import myvariant
mv = myvariant.MyVariantInfo()

# ============================================================
# BUILD HGVS IDs
# ============================================================

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
            if del_start == del_end:
                return f"chr{chrom}:g.{del_start}del"
            else:
                return f"chr{chrom}:g.{del_start}_{del_end}del"
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
# SCORE EXTRACTION — confirmed field paths from raw inspection
# ============================================================

def safe_num(val):
    """Convert any value to float, handling lists by averaging."""
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


def extract_scores(r):
    """
    Extract all scores from one MyVariant.info result dict.
    All field paths confirmed from live API response inspection.
    """
    if not isinstance(r, dict) or r.get('notfound'):
        return {
            'hgvs_id'            : r.get('query', '') if isinstance(r, dict) else '',
            'BayesDel_addAF'     : np.nan,
            'BayesDel_noAF'      : np.nan,
            'REVEL_score'        : np.nan,
            'CADD_phred'         : np.nan,
            'SIFT_score'         : np.nan,
            'Polyphen2_HDIV'     : np.nan,
            'Polyphen2_HVAR'     : np.nan,
            'SpliceAI_DS_max'    : np.nan,
            'phyloP17way'        : np.nan,
            'phastCons17way'     : np.nan,
            'GERP_RS'            : np.nan,
            'AlphaMissense_score': np.nan,
            'DANN_score'         : np.nan,
            'gnomAD_AF'          : np.nan,
        }

    d   = r.get('dbnsfp')   or {}
    gex = r.get('gnomad_exome')  or {}
    ggn = r.get('gnomad_genome') or {}

    # ── BayesDel — confirmed: dbnsfp.bayesdel.add_af / no_af ─────────────────
    bd     = d.get('bayesdel') or {}
    bd_add = safe_num(bd.get('add_af')) if isinstance(bd, dict) else np.nan
    bd_no  = safe_num(bd.get('no_af'))  if isinstance(bd, dict) else np.nan

    # ── REVEL — confirmed: dbnsfp.revel.score ────────────────────────────────
    rv      = d.get('revel') or {}
    revel   = safe_num(rv.get('score')) if isinstance(rv, dict) else np.nan

    # ── CADD — confirmed: dbnsfp.cadd.phred ──────────────────────────────────
    cd      = d.get('cadd') or {}
    cadd    = safe_num(cd.get('phred')) if isinstance(cd, dict) else np.nan

    # ── SIFT — confirmed: dbnsfp.sift.score ──────────────────────────────────
    sf      = d.get('sift') or {}
    sift    = safe_num(sf.get('score')) if isinstance(sf, dict) else np.nan

    # ── PolyPhen2 — confirmed: dbnsfp.polyphen2.hdiv.score ───────────────────
    pp2     = d.get('polyphen2') or {}
    hdiv    = pp2.get('hdiv') or {} if isinstance(pp2, dict) else {}
    hvar    = pp2.get('hvar') or {} if isinstance(pp2, dict) else {}
    pp2_h   = safe_num(hdiv.get('score')) if isinstance(hdiv, dict) else np.nan
    pp2_v   = safe_num(hvar.get('score')) if isinstance(hvar, dict) else np.nan

    # ── SpliceAI — not in dbnsfp, check top-level spliceai ───────────────────
    # Only available for variants near splice sites
    sp      = r.get('spliceai') or {}
    if isinstance(sp, dict):
        ds_vals = []
        for k in ['ds_ag', 'ds_al', 'ds_dg', 'ds_dl']:
            v = safe_num(sp.get(k))
            if not np.isnan(v):
                ds_vals.append(v)
        spliceai = max(ds_vals) if ds_vals else np.nan
    else:
        spliceai = np.nan

    # Also check dbnsfp for spliceai
    if np.isnan(spliceai):
        sp2 = d.get('spliceai') or {}
        if isinstance(sp2, dict):
            ds_vals = []
            for k in ['ds_ag', 'ds_al', 'ds_dg', 'ds_dl', 'ds_max']:
                v = safe_num(sp2.get(k))
                if not np.isnan(v):
                    ds_vals.append(v)
            spliceai = max(ds_vals) if ds_vals else np.nan

    # ── PhyloP — confirmed: dbnsfp.phylop.17way_primate ──────────────────────
    phy     = d.get('phylop') or {}
    phylop  = safe_num(phy.get('17way_primate')) if isinstance(phy, dict) else np.nan

    # ── PhastCons — confirmed: dbnsfp.phastcons.17way_primate ────────────────
    phs     = d.get('phastcons') or {}
    phastc  = safe_num(phs.get('17way_primate')) if isinstance(phs, dict) else np.nan

    # ── GERP — confirmed: dbnsfp['gerp++'].rs ────────────────────────────────
    gerp_d  = d.get('gerp++') or {}
    gerp    = safe_num(gerp_d.get('rs')) if isinstance(gerp_d, dict) else np.nan

    # ── AlphaMissense — confirmed: dbnsfp.alphamissense.score ────────────────
    am      = d.get('alphamissense') or {}
    alpha   = safe_num(am.get('score')) if isinstance(am, dict) else np.nan

    # ── DANN — confirmed: dbnsfp.dann.score ──────────────────────────────────
    dn      = d.get('dann') or {}
    dann    = safe_num(dn.get('score')) if isinstance(dn, dict) else np.nan

    # ── gnomAD AF — confirmed: gnomad_exome.af.af ────────────────────────────
    gnomad  = np.nan
    if isinstance(gex, dict):
        af_obj = gex.get('af') or {}
        gnomad = safe_num(af_obj.get('af')) if isinstance(af_obj, dict) else safe_num(af_obj)
    if np.isnan(gnomad) and isinstance(ggn, dict):
        af_obj = ggn.get('af') or {}
        gnomad = safe_num(af_obj.get('af')) if isinstance(af_obj, dict) else safe_num(af_obj)

    return {
        'hgvs_id'            : r.get('query', r.get('_id', '')),
        'BayesDel_addAF'     : bd_add,
        'BayesDel_noAF'      : bd_no,
        'REVEL_score'        : revel,
        'CADD_phred'         : cadd,
        'SIFT_score'         : sift,
        'Polyphen2_HDIV'     : pp2_h,
        'Polyphen2_HVAR'     : pp2_v,
        'SpliceAI_DS_max'    : spliceai,
        'phyloP17way'        : phylop,
        'phastCons17way'     : phastc,
        'GERP_RS'            : gerp,
        'AlphaMissense_score': alpha,
        'DANN_score'         : dann,
        'gnomAD_AF'          : gnomad,
    }


# ============================================================
# FETCH AND PARSE
# ============================================================

def fetch_and_score(df_input, label, batch_size=200):
    valid_df  = df_input[df_input['hgvs_id'].notna()].copy()
    all_ids   = valid_df['hgvs_id'].tolist()
    total_bat = (len(all_ids) + batch_size - 1) // batch_size

    print(f"\nFetching {len(all_ids):,} variants — {label}...")

    all_raw = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        bn    = i // batch_size + 1
        print(f"  Batch {bn}/{total_bat}...", end='\r')
        try:
            res = mv.getvariants(
                batch,
                fields='dbnsfp,gnomad_exome,gnomad_genome,spliceai',
                assembly='hg38'
            )
            all_raw.extend(res)
        except Exception as e:
            print(f"\n  Batch {bn} error: {e}")
            for vid in batch:
                all_raw.append({'query': vid, 'notfound': True})

    print(f"\n  Completed: {len(all_raw):,} results")

    # Parse
    parsed = [extract_scores(r) for r in all_raw]
    scores_df = pd.DataFrame(parsed)

    # Deduplicate scores before merge
    scores_df = scores_df.drop_duplicates(subset='hgvs_id', keep='first')

    # Coverage report
    score_cols = [
        'BayesDel_addAF', 'BayesDel_noAF', 'REVEL_score',
        'CADD_phred', 'SIFT_score', 'Polyphen2_HDIV',
        'SpliceAI_DS_max', 'phyloP17way', 'GERP_RS',
        'AlphaMissense_score', 'gnomAD_AF'
    ]
    print(f"\n  Score coverage for {label}:")
    for col in score_cols:
        if col in scores_df.columns:
            n   = scores_df[col].notna().sum()
            pct = n / len(scores_df) * 100
            bar = '█' * int(pct / 5)
            print(f"    {col:<25} {n:>5}/{len(scores_df):>5} ({pct:5.1f}%)  {bar}")

    # Merge — left join, one-to-one
    merged = df_input.merge(
        scores_df, on='hgvs_id', how='left'
    )

    # Deduplicate if any explosion occurred
    if len(merged) != len(df_input):
        print(f"\n  ⚠ Row mismatch ({len(merged)} vs {len(df_input)}) — fixing...")
        merged = merged.drop_duplicates(subset='VariationID', keep='first')

    assert len(merged) == len(df_input), \
        f"FATAL: {len(merged)} != {len(df_input)}"
    print(f"  ✓ Row count verified: {len(merged):,}")

    return merged


# ── Execute ───────────────────────────────────────────────────────────────────
df_train_scored = fetch_and_score(df_train, 'Training Pool')
df_vus_scored   = fetch_and_score(df_vus,   'VUS Pool')

# ── Save ──────────────────────────────────────────────────────────────────────
df_train_scored.to_csv('data/processed/brca1_features_scored.csv', index=False)
df_vus_scored.to_csv('data/processed/vus_features_scored.csv',     index=False)

print(f"\n[Saved]")
print(f"  brca1_features_scored.csv  : {len(df_train_scored):,} rows")
print(f"  vus_features_scored.csv    : {len(df_vus_scored):,} rows")

# ── Verify one variant manually ───────────────────────────────────────────────
print("\n[Verification] Sample scored training variant:")
sample = df_train_scored[
    df_train_scored['REVEL_score'].notna()
].head(1)[['Name', 'REVEL_score', 'CADD_phred',
           'BayesDel_addAF', 'phyloP17way', 'gnomAD_AF']].to_string()
print(sample)

print("\n" + "=" * 60)
print("PHASE 3 STEP 1 FINAL COMPLETE")
print("=" * 60)
print("→ Run: python src/update_features.py")