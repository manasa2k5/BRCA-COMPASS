# ============================================================
# PATIENT-LEVEL MULTI-VARIANT BRCA1 ANALYSIS
# File: src/patient_analysis.py
#
# Handles:
#   - Multiple variants per patient
#   - HGVS normalization (legacy notation, dupC, insC, etc.)
#   - Novel variant prediction via MyVariant.info
#   - Patient-level risk aggregation
#   - Clinically correct SHAP explanation (signed, separated)
# ============================================================

import pandas as pd
import numpy as np
import json
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/patients', exist_ok=True)

# ── Load model artifacts ──────────────────────────────────────────────────────
xgb_base = joblib.load('models/xgb_base.pkl')
imputer  = joblib.load('models/imputer.pkl')

with open('data/processed/feature_cols_clean.json') as f:
    FEATURE_COLS = json.load(f)

df_ref = pd.read_csv(
    'data/processed/brca1_features_scored.csv', low_memory=False
)

print(f"Reference database loaded: {len(df_ref):,} BRCA1 variants")

import shap as shap_lib
_explainer = None

def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = shap_lib.TreeExplainer(xgb_base)
    return _explainer


# ============================================================
# SECTION 1: HGVS NORMALIZATION
# ============================================================

def normalize_query(query):
    """
    Normalize variant notation to maximize lookup success.
    Returns ordered list of candidates to try.
    """
    q = str(query).strip()
    candidates = [q]

    # Rule 1: Strip nucleotide from dup notation
    # c.5266dupC --> c.5266dup
    dup_stripped = re.sub(r'(dup)[ACGT]+', r'\1', q, flags=re.IGNORECASE)
    if dup_stripped != q:
        candidates.append(dup_stripped)

    # Rule 2: Strip nucleotide from del notation
    # c.1234delA --> c.1234del
    del_stripped = re.sub(r'(del)[ACGT]+', r'\1', q, flags=re.IGNORECASE)
    if del_stripped != q:
        candidates.append(del_stripped)

    # Rule 3: Legacy alias map for well-known BRCA1 variants
    legacy_map = {
        '5382insc'  : 'c.5266dup',
        '5382ins'   : 'c.5266dup',
        '185delag'  : 'c.68_69del',
        '185delag'  : 'c.68_69del',
        '300t>g'    : 'c.181T>G',
        '1294del40' : 'c.1175_1214del',
        '3819del5'  : 'c.3700_3704del',
        '3875del4'  : 'c.3756_3759del',
        '6174delt'  : 'c.5946del',
    }
    q_lower = q.lower().strip()
    if q_lower in legacy_map:
        candidates.append(legacy_map[q_lower])

    # Rule 4: ins --> dup for single nucleotide insertions
    # c.5266insC --> c.5266dup
    ins_to_dup = re.sub(r'ins[ACGT]$', 'dup', q, flags=re.IGNORECASE)
    if ins_to_dup != q:
        candidates.append(ins_to_dup)

    # Rule 5: Extract c. notation from full HGVS string
    # NM_007294.4(BRCA1):c.5266dup (p.Gln1756fs) --> c.5266dup
    c_match = re.search(r'(c\.[^\s\)]+)', q)
    if c_match:
        candidates.append(c_match.group(1))

    # Rule 6: Extract p. notation if embedded
    p_match = re.search(r'(p\.[^\s\)]+)', q)
    if p_match:
        candidates.append(p_match.group(1))

    # Rule 7: Try numeric part only for simple queries
    num_match = re.search(r'(\d{4,})', q)
    if num_match:
        candidates.append(num_match.group(1))

    # Deduplicate preserving order
    seen   = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ============================================================
# SECTION 2: VARIANT LOOKUP
# ============================================================

def lookup_variant(query):
    """
    Look up variant with normalization fallback chain.
    Returns (matched_row, matched_alias) or (None, None).
    """
    candidates = normalize_query(query)

    for candidate in candidates:
        q = str(candidate).strip()

        # Name column search
        match = df_ref[df_ref['Name'].astype(str).str.contains(
            re.escape(q), case=False, na=False
        )]
        if len(match) > 0:
            return match.iloc[[0]], candidate

        # VariationID (numeric)
        try:
            vid   = int(q)
            match = df_ref[df_ref['VariationID'] == vid]
            if len(match) > 0:
                return match.iloc[[0]], candidate
        except:
            pass

        # rsID in OtherIDs column
        if q.lower().startswith('rs') and 'OtherIDs' in df_ref.columns:
            match = df_ref[df_ref['OtherIDs'].astype(str).str.contains(
                q, case=False, na=False
            )]
            if len(match) > 0:
                return match.iloc[[0]], candidate

    return None, None


# ============================================================
# SECTION 3: NOVEL VARIANT FEATURE ENGINEERING
# ============================================================

def engineer_novel_variant(query):
    """
    For a variant not in the reference database:
    1. Parse HGVS notation to extract consequence features
    2. Fetch in-silico scores via MyVariant.info
    3. Return feature DataFrame ready for model prediction
    Returns (feature_df, score_coverage_pct) or (None, 0)
    """
    try:
        import myvariant
        mv = myvariant.MyVariantInfo()
    except ImportError:
        print("    myvariant not installed — cannot score novel variant")
        return None, 0

    print(f"    Novel variant — attempting feature engineering...")

    q = str(query).strip()
    feat = {col: 0 for col in FEATURE_COLS}

    # Variant type
    q_lower = q.lower()
    if 'del' in q_lower and 'ins' not in q_lower:
        feat['variant_type_encoded'] = 1
        feat['cons_deletion']        = 1
    elif 'ins' in q_lower and 'del' not in q_lower:
        feat['variant_type_encoded'] = 2
        feat['cons_insertion']       = 1
    elif 'dup' in q_lower:
        feat['variant_type_encoded'] = 3
        feat['cons_duplication']     = 1
    elif '>' in q_lower:
        feat['variant_type_encoded'] = 0
        feat['cons_snv']             = 1
    else:
        feat['variant_type_encoded'] = 5

    # Frameshift
    if 'fs' in q_lower or (
        ('del' in q_lower or 'ins' in q_lower or 'dup' in q_lower) and
        not re.search(r'_\d+', q)
    ):
        feat['is_frameshift']         = 1
        feat['is_frameshift_protein'] = 1
        feat['cons_frameshift']       = 1
        feat['is_likely_lof']         = 1

    # Nonsense
    if 'ter' in q_lower or q.endswith('*'):
        feat['is_nonsense']   = 1
        feat['is_likely_lof'] = 1
        feat['cons_nonsense'] = 1

    # Splice
    if re.search(r'c\.\d+[+\-][12][^0-9]', q):
        feat['is_canonical_splice_site'] = 1
        feat['cons_splice']              = 1
        feat['is_likely_lof']            = 1

    # Missense
    if re.search(r'p\.[A-Za-z]{3}\d+[A-Za-z]{3}$', q):
        feat['is_missense']        = 1
        feat['has_protein_change'] = 1

    # Synonymous
    if '=' in q:
        feat['is_synonymous']      = 1
        feat['has_protein_change'] = 1

    # Amino acid position and domain
    p_match = re.search(r'p\.[A-Za-z]{1,3}(\d+)', q)
    if p_match:
        aa_pos = int(p_match.group(1))
        feat['aa_position']          = aa_pos
        feat['in_RING_domain']       = int(1    <= aa_pos <= 109)
        feat['in_BRCT_domain']       = int(1642 <= aa_pos <= 1863)
        feat['in_coiled_coil']       = int(1391 <= aa_pos <= 1424)
        feat['in_disordered_region'] = int(
            (110 <= aa_pos <= 1390) or (1425 <= aa_pos <= 1641)
        )
        feat['in_pathogenic_hotspot'] = int(
            feat['in_RING_domain'] or feat['in_BRCT_domain']
        )
        feat['domain_known'] = 1

    feat['review_strength'] = 1
    feat['num_submitters']  = 1

    # Fetch in-silico scores via MyVariant.info
    hgvs_id          = None
    score_cols_found  = 0
    c_match           = re.search(r'c\.(\S+)', q)

    if c_match:
        try:
            results = mv.query(
                f'brca1 {q}',
                fields='dbnsfp,gnomad_exome',
                assembly='hg38',
                size=1
            )
            if results and results.get('hits'):
                hgvs_id = results['hits'][0].get('_id', '')
        except:
            pass

    if hgvs_id:
        try:
            FIELDS = ','.join([
                'dbnsfp.bayesdel', 'dbnsfp.revel', 'dbnsfp.cadd',
                'dbnsfp.sift', 'dbnsfp.polyphen2', 'dbnsfp.phylop',
                'dbnsfp.phastcons', 'dbnsfp.alphamissense',
                'dbnsfp.dann', 'gnomad_exome.af'
            ])
            r = mv.getvariant(hgvs_id, fields=FIELDS, assembly='hg38')
            if r and 'dbnsfp' in r:
                d = r['dbnsfp']

                def safe(val):
                    if val is None: return np.nan
                    if isinstance(val, list):
                        nums = [float(v) for v in val if v is not None]
                        return float(np.mean(nums)) if nums else np.nan
                    try: return float(val)
                    except: return np.nan

                def gn(obj, *keys):
                    for k in keys:
                        if isinstance(obj, list):
                            obj = obj[0] if obj else None
                        if isinstance(obj, dict):
                            obj = obj.get(k)
                        else:
                            return np.nan
                    return safe(obj)

                bd  = d.get('bayesdel')      or {}
                rv  = d.get('revel')         or {}
                ca  = d.get('cadd')          or {}
                sf  = d.get('sift')          or {}
                pp  = d.get('polyphen2')     or {}
                phy = d.get('phylop')        or {}
                phs = d.get('phastcons')     or {}
                am  = d.get('alphamissense') or {}
                dn  = d.get('dann')          or {}
                gex = r.get('gnomad_exome')  or {}

                score_map = {
                    'BayesDel_addAF'     : gn(bd,'add_af','score') if isinstance(bd,dict) else np.nan,
                    'BayesDel_noAF'      : gn(bd,'no_af','score')  if isinstance(bd,dict) else np.nan,
                    'REVEL_score'        : gn(rv,'score'),
                    'CADD_phred'         : gn(ca,'phred'),
                    'SIFT_score'         : gn(sf,'score'),
                    'Polyphen2_HDIV'     : gn(pp,'hdiv','score'),
                    'Polyphen2_HVAR'     : gn(pp,'hvar','score'),
                    'phyloP17way'        : gn(phy,'17way_primate'),
                    'phastCons17way'     : gn(phs,'17way_primate'),
                    'AlphaMissense_score': gn(am,'score'),
                    'DANN_score'         : gn(dn,'score'),
                }
                af_obj = gex.get('af') or {}
                score_map['gnomAD_AF'] = (
                    gn(af_obj, 'af') if isinstance(af_obj, dict)
                    else safe(af_obj)
                )

                for col, val in score_map.items():
                    if col in feat and not np.isnan(val):
                        feat[col] = val
                        score_cols_found += 1

                print(f"    In-silico scores fetched: {score_cols_found}")
        except Exception as e:
            print(f"    Score fetch failed: {e}")

    coverage_pct = score_cols_found / 10 * 100
    X = pd.DataFrame([feat])[FEATURE_COLS]
    return X, coverage_pct


# ============================================================
# SECTION 4: SHAP INTERPRETATION LAYER
#
# Scientific basis:
#   XGBoost TreeExplainer returns SHAP values in log-odds space.
#   Positive SHAP = pushes prediction toward Pathogenic (label=1).
#   Negative SHAP = pushes prediction toward Benign (label=0).
#
#   Critical distinction for clinical reporting:
#   A negative SHAP on a binary absence feature (e.g. is_likely_lof=0)
#   means "this variant lacks a LOF consequence" — it is not independent
#   evidence of benignity. Presenting it as "supports Benign" is
#   scientifically misleading. These features require context-aware
#   language that distinguishes absence-of-pathogenic-signal from
#   presence-of-benign-signal.
# ============================================================

# Features where a negative SHAP value means absence of a pathogenic
# signal, NOT presence of a benign signal. These need neutral language.
ABSENCE_FEATURES = {
    'is_likely_lof',
    'cons_frameshift',
    'cons_nonsense',
    'cons_splice',
    'is_nonsense',
    'is_frameshift_protein',
    'is_canonical_splice_site',
    'in_pathogenic_hotspot',
    'in_RING_domain',
    'in_BRCT_domain',
}

# Features where a positive SHAP value means absence of a benign
# signal (e.g. cons_snv=1 correlates with benign in training data).
# Positive SHAP here does NOT mean "strong pathogenic evidence".
AMBIGUOUS_POSITIVE_FEATURES = {
    'cons_snv',          # SNV type correlates with benign in training
    'is_synonymous',     # synonymous = usually benign
    'in_disordered_region',  # disordered region = lower pathogenic prior
}

NOISE_THRESHOLD = 0.05  # SHAP values below this are not clinically meaningful


def classify_shap_contribution(feat_name, shap_value):
    """
    Classify a single SHAP contribution into one of four categories:
      'pathogenic'  — genuine evidence supporting pathogenicity
      'benign'      — genuine evidence supporting benign classification
      'absence_lof' — absence of LOF signal (not positive benign evidence)
      'contextual'  — ambiguous or low-signal contribution
      'noise'       — below threshold, omit from report

    Returns (category, clinical_label, strength_word)
    """
    if abs(shap_value) < NOISE_THRESHOLD:
        return 'noise', None, None

    strength = (
        'strong'    if abs(shap_value) > 1.0 else
        'moderate'  if abs(shap_value) > 0.3 else
        'weak'
    )

    if shap_value > 0:
        # Positive SHAP pushes toward Pathogenic
        if feat_name in AMBIGUOUS_POSITIVE_FEATURES:
            # These are artifacts of training distribution,
            # not genuine pathogenic evidence
            return 'contextual', None, strength
        return 'pathogenic', None, strength

    else:
        # Negative SHAP pushes toward Benign
        if feat_name in ABSENCE_FEATURES:
            # This variant lacks a LOF/hotspot feature.
            # That is context, not independent benign evidence.
            return 'absence_lof', None, strength
        return 'benign', None, strength


def format_shap_explanation(top5_names, top5_shap_signs):
    """
    Separate SHAP contributions into three clinically honest groups:
      1. Evidence supporting Pathogenic classification
      2. Contextual factors (absence of pathogenic signals)
      3. Evidence supporting Benign classification

    Returns three lists of (feature_name, clinical_label, strength, category).
    """
    path_evidence    = []
    absence_context  = []
    benign_evidence  = []

    for feat, shap_val in zip(top5_names, top5_shap_signs):
        category, _, strength = classify_shap_contribution(feat, shap_val)

        if category == 'noise':
            continue

        label = readable_feature(feat)

        if category == 'pathogenic':
            path_evidence.append((feat, label, strength))

        elif category == 'absence_lof':
            # Translate absence features into neutral plain-language context
            absence_map = {
                'is_likely_lof'           : 'Not a frameshift, nonsense, or splice-disrupting variant',
                'cons_frameshift'         : 'No frameshift consequence detected',
                'cons_nonsense'           : 'No protein-truncating consequence detected',
                'cons_splice'             : 'Not located at a canonical splice site',
                'is_nonsense'             : 'No stop-codon-introducing change',
                'is_frameshift_protein'   : 'No frameshift at protein level',
                'is_canonical_splice_site': 'Not at a canonical +1/+2/-1/-2 splice position',
                'in_pathogenic_hotspot'   : 'Not located in RING or BRCT hotspot region',
                'in_RING_domain'          : 'Not located in RING domain',
                'in_BRCT_domain'          : 'Not located in BRCT domain',
            }
            neutral_label = absence_map.get(feat, f'Absent: {label}')
            absence_context.append((feat, neutral_label, strength))

        elif category == 'benign':
            benign_evidence.append((feat, label, strength))

        # 'contextual' category is silently dropped — not clinically useful

    return path_evidence, absence_context, benign_evidence


# ============================================================
# SECTION 5: PREDICT ONE VARIANT
# ============================================================

def predict_variant(X_raw, is_novel=False):
    """
    Apply imputer + XGBoost + SHAP + ACMG to one variant.
    is_novel: if True, cap confidence at Medium (less certainty
    about novel variants not seen during training).
    """
    X_imp = imputer.transform(X_raw.values)
    prob  = float(xgb_base.predict_proba(X_imp)[0, 1])
    pred  = 'Pathogenic' if prob >= 0.5 else 'Benign'
    dist  = abs(prob - 0.5)

    if is_novel:
        conf = 'Medium' if dist >= 0.35 else 'Low'
    else:
        conf = 'High' if dist >= 0.35 else 'Medium' if dist >= 0.15 else 'Low'

    # SHAP — extract signed values (positive = toward Pathogenic)
    explainer  = get_explainer()
    sv         = explainer.shap_values(X_imp)
    shap_s     = pd.Series(sv[0], index=FEATURE_COLS)

    # Rank by absolute magnitude, retrieve signed values
    top5_names  = list(shap_s.abs().nlargest(5).index)
    top5_signed = [float(shap_s[f]) for f in top5_names]
    top5_abs    = [round(abs(float(shap_s[f])), 4) for f in top5_names]

    # ACMG evidence mapping
    fv   = dict(zip(FEATURE_COLS, X_imp[0]))
    acmg = []
    if pred == 'Pathogenic':
        if fv.get('is_likely_lof', 0) == 1 and 'is_likely_lof' in top5_names:
            acmg.append('PVS1')
        if fv.get('in_pathogenic_hotspot', 0) == 1 and 'in_pathogenic_hotspot' in top5_names:
            acmg.append('PM1')
        if any(c in top5_names for c in
               ['BayesDel_addAF', 'REVEL_score', 'AlphaMissense_score']):
            acmg.append('PP3_Strong' if prob >= 0.90 else 'PP3')
        if fv.get('is_canonical_splice_site', 0) == 1:
            acmg.append('PS3_Moderate')
        if fv.get('in_BRCT_domain', 0) == 1 and 'in_BRCT_domain' in top5_names:
            acmg.append('PM1_BRCT')
    else:
        if any(c in top5_names for c in
               ['BayesDel_addAF', 'REVEL_score', 'AlphaMissense_score']):
            acmg.append('BP4')
        if fv.get('gnomAD_AF', 0) > 0.001:
            acmg.append('BS1')
        if fv.get('is_synonymous', 0) == 1:
            acmg.append('BP7')
        if fv.get('in_disordered_region', 0) == 1 and prob < 0.10:
            acmg.append('BP1')

    return {
        'probability'        : round(prob, 4),
        'probability_pct'    : round(prob * 100, 1),
        'predicted_class'    : pred,
        'confidence'         : conf,
        'top5_shap_features' : top5_names,
        'top5_shap_values'   : top5_abs,
        'top5_shap_signs'    : top5_signed,
        'acmg_codes'         : acmg if acmg else ['Insufficient_evidence'],
        'is_novel'           : is_novel,
    }


# ============================================================
# SECTION 6: PATIENT RISK AGGREGATION
# ============================================================

def patient_risk_summary(results):
    n_total  = len(results)
    n_path   = sum(1 for v in results if v['predicted_class'] == 'Pathogenic')
    n_ben    = sum(1 for v in results if v['predicted_class'] == 'Benign')
    n_high_p = sum(1 for v in results
                   if v['predicted_class'] == 'Pathogenic'
                   and v['confidence'] == 'High')
    n_high_b = sum(1 for v in results
                   if v['predicted_class'] == 'Benign'
                   and v['confidence'] == 'High')
    max_prob = max(
        (v['probability'] for v in results if v['predicted_class'] == 'Pathogenic'),
        default=0.0
    )

    if n_high_p >= 1:
        tier = 'HIGH'
        msg  = (
            f"{n_high_p} variant(s) carry high-confidence pathogenic prediction. "
            "Urgent referral to a certified genetic specialist is strongly recommended. "
            "Confirmatory laboratory testing and cascade family screening "
            "should be considered."
        )
    elif n_path >= 1:
        tier = 'MODERATE'
        msg  = (
            f"{n_path} variant(s) show suggestive pathogenic evidence. "
            "Genetic counseling and further clinical evaluation are recommended. "
            "Monitor for updated classifications from clinical databases."
        )
    elif n_ben == n_total and n_high_b >= round(n_total * 0.8):
        tier = 'LOW'
        msg  = (
            "All analyzed variants show high-confidence benign predictions. "
            "No strong evidence of pathogenic BRCA1 variants was identified. "
            "Standard cancer screening guidelines apply. "
            "Periodic re-evaluation is recommended as evidence evolves."
        )
    else:
        tier = 'UNCERTAIN'
        msg  = (
            "Mixed or low-confidence predictions were obtained. "
            "Genetic counseling is recommended for comprehensive evaluation. "
            "Additional functional or family history evidence may be required."
        )

    return {
        'n_total'          : n_total,
        'n_pathogenic'     : n_path,
        'n_benign'         : n_ben,
        'n_high_conf_path' : n_high_p,
        'n_high_conf_ben'  : n_high_b,
        'max_path_prob'    : round(max_prob, 4),
        'tier'             : tier,
        'message'          : msg,
    }


# ============================================================
# SECTION 7: FEATURE LABELS AND REPORT BUILDER
# ============================================================

FEATURE_LABELS = {
    'is_likely_lof'           : 'Loss-of-function consequence',
    'cons_frameshift'         : 'Frameshift variant',
    'cons_nonsense'           : 'Nonsense (stop-gain) variant',
    'cons_splice'             : 'Splice site disruption',
    'cons_snv'                : 'Single nucleotide substitution',
    'cons_deletion'           : 'Deletion variant',
    'cons_insertion'          : 'Insertion variant',
    'is_missense'             : 'Missense amino acid change',
    'is_synonymous'           : 'Synonymous (silent) change',
    'is_nonsense'             : 'Protein-truncating change',
    'is_frameshift_protein'   : 'Frameshift protein change',
    'is_canonical_splice_site': 'Canonical splice site position',
    'in_RING_domain'          : 'Located in RING functional domain',
    'in_BRCT_domain'          : 'Located in BRCT functional domain',
    'in_coiled_coil'          : 'Located in coiled-coil region',
    'in_disordered_region'    : 'Located in disordered linker region',
    'in_pathogenic_hotspot'   : 'In known pathogenic hotspot (RING/BRCT)',
    'BayesDel_addAF'          : 'BayesDel pathogenicity score',
    'BayesDel_noAF'           : 'BayesDel score (population-free)',
    'REVEL_score'             : 'REVEL missense pathogenicity score',
    'CADD_phred'              : 'CADD deleteriousness score',
    'AlphaMissense_score'     : 'AlphaMissense structural impact score',
    'SIFT_score'              : 'SIFT functional tolerance score',
    'Polyphen2_HDIV'          : 'PolyPhen-2 structural impact score',
    'phyloP17way'             : 'Evolutionary conservation (phyloP)',
    'phastCons17way'          : 'Evolutionary conservation (phastCons)',
    'DANN_score'              : 'DANN deleteriousness score',
    'gnomAD_AF'               : 'Population allele frequency (gnomAD)',
    'aa_position'             : 'Amino acid position in protein',
    'normalized_position'     : 'Genomic position within BRCA1',
    'review_strength'         : 'Clinical evidence strength',
    'num_submitters'          : 'Number of independent observations',
}

def readable_feature(feat_name):
    return FEATURE_LABELS.get(
        feat_name,
        feat_name.replace('_', ' ').title()
    )


def build_report(patient_id, variants, summary, not_found):
    L   = []
    SEP = "=" * 68
    DIV = "─" * 68

    L.append(SEP)
    L.append("       BRCA1 VARIANT PATHOGENICITY ANALYSIS REPORT")
    L.append(SEP)
    L.append(f"  Patient Reference   : {patient_id}")
    L.append(f"  Gene Analyzed       : BRCA1")
    L.append(f"  Variants Submitted  : {len(variants) + len(not_found)}")
    L.append(f"  Variants Analyzed   : {len(variants)}")
    if not_found:
        L.append(f"  Unresolved Queries  : {len(not_found)}")
    L.append("")

    tier_label = {
        'HIGH'     : '*** HIGH RISK ***',
        'MODERATE' : '**  MODERATE RISK  **',
        'LOW'      : '*   LOW RISK   *',
        'UNCERTAIN': '~   UNCERTAIN   ~',
    }
    L.append(DIV)
    L.append(
        f"  OVERALL RISK ASSESSMENT :  "
        f"{tier_label.get(summary['tier'], summary['tier'])}"
    )
    L.append(DIV)
    L.append("")

    # Word-wrap risk message
    words = summary['message'].split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > 68:
            L.append(line.rstrip())
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        L.append(line.rstrip())
    L.append("")

    # Per-variant details
    L.append(DIV)
    L.append("  VARIANT-BY-VARIANT ANALYSIS")
    L.append("  (Ranked by pathogenic probability, highest first)")
    L.append(DIV)

    for i, v in enumerate(variants, 1):
        L.append("")
        name = v['matched_name']
        if len(name) > 62:
            name = name[:60] + ".."
        L.append(f"  [{i}]  {name}")
        if v.get('is_novel'):
            L.append(
                "       Note: Novel variant — prediction based on "
                "computational annotation."
            )
        L.append("")

        # Probability bar
        filled = int(v['probability'] * 28)
        bar    = "█" * filled + "░" * (28 - filled)
        L.append(f"       Benign  ◄{'─'*14}┤{'─'*14}►  Pathogenic")
        L.append(f"       [{bar}]  {v['probability_pct']}%")
        L.append("")

        L.append(f"       Result      :  {v['predicted_class'].upper()}")
        L.append(f"       Confidence  :  {v['confidence']}")
        L.append(f"       ACMG Codes  :  {', '.join(v['acmg_codes'])}")
        L.append("")

        # ── CORRECTED SHAP EXPLANATION BLOCK ─────────────────────────────────
        # Uses format_shap_explanation() which separates:
        #   1. Genuine pathogenic evidence (positive SHAP, non-ambiguous)
        #   2. Contextual absence factors (negative SHAP on LOF features)
        #   3. Genuine benign evidence (negative SHAP on non-LOF features)
        # This prevents the clinically misleading pattern of showing
        # "supports Benign" for features that simply lack a LOF signal.
        path_ev, absence_ctx, benign_ev = format_shap_explanation(
            v['top5_shap_features'],
            v['top5_shap_signs']
        )

        if path_ev:
            L.append("       Evidence supporting Pathogenic classification:")
            for feat, label, strength in path_ev:
                L.append(f"         ✦  {label:<42}  ({strength})")
            L.append("")

        if absence_ctx:
            L.append("       Variant context (not independent benign evidence):")
            for feat, label, strength in absence_ctx:
                L.append(f"         ◦  {label}")
            L.append("")

        if benign_ev:
            L.append("       Factors partially offsetting pathogenic score:")
            for feat, label, strength in benign_ev:
                L.append(f"         ▽  {label:<42}  ({strength})")
            L.append("")

        if not path_ev and not benign_ev and not absence_ctx:
            L.append(
                "       Prediction driven by combined effect of multiple "
                "weak signals."
            )
            L.append("")
        # ── END SHAP BLOCK ────────────────────────────────────────────────────

        # Priority flag
        if v['predicted_class'] == 'Pathogenic' and v['confidence'] == 'High':
            L.append(
                "       ► HIGH PRIORITY — Urgent clinical review recommended"
            )
        elif v['predicted_class'] == 'Pathogenic' and v['confidence'] == 'Medium':
            L.append(
                "       ► MODERATE PRIORITY — Genetic counselor consultation advised"
            )
        elif v['predicted_class'] == 'Pathogenic' and v['confidence'] == 'Low':
            L.append(
                "       ► LOW CONFIDENCE PATHOGENIC — Additional evidence required"
            )
        elif v['predicted_class'] == 'Benign' and v['confidence'] == 'High':
            L.append(
                "       ► LOW PRIORITY — High-confidence benign prediction"
            )
        else:
            L.append("       ► UNCERTAIN — Additional evidence required")

        L.append("")
        L.append("  " + "─" * 64)

    # Unresolved variants
    if not_found:
        L.append("")
        L.append(DIV)
        L.append("  UNRESOLVED VARIANTS")
        L.append(DIV)
        for vq in not_found:
            L.append(f"    • {vq}")
        L.append("")
        L.append("  These variants could not be matched in the reference")
        L.append("  database after normalization. They may be genuinely")
        L.append("  novel or use non-standard notation. Manual specialist")
        L.append("  review is required.")

    # Summary table
    L.append("")
    L.append(DIV)
    L.append("  SUMMARY TABLE")
    L.append(DIV)
    L.append(
        f"  {'#':<3} {'Variant':<38} {'Result':<14} "
        f"{'Prob%':>6}  {'Confidence'}"
    )
    L.append(
        f"  {'─'*3} {'─'*38} {'─'*14} {'─'*6}  {'─'*10}"
    )
    for i, v in enumerate(variants, 1):
        n = (
            v['matched_name'][:36] + ".."
            if len(v['matched_name']) > 36
            else v['matched_name']
        )
        novel_flag = " *" if v.get('is_novel') else ""
        L.append(
            f"  {i:<3} {n:<38} {v['predicted_class']:<14} "
            f"{v['probability_pct']:>5.1f}%  {v['confidence']}{novel_flag}"
        )
    if any(v.get('is_novel') for v in variants):
        L.append(
            "  * Novel variant prediction (computational annotation only)"
        )

    # Next steps
    steps = {
        'HIGH': [
            "1. Refer immediately to a certified genetic counselor",
            "2. Arrange confirmatory laboratory testing",
            "3. Assess complete family history for BRCA1-related cancers",
            "4. Discuss preventive options with a specialist oncologist",
            "5. Consider cascade genetic testing for first-degree relatives",
        ],
        'MODERATE': [
            "1. Schedule genetic counselor consultation",
            "2. Review personal and family cancer history",
            "3. Consider enhanced surveillance protocols",
            "4. Monitor variant databases for reclassification updates",
        ],
        'LOW': [
            "1. Continue standard cancer screening guidelines",
            "2. Document findings in the patient record",
            "3. Re-evaluate periodically as new evidence emerges",
        ],
        'UNCERTAIN': [
            "1. Genetic counselor consultation for full clinical assessment",
            "2. Consider functional studies for unresolved variants",
            "3. Await updated ClinVar and ENIGMA classifications",
            "4. Review family history for additional risk context",
        ],
    }
    L.append("")
    L.append(DIV)
    L.append("  RECOMMENDED NEXT STEPS")
    L.append(DIV)
    for step in steps.get(summary['tier'], steps['UNCERTAIN']):
        L.append(f"    {step}")

    L.append("")
    L.append(DIV)
    L.append("  IMPORTANT NOTICE")
    L.append(DIV)
    L.append(
        "  This report is produced by a computational analysis tool"
    )
    L.append(
        "  and is intended to support — not replace — clinical judgment."
    )
    L.append(
        "  All findings must be reviewed and interpreted by a qualified"
    )
    L.append(
        "  medical geneticist or certified genetic counselor before any"
    )
    L.append("  clinical or personal decisions are made.")
    L.append("")
    L.append(SEP)

    return L


# ============================================================
# SECTION 8: MAIN PATIENT ANALYSIS FUNCTION
# ============================================================

def analyze_patient(patient_id, variant_queries, save=True):
    """
    Analyze all BRCA1 variants for one patient.

    Args:
        patient_id      : str  — patient reference ID (anonymized)
        variant_queries : list — HGVS names, VariationIDs, rsIDs, etc.
        save            : bool — save report and CSV to results/patients/

    Returns:
        dict with variants, summary, report text
    """
    print(f"\n{'='*65}")
    print(f"  Patient: {patient_id}")
    print(f"  Variants submitted: {len(variant_queries)}")
    print(f"{'='*65}")

    results   = []
    not_found = []

    for i, query in enumerate(variant_queries, 1):
        print(f"\n  [{i}/{len(variant_queries)}] Query: {query}")

        # Step 1: Try database lookup with normalization
        row, matched_alias = lookup_variant(query)

        if row is not None:
            print(f"    Found in database via: '{matched_alias}'")
            print(f"    Name: {row['Name'].values[0][:58]}")

            feat_dict = {}
            for col in FEATURE_COLS:
                feat_dict[col] = (
                    row[col].values[0] if col in row.columns else np.nan
                )
            X_raw = pd.DataFrame([feat_dict])[FEATURE_COLS]

            pred = predict_variant(X_raw, is_novel=False)
            pred['query']         = query
            pred['matched_name']  = str(row['Name'].values[0])
            pred['variation_id']  = (
                str(row['VariationID'].values[0])
                if 'VariationID' in row.columns else 'N/A'
            )
            pred['matched_alias'] = matched_alias
            results.append(pred)

        else:
            # Step 2: Attempt novel variant engineering
            print(
                f"    Not in database — attempting novel variant analysis..."
            )
            X_novel, coverage = engineer_novel_variant(query)

            if X_novel is not None:
                pred = predict_variant(X_novel, is_novel=True)
                pred['query']          = query
                pred['matched_name']   = query
                pred['variation_id']   = 'Novel'
                pred['matched_alias']  = 'novel_engineered'
                pred['score_coverage'] = coverage
                results.append(pred)
                print(
                    f"    Novel variant prediction generated "
                    f"(score coverage: {coverage:.0f}%)"
                )
            else:
                print(
                    f"    Could not generate prediction — added to unresolved"
                )
                not_found.append(query)
                continue

        v = results[-1]
        print(f"    Result     : {v['predicted_class']}")
        print(f"    Probability: {v['probability_pct']}%")
        print(f"    Confidence : {v['confidence']}")
        print(f"    ACMG       : {', '.join(v['acmg_codes'])}")

    if not results:
        print(f"\n  No variants could be analyzed.")
        return None

    # Sort by probability descending
    results.sort(key=lambda x: x['probability'], reverse=True)

    summary      = patient_risk_summary(results)
    report_lines = build_report(patient_id, results, summary, not_found)
    report_text  = "\n".join(report_lines)

    print(f"\n  {'─'*55}")
    print(f"  PATIENT RISK TIER : {summary['tier']}")
    print(f"  Pathogenic        : {summary['n_pathogenic']}")
    print(f"  Benign            : {summary['n_benign']}")
    print(f"  High-conf path    : {summary['n_high_conf_path']}")
    print(f"  Unresolved        : {len(not_found)}")

    if save:
        txt_path = f"results/patients/{patient_id}_report.txt"
        csv_path = f"results/patients/{patient_id}_variants.csv"

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        csv_rows = []
        for v in results:
            csv_rows.append({
                'Patient_ID'      : patient_id,
                'Query'           : v['query'],
                'Variant'         : v['matched_name'],
                'VariationID'     : v['variation_id'],
                'Result'          : v['predicted_class'],
                'Probability_Pct' : v['probability_pct'],
                'Confidence'      : v['confidence'],
                'ACMG_Codes'      : ', '.join(v['acmg_codes']),
                'Novel_Variant'   : v.get('is_novel', False),
                'Risk_Tier'       : summary['tier'],
            })
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

        print(f"\n  Report : {txt_path}")
        print(f"  CSV    : {csv_path}")

    print("\n" + report_text)

    return {
        'patient_id'  : patient_id,
        'variants'    : results,
        'summary'     : summary,
        'not_found'   : not_found,
        'report_text' : report_text,
    }


# ============================================================
# DEMO
# ============================================================

if __name__ == '__main__':

    # Patient 1 — uses legacy notation (tests normalization)
    analyze_patient(
        patient_id="PATIENT_001",
        variant_queries=[
            "c.5266dupC",    # legacy → normalizes to c.5266dup
            "c.181T>G",      # missense (p.Cys61Gly)
            "c.190T>G",      # missense
        ]
    )

    # Patient 2 — single deletion variant
    analyze_patient(
        patient_id="PATIENT_002",
        variant_queries=[
            "c.1175_1214del",
        ]
    )

    # Patient 3 — multiple VUS-like variants
    analyze_patient(
        patient_id="PATIENT_003",
        variant_queries=[
            "c.2681_2682del",
            "c.3005del",
            "c.190T>G",
        ]
    )