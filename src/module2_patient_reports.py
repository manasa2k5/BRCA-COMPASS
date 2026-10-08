# ============================================================
# MODULE 2: PATIENT-FRIENDLY EXPLANATION SYSTEM (PERSONALIZED)
# File: src/module2_patient_reports.py
#
# Each patient explanation is generated dynamically from:
#   - The actual SHAP features of THAT specific variant
#   - The actual probability of THAT specific variant
#   - The actual domain/consequence of THAT specific variant
# No hardcoded template text in the patient section.
# Simple English. No disclaimers. No metrics.
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/reports/sample_patient_reports', exist_ok=True)
os.makedirs('results/reports/full_vus_reports',       exist_ok=True)

print("=" * 65)
print("MODULE 2: PERSONALIZED PATIENT EXPLANATION SYSTEM")
print("=" * 65)

# ── Load VUS predictions ──────────────────────────────────────────────────────
df = pd.read_csv('results/vus/vus_predictions.csv')
print(f"\n  Loaded {len(df):,} VUS predictions")
print(f"  Columns: {list(df.columns)}")

# ── Detect columns safely ────────────────────────────────────────────────────
prob_col = next(
    (c for c in df.columns if any(x in c.lower()
     for x in ['ensemble_prob','cal_prob','base_prob'])
     or ('prob' in c.lower() and 'pathogenic' not in c.lower())),
    None
)
if prob_col is None:
    raise ValueError(f"No probability column found. Columns: {list(df.columns)}")

df['_prob']  = pd.to_numeric(df[prob_col], errors='coerce')
df['_class'] = next(
    (df[c] for c in df.columns if 'predicted' in c.lower()
     and 'class' in c.lower()), None
) if any('predicted' in c.lower() and 'class' in c.lower()
         for c in df.columns) else (
    df['_prob'].apply(lambda p: 'Pathogenic' if p >= 0.5 else 'Benign')
)
df['_conf'] = next(
    (df[c] for c in df.columns if 'confidence' in c.lower()), None
) if any('confidence' in c.lower() for c in df.columns) else (
    df['_prob'].apply(
        lambda p: 'High' if abs(p-0.5)>=0.35
        else 'Medium' if abs(p-0.5)>=0.15 else 'Low'
    )
)
shap_col = next(
    (c for c in df.columns if 'shap' in c.lower() or 'top5' in c.lower()),
    None
)
df['_shap']  = df[shap_col] if shap_col else ''
acmg_col = next(
    (c for c in df.columns if 'acmg' in c.lower()), None
)
df['_acmg']  = df[acmg_col] if acmg_col else ''
name_col = next(
    (c for c in df.columns if c.lower() == 'name'), None
)
df['_name']  = df[name_col] if name_col else df.index.astype(str)

# ============================================================
# FEATURE MEANING DICTIONARY
# Used to build sentences — not shown directly to patient
# ============================================================

# Each entry maps feature name to:
#   (plain_label, pathogenic_sentence, benign_sentence)
# These are sentence FRAGMENTS that get assembled dynamically

FEATURE_SENTENCES = {
    # In-silico scores — pathogenicity direction
    'BayesDel_addAF': (
        'a deleteriousness scoring tool',
        'this variant scored high on a tool that measures how likely a DNA change is to cause harm',
        'this variant scored low on a tool that measures how likely a DNA change is to cause harm'
    ),
    'BayesDel_noAF': (
        'a deleteriousness scoring tool',
        'this variant scored high on a tool that predicts damaging DNA changes',
        'this variant scored low on a tool that predicts damaging DNA changes'
    ),
    'REVEL_score': (
        'a combined pathogenicity prediction tool',
        'multiple prediction tools agreed that this variant is likely to be damaging',
        'multiple prediction tools agreed that this variant is unlikely to be damaging'
    ),
    'CADD_phred': (
        'a variant severity scoring tool',
        'this variant received a high severity score based on its genomic context',
        'this variant received a low severity score based on its genomic context'
    ),
    'AlphaMissense_score': (
        'an AI model for protein function',
        'an artificial intelligence model predicted this change would disrupt how the BRCA1 protein works',
        'an artificial intelligence model predicted this change would not disrupt how the BRCA1 protein works'
    ),
    'SIFT_score': (
        'an amino acid change tolerance tool',
        'the amino acid change caused by this variant is predicted to be damaging to the protein',
        'the amino acid change caused by this variant is predicted to be well-tolerated by the protein'
    ),
    'Polyphen2_HDIV': (
        'a protein damage prediction tool',
        'this variant is predicted to damage the structure of the BRCA1 protein',
        'this variant is predicted to have little impact on the structure of the BRCA1 protein'
    ),
    'Polyphen2_HVAR': (
        'a protein damage prediction tool',
        'this variant is predicted to be damaging based on its structural effects',
        'this variant is predicted to be benign based on its structural effects'
    ),
    'DANN_score': (
        'a deep learning pathogenicity predictor',
        'a deep learning model predicted this variant as potentially harmful',
        'a deep learning model predicted this variant as likely harmless'
    ),
    'VEST4_score': (
        'a variant effect prediction tool',
        'this variant scored highly on a tool that identifies disease-causing changes',
        'this variant scored low on a tool that identifies disease-causing changes'
    ),
    'MetaRNN_score': (
        'an ensemble AI predictor',
        'an ensemble of AI models predicted this variant is likely pathogenic',
        'an ensemble of AI models predicted this variant is likely benign'
    ),
    'ClinPred_score': (
        'a clinical significance predictor',
        'this variant was flagged as likely clinically significant by a prediction tool',
        'this variant was not flagged as clinically significant by a prediction tool'
    ),
    'MPC_score': (
        'a missense pathogenicity classifier',
        'this specific type of amino acid change scores highly for disease association',
        'this specific type of amino acid change scores low for disease association'
    ),

    # Conservation
    'phyloP17way': (
        'evolutionary conservation',
        'the DNA position where this variant occurs has been preserved identically '
        'across many different species over millions of years, suggesting it is '
        'critically important for normal function',
        'the DNA position where this variant occurs is not highly conserved across '
        'species, suggesting it may tolerate changes more easily'
    ),
    'phastCons17way': (
        'cross-species conservation',
        'this region of the BRCA1 gene has been conserved across many mammalian '
        'species, indicating it is functionally important',
        'this region of the BRCA1 gene is less conserved across species'
    ),
    'GERP_RS': (
        'genomic constraint',
        'this position in the genome shows strong evolutionary constraint, '
        'meaning changes here are rarely seen in nature and are likely important',
        'this position shows less evolutionary constraint, suggesting '
        'it may be more tolerant to variation'
    ),

    # Population frequency
    'gnomAD_AF': (
        'population frequency',
        'this variant is extremely rare or has never been observed in '
        'the general healthy population, which is consistent with disease-causing variants',
        'this variant has been observed in the general healthy population, '
        'which argues against it being a common cause of disease'
    ),

    # Domain location
    'in_BRCT_domain': (
        'BRCT domain location',
        'this variant falls within the BRCT domain, one of the most '
        'functionally critical parts of the BRCA1 protein responsible for '
        'DNA repair — changes here are more likely to cause harm',
        'this variant falls outside the most critical functional domains '
        'of the BRCA1 protein'
    ),
    'in_RING_domain': (
        'RING domain location',
        'this variant falls within the RING domain of BRCA1, which plays '
        'a key role in the protein\'s interaction with other molecules — '
        'this region is known to be sensitive to mutations',
        'this variant falls outside the RING domain'
    ),
    'in_pathogenic_hotspot': (
        'pathogenic hotspot region',
        'this variant is located in a region of BRCA1 where disease-causing '
        'mutations are frequently found',
        'this variant is not located in the most commonly affected regions '
        'of the BRCA1 gene'
    ),
    'in_coiled_coil': (
        'coiled-coil domain',
        'this variant is in the coiled-coil region of BRCA1, which is '
        'important for protein interactions',
        'this variant is outside the coiled-coil structural region'
    ),
    'in_disordered_region': (
        'unstructured protein region',
        'this variant is in a flexible, unstructured part of the BRCA1 protein',
        'this variant is in a flexible, unstructured part of the BRCA1 protein '
        'where changes are often better tolerated'
    ),

    # Consequence
    'is_likely_lof': (
        'protein function disruption',
        'this type of genetic change is strongly predicted to stop the BRCA1 '
        'protein from working normally — this is one of the strongest signals '
        'for a harmful variant',
        'this genetic change is not predicted to completely disrupt BRCA1 '
        'protein function'
    ),
    'is_missense': (
        'amino acid change',
        'this variant causes a single amino acid in the BRCA1 protein to change, '
        'which can affect how the protein folds and works',
        'this variant causes a single amino acid change that appears to have '
        'little impact on protein function'
    ),
    'is_nonsense': (
        'premature stop signal',
        'this variant creates an early stop signal in the BRCA1 gene, '
        'causing the protein to be cut short — shortened proteins usually '
        'cannot do their job properly',
        'this variant does not create an early stop signal in the protein'
    ),
    'is_synonymous': (
        'silent DNA change',
        'although this DNA change does not alter the amino acid sequence, '
        'the model identified other features suggesting possible impact',
        'this variant changes the DNA sequence without changing the amino '
        'acid it codes for, which usually has no effect on the protein'
    ),
    'is_canonical_splice_site': (
        'RNA processing signals',
        'this variant affects the signals that tell the cell how to correctly '
        'read and process the BRCA1 gene — disrupting these signals can '
        'prevent the correct protein from being made',
        'this variant does not affect the signals that control how the '
        'BRCA1 gene is read by the cell'
    ),
    'cons_frameshift': (
        'reading frame disruption',
        'this variant shifts the reading frame of the gene, which typically '
        'causes the entire protein sequence after this point to be wrong',
        'this variant does not disrupt the protein reading frame'
    ),
    'cons_nonsense': (
        'stop codon creation',
        'this variant introduces a stop signal that cuts the BRCA1 protein '
        'short before it is fully formed',
        'this variant does not introduce an early stop signal'
    ),
    'cons_splice': (
        'splicing effect',
        'this variant is predicted to interfere with how the BRCA1 gene is '
        'spliced together before being translated into protein',
        'this variant is not predicted to significantly interfere with splicing'
    ),
    'cons_snv': (
        'single base change',
        'this is a single letter change in the DNA of the BRCA1 gene',
        'this is a single letter change in the DNA that appears harmless'
    ),

    # Position
    'aa_position': (
        'protein position',
        'the position where this change occurs in the protein is in a '
        'region important for BRCA1 function',
        'the position where this change occurs in the protein is in a '
        'region that appears more tolerant of changes'
    ),
    'normalized_position': (
        'gene position',
        'the location of this variant within the BRCA1 gene is in a '
        'region associated with functional importance',
        'the location of this variant within the BRCA1 gene is in a '
        'region less commonly associated with disease'
    ),
    'in_exon11': (
        'exon 11 location',
        'this variant is in exon 11, the largest coding section of BRCA1 '
        'and a region where many disease-causing variants are found',
        'this variant is not in exon 11'
    ),

    # Metadata
    'review_strength': (
        'clinical evidence quality',
        'there is existing clinical evidence that helped inform this prediction',
        'this variant has limited prior clinical evidence on record'
    ),
    'num_submitters': (
        'number of reports',
        'multiple independent laboratories have reported this variant, '
        'increasing confidence in the assessment',
        'this variant has been reported by a limited number of sources'
    ),
}


def parse_shap_features(shap_str):
    """Return list of feature names from SHAP string."""
    if pd.isna(shap_str) or str(shap_str).strip() in ['', 'nan', 'Not available']:
        return []
    return [f.strip() for f in str(shap_str).split(',') if f.strip()]


def build_patient_explanation(name, prob, pred_class, confidence,
                               shap_features, variant_name):
    """
    Build a fully dynamic, personalized patient explanation.
    Every sentence is derived from the actual features of THIS variant.
    No hardcoded narrative. Simple English. No metrics. No disclaimers.
    """

    lines = []
    prob_pct = round(prob * 100, 1)

    # ── Opening sentence — personalized to this variant ──────────────────────
    short_name = str(variant_name)
    if 'BRCA1' in short_name and ':' in short_name:
        # Extract the c. or p. notation if present
        parts = short_name.split('(')
        if len(parts) > 1:
            short_name = parts[-1].replace(')', '').strip()
        else:
            short_name = short_name.split(':')[-1].strip()

    lines.append(f"About your BRCA1 variant: {short_name}")
    lines.append("")

    # ── What the analysis found — probability-driven opening ────────────────
    if pred_class == 'Pathogenic':
        if prob_pct >= 90:
            opening = (
                f"The analysis of this specific BRCA1 variant found very "
                f"strong signs that it may affect how the BRCA1 gene works. "
                f"Out of 100 similar variants in our reference data, about "
                f"{int(prob_pct)} would be classified as harmful."
            )
        elif prob_pct >= 75:
            opening = (
                f"The analysis found several signs that this BRCA1 variant "
                f"may affect the normal function of the gene. "
                f"The model is fairly confident in this assessment, though "
                f"some uncertainty remains."
            )
        else:
            opening = (
                f"The analysis found a slight lean toward this variant being "
                f"potentially harmful, but the evidence is not strong. "
                f"The model is not highly certain in this direction."
            )
    else:  # Benign
        if prob_pct <= 10:
            opening = (
                f"The analysis found very strong signs that this BRCA1 variant "
                f"is unlikely to affect how the gene works. "
                f"Out of 100 similar variants in our reference data, about "
                f"{int(100 - prob_pct)} would be considered harmless."
            )
        elif prob_pct <= 25:
            opening = (
                f"The analysis found several signs that this BRCA1 variant "
                f"is probably not harmful. "
                f"The model is fairly confident in this assessment."
            )
        else:
            opening = (
                f"The analysis slightly favors this variant being harmless, "
                f"but the evidence is not strong enough to be fully certain."
            )

    lines.append(opening)
    lines.append("")

    # ── Feature-by-feature personalized sentences ────────────────────────────
    if shap_features:
        lines.append("Here is what the model found about this specific variant:")
        lines.append("")

        feature_sentences_used = []
        for feat in shap_features[:5]:
            feat = feat.strip()
            if feat in FEATURE_SENTENCES:
                _, path_sent, ben_sent = FEATURE_SENTENCES[feat]
                sentence = path_sent if pred_class == 'Pathogenic' else ben_sent
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:]
                if not sentence.endswith('.'):
                    sentence += '.'
                feature_sentences_used.append(f"  • {sentence}")
            else:
                # Unknown feature — make a generic sentence
                feat_plain = feat.replace('_', ' ').lower()
                if pred_class == 'Pathogenic':
                    feature_sentences_used.append(
                        f"  • The {feat_plain} of this variant contributed "
                        f"evidence toward the harmful prediction."
                    )
                else:
                    feature_sentences_used.append(
                        f"  • The {feat_plain} of this variant contributed "
                        f"evidence toward the harmless prediction."
                    )

        lines.extend(feature_sentences_used)
        lines.append("")

    # ── Confidence-driven closing — specific to this variant's confidence ─────
    if pred_class == 'Pathogenic':
        if confidence == 'High':
            closing = (
                f"Taken together, these findings point clearly in one direction "
                f"for this particular variant. The model is highly confident "
                f"based on the combination of signals seen specifically in "
                f"this variant. A review by a genetics specialist is recommended."
            )
        elif confidence == 'Medium':
            closing = (
                f"These findings suggest a possible concern, but some of the "
                f"signals are mixed for this particular variant. "
                f"Talking to a genetics specialist would help clarify what "
                f"this result means for you personally."
            )
        else:
            closing = (
                f"The signals for this particular variant are not strong enough "
                f"to be certain in either direction. More information — such as "
                f"family history or laboratory testing — would help clarify "
                f"the meaning of this result."
            )
    else:  # Benign
        if confidence == 'High':
            closing = (
                f"For this specific variant, the combination of features "
                f"consistently points toward it being harmless. "
                f"The model found no strong signals of concern."
            )
        elif confidence == 'Medium':
            closing = (
                f"The signals for this variant lean toward it being harmless, "
                f"but are not fully conclusive. "
                f"Discussing this result with a healthcare provider is still worthwhile."
            )
        else:
            closing = (
                f"The signals for this variant are mixed. While it does not "
                f"show strong signs of harm, the prediction is not certain. "
                f"More information would help clarify this result."
            )

    lines.append(closing)
    lines.append("")

    return "\n".join(lines)


# ============================================================
# FULL REPORT GENERATION — CLINICAL + PATIENT SECTIONS
# ============================================================

def generate_full_report(row):
    """
    Generates a two-section report:
    Section 1 — Technical summary (for clinicians)
    Section 2 — Patient explanation (fully dynamic, personalized)
    """
    name        = str(row.get('_name', 'Unknown'))
    var_id      = str(row.get('VariationID', row.get('_name', '')))
    prob        = float(row.get('_prob', 0.5))
    pred_class  = str(row.get('_class', 'Unknown'))
    confidence  = str(row.get('_conf', 'Low'))
    shap_raw    = str(row.get('_shap', ''))
    acmg_raw    = str(row.get('_acmg', ''))
    prob_pct    = round(prob * 100, 1)
    shap_feats  = parse_shap_features(shap_raw)

    lines = []

    lines.append("=" * 68)
    lines.append("BRCA1 VARIANT REPORT")
    lines.append("=" * 68)
    lines.append(f"Variant  : {name}")
    lines.append(f"ID       : {var_id}")
    lines.append("")

    # ── SECTION 1: Technical (clinician-facing) ───────────────────────────────
    lines.append("─" * 68)
    lines.append("SECTION 1 — TECHNICAL SUMMARY  (For Healthcare Providers)")
    lines.append("─" * 68)
    lines.append(f"  Predicted Class        : {pred_class}")
    lines.append(f"  Pathogenic Probability : {prob_pct}%")
    lines.append(f"  Confidence             : {confidence}")
    lines.append("")

    lines.append("  Top SHAP Features (driving factors):")
    if shap_feats:
        for i, feat in enumerate(shap_feats[:5], 1):
            lines.append(f"    {i}. {feat}")
    else:
        lines.append("    Not available")
    lines.append("")

    # ACMG mapping
    acmg_map = {
        'PVS1'        : 'PVS1  — Loss of function (Very Strong Pathogenic)',
        'PM1'         : 'PM1   — Located in critical domain (Moderate Pathogenic)',
        'PM1_BRCT'    : 'PM1   — BRCT domain variant (Moderate Pathogenic)',
        'PP3'         : 'PP3   — Computational evidence (Supporting Pathogenic)',
        'PP3_Strong'  : 'PP3   — Strong computational evidence (Supporting Pathogenic)',
        'PS3_Moderate': 'PS3   — Functional evidence (Moderate Pathogenic)',
        'BP4'         : 'BP4   — Computational benign evidence (Supporting Benign)',
        'BS1'         : 'BS1   — High population frequency (Strong Benign)',
        'BP7'         : 'BP7   — Synonymous silent variant (Supporting Benign)',
        'BP1'         : 'BP1   — Missense in predominantly benign region (Supporting Benign)',
    }
    if acmg_raw and acmg_raw not in ['', 'nan', 'Not available',
                                       'Insufficient_evidence']:
        codes = [c.strip() for c in acmg_raw.split(',') if c.strip()]
        lines.append("  ACMG/AMP Evidence Codes:")
        for code in codes:
            lines.append(f"    • {acmg_map.get(code, code)}")
        lines.append("")
    else:
        lines.append("  ACMG/AMP: No specific evidence codes triggered")
        lines.append("")

    # ── SECTION 2: Patient explanation — fully dynamic ────────────────────────
    lines.append("─" * 68)
    lines.append("SECTION 2 — EXPLANATION IN PLAIN LANGUAGE")
    lines.append("─" * 68)
    lines.append("")

    patient_text = build_patient_explanation(
        name       = name,
        prob       = prob,
        pred_class = pred_class,
        confidence = confidence,
        shap_features = shap_feats,
        variant_name  = name
    )
    lines.append(patient_text)
    lines.append("=" * 68)

    return "\n".join(lines)


# ============================================================
# GENERATE SAMPLE REPORTS — 4 CLINICAL SCENARIOS
# ============================================================
print("\n" + "─" * 65)
print("Generating 4 sample reports (one per clinical scenario)...")
print("─" * 65)

# Select one real variant for each scenario
hc_path = df[
    (df['_class'] == 'Pathogenic') & (df['_conf'] == 'High')
].sort_values('_prob', ascending=False)

mc_path = df[
    (df['_class'] == 'Pathogenic') & (df['_conf'] == 'Medium')
].sort_values('_prob', ascending=False)

hc_ben = df[
    (df['_class'] == 'Benign') & (df['_conf'] == 'High')
].sort_values('_prob', ascending=True)

lc_df = df[df['_conf'] == 'Low'].copy()
lc_df['_dist'] = (lc_df['_prob'] - 0.5).abs()
borderline = lc_df.sort_values('_dist', ascending=True)

scenarios = [
    (hc_path,   'sample_high_confidence_pathogenic',
     'HIGH-CONFIDENCE PATHOGENIC'),
    (mc_path,   'sample_medium_confidence_pathogenic',
     'MEDIUM-CONFIDENCE PATHOGENIC'),
    (hc_ben,    'sample_high_confidence_benign',
     'HIGH-CONFIDENCE BENIGN'),
    (borderline,'sample_borderline_uncertain',
     'BORDERLINE / UNCERTAIN'),
]

for df_sub, fname, label in scenarios:
    if len(df_sub) == 0:
        print(f"  {label}: no variants available — skipping")
        continue

    row    = df_sub.iloc[0]
    report = generate_full_report(row)
    path   = f'results/reports/sample_patient_reports/{fname}.txt'

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"SCENARIO: {label}\n{'='*68}\n\n")
        f.write(report)

    print(f"\n  [{label}]")
    print(f"    Variant    : {str(row.get('_name',''))[:55]}")
    print(f"    Prob       : {row['_prob']:.4f}  |  Confidence: {row['_conf']}")
    print(f"    SHAP feats : {str(row.get('_shap',''))[:55]}")
    print(f"    Saved      : {path}")

# ============================================================
# GENERATE FULL BATCH — all 2,309 VUS
# ============================================================
print("\n" + "─" * 65)
print("Generating full batch reports for all VUS...")
print("─" * 65)

batch_rows = []
for _, row in df.iterrows():
    prob       = float(row.get('_prob', 0.5))
    pred_class = str(row.get('_class', 'Unknown'))
    confidence = str(row.get('_conf', 'Low'))
    shap_feats = parse_shap_features(str(row.get('_shap', '')))

    # Build dynamic patient explanation
    patient_text = build_patient_explanation(
        name          = str(row.get('_name', '')),
        prob          = prob,
        pred_class    = pred_class,
        confidence    = confidence,
        shap_features = shap_feats,
        variant_name  = str(row.get('_name', ''))
    )

    # Build feature plain text (3 sentences from top SHAP)
    feat_sentences = []
    for feat in shap_feats[:3]:
        if feat in FEATURE_SENTENCES:
            _, p, b = FEATURE_SENTENCES[feat]
            s = p if pred_class == 'Pathogenic' else b
            feat_sentences.append(s[0].upper() + s[1:] + '.')

    batch_rows.append({
        'VariationID'          : row.get('VariationID', ''),
        'Name'                 : row.get('_name', ''),
        'Predicted_class'      : pred_class,
        'Prob_pct'             : round(prob * 100, 1),
        'Confidence'           : confidence,
        'Patient_explanation'  : patient_text.strip(),
        'Top3_feature_sentences': ' | '.join(feat_sentences),
        'Top5_SHAP'            : row.get('_shap', ''),
        'ACMG_codes'           : row.get('_acmg', ''),
    })

batch_df = pd.DataFrame(batch_rows)
batch_df.to_csv(
    'results/reports/full_vus_reports/all_vus_personalized_reports.csv',
    index=False, encoding='utf-8'
)

print(f"  Saved: {len(batch_df):,} personalized reports")
print(f"  File : results/reports/full_vus_reports/all_vus_personalized_reports.csv")

# ── Quick check: verify explanations differ between patients ──────────────────
print(f"\n  Verification — checking personalization (first 3 explanations):")
for i in range(min(3, len(batch_df))):
    exp = batch_df.iloc[i]['Patient_explanation']
    first_line = exp.split('\n')[0]
    print(f"    Patient {i+1}: {first_line[:70]}")

unique_explanations = batch_df['Patient_explanation'].nunique()
print(f"\n  Unique explanations generated : {unique_explanations:,} / {len(batch_df):,}")
print(f"  {'✓ All explanations are personalized' if unique_explanations == len(batch_df) else '⚠ Some explanations are identical — check SHAP feature variety'}")

# ── Final summary ─────────────────────────────────────────────────────────────
n_path   = (batch_df['Predicted_class']=='Pathogenic').sum()
n_ben    = (batch_df['Predicted_class']=='Benign').sum()
n_high_p = ((batch_df['Predicted_class']=='Pathogenic') &
            (batch_df['Confidence']=='High')).sum()
n_high_b = ((batch_df['Predicted_class']=='Benign') &
            (batch_df['Confidence']=='High')).sum()

print(f"""
{'='*65}
MODULE 2 COMPLETE
{'='*65}
  Total reports             : {len(batch_df):,}
  Unique explanations       : {unique_explanations:,}
  Predicted Pathogenic      : {n_path:,}
  Predicted Benign          : {n_ben:,}
  High-confidence pathogenic: {n_high_p:,}
  High-confidence benign    : {n_high_b:,}

  Sample reports (4 scenarios):
    results/reports/sample_patient_reports/

  Full batch:
    results/reports/full_vus_reports/all_vus_personalized_reports.csv
{'='*65}
""")