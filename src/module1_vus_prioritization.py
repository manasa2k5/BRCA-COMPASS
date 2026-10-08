# ============================================================
# MODULE 1: VUS PRIORITIZATION FRAMEWORK
# File: src/module1_vus_prioritization.py
#
# NO retraining — uses existing vus_predictions.csv
# Input : results/vus/vus_predictions.csv
# Output: ranked lists + publication figures
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results/vus',              exist_ok=True)
os.makedirs('results/figures/manuscript', exist_ok=True)

print("=" * 65)
print("MODULE 1: VUS PRIORITIZATION FRAMEWORK")
print("=" * 65)

# ── Load VUS predictions ──────────────────────────────────────────────────────
df = pd.read_csv('results/vus/vus_predictions.csv')
print(f"\n  Loaded VUS predictions : {len(df):,} variants")
print(f"  Columns available      : {list(df.columns)}")

# ── Standardize column names ──────────────────────────────────────────────────
# Handle different column naming from different pipeline runs
col_map = {}
for col in df.columns:
    cl = col.lower()
    if 'ensemble_prob' in cl or 'ensemble prob' in cl:
        col_map[col] = 'Ensemble_prob'
    elif 'cal_prob' in cl or 'calibrated' in cl:
        col_map[col] = 'Cal_prob'
    elif 'base_prob' in cl:
        col_map[col] = 'Base_prob'
    elif 'predicted_class' in cl or 'predicted class' in cl:
        col_map[col] = 'Predicted_class'
    elif 'confidence' in cl and 'final' not in cl:
        col_map[col] = 'Confidence'
    elif 'final_confidence' in cl:
        col_map[col] = 'Confidence'
    elif 'top5_shap' in cl or 'top5 shap' in cl or 'shap' in cl:
        col_map[col] = 'Top5_SHAP'
    elif 'acmg' in cl:
        col_map[col] = 'ACMG_codes'
    elif 'name' == cl:
        col_map[col] = 'Name'
    elif 'variationid' in cl:
        col_map[col] = 'VariationID'

df = df.rename(columns=col_map)
print(f"\n  Columns after standardization: {list(df.columns)}")

# ── Ensure required columns exist ────────────────────────────────────────────
# Primary probability column
if 'Ensemble_prob' not in df.columns:
    if 'Cal_prob' in df.columns:
        df['Ensemble_prob'] = df['Cal_prob']
    elif 'Base_prob' in df.columns:
        df['Ensemble_prob'] = df['Base_prob']
    else:
        # Try to find any probability column
        prob_cols = [c for c in df.columns if 'prob' in c.lower()]
        if prob_cols:
            df['Ensemble_prob'] = df[prob_cols[0]]
        else:
            raise ValueError(f"No probability column found. Columns: {list(df.columns)}")

df['Ensemble_prob'] = pd.to_numeric(df['Ensemble_prob'], errors='coerce')

# Predicted class
if 'Predicted_class' not in df.columns:
    df['Predicted_class'] = (df['Ensemble_prob'] >= 0.5).map(
        {True: 'Pathogenic', False: 'Benign'}
    )

# Confidence
if 'Confidence' not in df.columns:
    def assign_confidence(prob):
        dist = abs(prob - 0.5)
        if dist >= 0.35:
            return 'High'
        elif dist >= 0.15:
            return 'Medium'
        else:
            return 'Low'
    df['Confidence'] = df['Ensemble_prob'].apply(assign_confidence)

# SHAP features
if 'Top5_SHAP' not in df.columns:
    df['Top5_SHAP'] = 'Not available'

# ACMG codes
if 'ACMG_codes' not in df.columns:
    df['ACMG_codes'] = 'Not available'

# Name
if 'Name' not in df.columns:
    df['Name'] = df.get('VariationID', pd.Series(range(len(df)))).astype(str)

print(f"\n  Class distribution:")
print(f"    Predicted Pathogenic : {(df['Predicted_class']=='Pathogenic').sum():,}")
print(f"    Predicted Benign     : {(df['Predicted_class']=='Benign').sum():,}")
print(f"\n  Confidence distribution:")
for tier in ['High','Medium','Low']:
    n   = (df['Confidence']==tier).sum()
    pct = n/len(df)*100
    print(f"    {tier:<8} : {n:,} ({pct:.1f}%)")

# ============================================================
# SECTION 1: RANKING AND FILTERING
# ============================================================
print("\n" + "─" * 65)
print("SECTION 1: Generating Ranked Lists")
print("─" * 65)

# ── Confidence numeric score ──────────────────────────────────────────────────
conf_rank_map = {'High': 3, 'Medium': 2, 'Low': 1}
df['Confidence_rank'] = df['Confidence'].map(conf_rank_map).fillna(1)

# ── A. Top 50 Highest-Risk VUS (Pathogenic) ───────────────────────────────────
pathogenic_vus = df[df['Predicted_class'] == 'Pathogenic'].copy()
pathogenic_vus = pathogenic_vus.sort_values(
    ['Confidence_rank', 'Ensemble_prob'],
    ascending=[False, False]
).reset_index(drop=True)
pathogenic_vus['Rank'] = pathogenic_vus.index + 1

top50_path = pathogenic_vus.head(50)[
    ['Rank','Name','Predicted_class','Ensemble_prob',
     'Confidence','Top5_SHAP','ACMG_codes']
].copy()
top50_path.columns = [
    'Rank','Variant_Name','Predicted_Class','Pathogenic_Probability',
    'Confidence','Top5_SHAP_Features','ACMG_Evidence_Codes'
]
top50_path['Pathogenic_Probability'] = top50_path['Pathogenic_Probability'].round(4)

top50_path.to_csv('results/vus/top50_high_risk_vus.csv', index=False)
print(f"\n  Top 50 High-Risk VUS saved")
print(f"    High confidence  : {(top50_path['Confidence']=='High').sum()}")
print(f"    Medium confidence: {(top50_path['Confidence']=='Medium').sum()}")
print(f"    Low confidence   : {(top50_path['Confidence']=='Low').sum()}")
print(f"    Prob range       : {top50_path['Pathogenic_Probability'].min():.4f} – "
      f"{top50_path['Pathogenic_Probability'].max():.4f}")

# ── B. Top 100 Highest-Risk VUS ───────────────────────────────────────────────
top100_path = pathogenic_vus.head(100)[
    ['Rank','Name','Predicted_class','Ensemble_prob',
     'Confidence','Top5_SHAP','ACMG_codes']
].copy()
top100_path.columns = [
    'Rank','Variant_Name','Predicted_Class','Pathogenic_Probability',
    'Confidence','Top5_SHAP_Features','ACMG_Evidence_Codes'
]
top100_path['Pathogenic_Probability'] = top100_path['Pathogenic_Probability'].round(4)
top100_path.to_csv('results/vus/top100_high_risk_vus.csv', index=False)
print(f"\n  Top 100 High-Risk VUS saved")

# ── C. Top 50 High-Confidence Benign VUS ──────────────────────────────────────
benign_vus = df[df['Predicted_class'] == 'Benign'].copy()
benign_vus = benign_vus.sort_values(
    ['Confidence_rank', 'Ensemble_prob'],
    ascending=[False, True]
).reset_index(drop=True)
benign_vus['Rank'] = benign_vus.index + 1

top50_benign = benign_vus.head(50)[
    ['Rank','Name','Predicted_class','Ensemble_prob',
     'Confidence','Top5_SHAP','ACMG_codes']
].copy()
top50_benign.columns = [
    'Rank','Variant_Name','Predicted_Class','Pathogenic_Probability',
    'Confidence','Top5_SHAP_Features','ACMG_Evidence_Codes'
]
top50_benign['Pathogenic_Probability'] = top50_benign['Pathogenic_Probability'].round(4)
top50_benign.to_csv('results/vus/top50_high_confidence_benign.csv', index=False)
print(f"\n  Top 50 High-Confidence Benign VUS saved")
print(f"    Prob range : {top50_benign['Pathogenic_Probability'].min():.4f} – "
      f"{top50_benign['Pathogenic_Probability'].max():.4f}")

# ============================================================
# SECTION 2: SUMMARY TABLE
# ============================================================
print("\n" + "─" * 65)
print("SECTION 2: Summary Table")
print("─" * 65)

summary_rows = []
for pred_class in ['Pathogenic', 'Benign']:
    for conf_tier in ['High', 'Medium', 'Low']:
        mask  = (
            (df['Predicted_class'] == pred_class) &
            (df['Confidence'] == conf_tier)
        )
        n     = mask.sum()
        pct   = n / len(df) * 100
        probs = df.loc[mask, 'Ensemble_prob']
        category = f"{conf_tier}-Risk {pred_class}" if pred_class == 'Pathogenic' \
                   else f"{conf_tier}-Confidence {pred_class}"
        summary_rows.append({
            'Category'         : category,
            'Count'            : n,
            'Percentage'       : round(pct, 1),
            'Mean_Probability' : round(probs.mean(), 4) if n > 0 else 0,
            'Min_Probability'  : round(probs.min(), 4)  if n > 0 else 0,
            'Max_Probability'  : round(probs.max(), 4)  if n > 0 else 0,
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/vus/vus_summary_table.csv', index=False)

print(f"\n  {'Category':<35} {'Count':>6} {'%':>6} {'Mean Prob':>10}")
print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*10}")
for _, row in summary_df.iterrows():
    print(f"  {row['Category']:<35} {row['Count']:>6} "
          f"{row['Percentage']:>5.1f}% {row['Mean_Probability']:>10.4f}")

# ============================================================
# SECTION 3: PUBLICATION FIGURES
# ============================================================
print("\n" + "─" * 65)
print("SECTION 3: Publication Figures")
print("─" * 65)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
})

COLORS = {
    'High'   : '#E24B4A',
    'Medium' : '#EF9F27',
    'Low'    : '#888780',
    'Benign' : '#378ADD',
}

# ── Figure A: VUS Probability Distribution ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    'VUS Pathogenicity Probability Distribution\n'
    f'n = {len(df):,} Variants of Uncertain Significance',
    fontsize=13, fontweight='bold', y=1.02
)

ax = axes[0]
ax.hist(df[df['Predicted_class']=='Pathogenic']['Ensemble_prob'],
        bins=35, alpha=0.75, color='#E24B4A',
        label=f'Predicted Pathogenic (n={(df["Predicted_class"]=="Pathogenic").sum():,})',
        edgecolor='white')
ax.hist(df[df['Predicted_class']=='Benign']['Ensemble_prob'],
        bins=35, alpha=0.75, color='#378ADD',
        label=f'Predicted Benign (n={(df["Predicted_class"]=="Benign").sum():,})',
        edgecolor='white')
ax.axvline(0.5, color='black', linestyle='--', lw=2,
           label='Decision boundary (0.5)')
ax.set_xlabel('Pathogenic Probability Score')
ax.set_ylabel('Number of VUS')
ax.set_title('(A) Full VUS Probability Distribution', fontweight='bold')
ax.legend(fontsize=9)

ax = axes[1]
colors_conf = [COLORS['High'], COLORS['Medium'], COLORS['Low']]
for pred, ls, lw in [('Pathogenic','-',2.0), ('Benign','--',1.8)]:
    sub = df[df['Predicted_class']==pred]['Ensemble_prob']
    ax.hist(sub, bins=30, alpha=0.55,
            color='#E24B4A' if pred=='Pathogenic' else '#378ADD',
            label=pred, edgecolor='white', linewidth=0.5)
ax.axvline(0.5, color='black', linestyle='--', lw=1.5)
ax.axvspan(0.0, 0.15, alpha=0.08, color='#378ADD', label='High-conf benign zone')
ax.axvspan(0.85, 1.0, alpha=0.08, color='#E24B4A', label='High-conf pathogenic zone')
ax.set_xlabel('Pathogenic Probability Score')
ax.set_ylabel('Number of VUS')
ax.set_title('(B) Distribution with Confidence Zones', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/figures/manuscript/vus_fig1_probability_distribution.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vus_fig1_probability_distribution.png")

# ── Figure B: Confidence Distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    'VUS Confidence Distribution\n'
    f'n = {len(df):,} Variants of Uncertain Significance',
    fontsize=13, fontweight='bold', y=1.02
)

ax = axes[0]
conf_counts = df.groupby(['Predicted_class','Confidence']).size().unstack(fill_value=0)
conf_order  = ['High','Medium','Low']
conf_counts = conf_counts.reindex(columns=conf_order, fill_value=0)
conf_counts.plot(
    kind='bar', ax=ax,
    color=[COLORS['High'], COLORS['Medium'], COLORS['Low']],
    edgecolor='white', alpha=0.88
)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Number of VUS')
ax.set_title('(A) Confidence by Predicted Class', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Confidence', fontsize=9)
for bar in ax.patches:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 2,
                f'{int(bar.get_height())}',
                ha='center', fontsize=9)

ax = axes[1]
# Pie chart
labels_pie = []
sizes_pie  = []
colors_pie = []
pie_color_map = {
    'High-Risk Pathogenic'       : '#C0392B',
    'Medium-Risk Pathogenic'     : '#E67E22',
    'Low-Risk Pathogenic'        : '#F39C12',
    'High-Confidence Benign'     : '#1A5276',
    'Medium-Confidence Benign'   : '#2980B9',
    'Low-Confidence Benign'      : '#85C1E9',
}
for _, row in summary_df.iterrows():
    if row['Count'] > 0:
        labels_pie.append(f"{row['Category']}\n(n={row['Count']})")
        sizes_pie.append(row['Count'])
        colors_pie.append(pie_color_map.get(row['Category'], '#888780'))

wedges, texts, autotexts = ax.pie(
    sizes_pie, labels=None, colors=colors_pie,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8)
ax.legend(wedges, labels_pie, loc='center left',
          bbox_to_anchor=(0.95, 0.5), fontsize=8)
ax.set_title('(B) VUS Category Breakdown', fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/manuscript/vus_fig2_confidence_distribution.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vus_fig2_confidence_distribution.png")

# ── Figure C: Top 50 High-Risk VUS Ranking Plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 14))

top50 = pathogenic_vus.head(50).copy()
top50 = top50.sort_values('Ensemble_prob', ascending=True)

conf_colors_map = {'High': '#C0392B', 'Medium': '#E67E22', 'Low': '#F39C12'}
bar_colors = top50['Confidence'].map(conf_colors_map).fillna('#888780')

bars = ax.barh(
    range(len(top50)),
    top50['Ensemble_prob'],
    color=bar_colors,
    alpha=0.88,
    edgecolor='white',
    linewidth=0.5,
    height=0.75
)
ax.axvline(0.85, color='#C0392B', linestyle='--', lw=1.5,
           alpha=0.7, label='High confidence threshold (0.85)')
ax.axvline(0.65, color='#E67E22', linestyle='--', lw=1.2,
           alpha=0.7, label='Medium confidence threshold (0.65)')

# Y-axis labels — shortened variant names
y_labels = []
for _, row in top50.iterrows():
    name = str(row.get('Name', ''))
    if len(name) > 45:
        name = name[:42] + '...'
    y_labels.append(name)

ax.set_yticks(range(len(top50)))
ax.set_yticklabels(y_labels, fontsize=7)

# Annotate probability values
for i, (bar, (_, row)) in enumerate(zip(bars, top50.iterrows())):
    ax.text(bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{row["Ensemble_prob"]:.3f}',
            va='center', fontsize=7.5,
            color=conf_colors_map.get(row['Confidence'], '#333'))

ax.set_xlabel('Pathogenic Probability Score', fontsize=11)
ax.set_title(
    'Top 50 Highest-Risk BRCA1 VUS\n'
    'Ranked by Pathogenic Probability and Confidence',
    fontsize=13, fontweight='bold'
)
ax.set_xlim([0.45, 1.08])

legend_patches = [
    mpatches.Patch(color='#C0392B', label=f'High confidence ({(top50["Confidence"]=="High").sum()})'),
    mpatches.Patch(color='#E67E22', label=f'Medium confidence ({(top50["Confidence"]=="Medium").sum()})'),
    mpatches.Patch(color='#F39C12', label=f'Low confidence ({(top50["Confidence"]=="Low").sum()})'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('results/figures/manuscript/vus_fig3_top50_ranking.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vus_fig3_top50_ranking.png")

# ── Figure D: High-Risk VUS Leaderboard Table ─────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

top15 = pathogenic_vus.head(15).copy()
top15['Ensemble_prob'] = top15['Ensemble_prob'].round(4)

# Shorten names and SHAP for table
def shorten(s, n=40):
    s = str(s)
    return s[:n] + '...' if len(s) > n else s

table_data = []
for _, row in top15.iterrows():
    table_data.append([
        str(int(row.get('Rank', 0))),
        shorten(row.get('Name', ''), 38),
        f"{row['Ensemble_prob']:.4f}",
        row['Confidence'],
        shorten(row.get('Top5_SHAP', ''), 35),
        shorten(row.get('ACMG_codes', ''), 20),
    ])

col_labels = ['Rank','Variant Name','P(Path)','Confidence',
              'Top SHAP Features','ACMG Codes']

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='left',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)

# Header style
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C2C2A')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_height(0.10)

# Row styles
for i in range(1, len(table_data)+1):
    conf = table_data[i-1][3]
    row_color = (
        '#FADBD8' if conf == 'High'   else
        '#FDEBD0' if conf == 'Medium' else
        '#FDFEFE'
    )
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(row_color)
        table[i, j].set_height(0.08)

ax.set_title(
    'Table: Top 15 Highest-Risk BRCA1 VUS — Candidate Reclassification List\n'
    '(Color: Red = High confidence | Orange = Medium | White = Low)',
    fontsize=11, fontweight='bold', pad=15
)

plt.tight_layout()
plt.savefig('results/figures/manuscript/vus_fig4_leaderboard_table.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vus_fig4_leaderboard_table.png")

# ── Figure E: Summary Bar Chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

categories = summary_df['Category'].tolist()
counts     = summary_df['Count'].tolist()
bar_colors = [
    '#C0392B','#E67E22','#F39C12',
    '#1A5276','#2980B9','#85C1E9'
]

bars = ax.bar(range(len(categories)), counts,
              color=bar_colors, edgecolor='white',
              linewidth=0.5, alpha=0.88)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{count}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Number of VUS', fontsize=11)
ax.set_title(
    'VUS Classification Summary\n'
    f'Total VUS Analyzed: {len(df):,}',
    fontsize=13, fontweight='bold'
)
ax.set_ylim([0, max(counts) * 1.15])

plt.tight_layout()
plt.savefig('results/figures/manuscript/vus_fig5_summary_bar.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: vus_fig5_summary_bar.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
n_path     = (df['Predicted_class']=='Pathogenic').sum()
n_ben      = (df['Predicted_class']=='Benign').sum()
n_high_p   = ((df['Predicted_class']=='Pathogenic') & (df['Confidence']=='High')).sum()
n_high_b   = ((df['Predicted_class']=='Benign')     & (df['Confidence']=='High')).sum()

print(f"""
{'='*65}
MODULE 1 COMPLETE — VUS PRIORITIZATION SUMMARY
{'='*65}

  Total VUS analyzed          : {len(df):,}
  Predicted Pathogenic        : {n_path:,} ({n_path/len(df)*100:.1f}%)
  Predicted Benign            : {n_ben:,} ({n_ben/len(df)*100:.1f}%)
  High-confidence pathogenic  : {n_high_p:,}
  High-confidence benign      : {n_high_b:,}

  Output files:
    results/vus/top50_high_risk_vus.csv
    results/vus/top100_high_risk_vus.csv
    results/vus/top50_high_confidence_benign.csv
    results/vus/vus_summary_table.csv
    results/figures/manuscript/vus_fig1_probability_distribution.png
    results/figures/manuscript/vus_fig2_confidence_distribution.png
    results/figures/manuscript/vus_fig3_top50_ranking.png
    results/figures/manuscript/vus_fig4_leaderboard_table.png
    results/figures/manuscript/vus_fig5_summary_bar.png
{'='*65}
""")