import pandas as pd
import re

df = pd.read_csv('data/processed/brca1_features_scored.csv', low_memory=False)

print("Checking c.5266dupC aliases in reference dataset:")
print("=" * 65)

queries = [
    'c.5266dupC',
    'c.5266dup',
    '5382insC',
    'rs80357906',
    'c.5266',
    '5266',
]

for q in queries:
    match = df[df['Name'].astype(str).str.contains(
        re.escape(q), case=False, na=False
    )]
    if len(match) > 0:
        found = match['Name'].values[0][:65]
        print(f"  FOUND  [{q}]  -->  {found}")
    else:
        print(f"  MISS   [{q}]  -->  not in dataset")

print()
print("All Name entries containing '5266':")
subset = df[df['Name'].astype(str).str.contains('5266', na=False)]
for name in subset['Name'].values[:10]:
    print(f"  {name}")

print()
print("All Name entries containing 'dup':")
subset = df[df['Name'].astype(str).str.contains('dupC', na=False)]
print(f"  Total dupC variants: {len(subset)}")
for name in subset['Name'].values[:5]:
    print(f"  {name}")