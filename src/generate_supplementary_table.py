# =============================================================================
# BRCA-COMPASS: Supplementary Table S1 Generator
# Generates a professional Excel workbook from held-out VUS predictions.
# Save as: src/generate_supplementary_table.py
# Run from project root: python src/generate_supplementary_table.py
# =============================================================================

import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter

# =============================================================================
# 1. FILE PATHS
# =============================================================================

INPUT_CSV   = os.path.join("results", "vus", "vus_predictions.csv")
OUTPUT_DIR  = os.path.join("results", "supplementary")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Supplementary_Table_S1.xlsx")

# =============================================================================
# 2. TIER ASSIGNMENT THRESHOLDS
# These match exactly the thresholds reported in the manuscript.
# Tier assignment is based on Cal_prob (calibrated probability).
# =============================================================================

def assign_tier(prob):
    """
    Assign a six-tier prioritization label based on calibrated probability.
    Thresholds match the manuscript exactly.

    Tier 1: High-risk pathogenic         (Cal_prob >= 0.85)
    Tier 2: Medium-risk pathogenic       (0.65 <= Cal_prob < 0.85)
    Tier 3: Low-risk pathogenic          (0.50 <= Cal_prob < 0.65)
    Tier 4: Low-confidence benign        (0.35 <= Cal_prob < 0.50)
    Tier 5: Medium-confidence benign     (0.15 <= Cal_prob < 0.35)
    Tier 6: High-confidence benign       (Cal_prob < 0.15)
    """
    if prob >= 0.85:
        return "Tier 1 - High-risk pathogenic"
    elif prob >= 0.65:
        return "Tier 2 - Medium-risk pathogenic"
    elif prob >= 0.50:
        return "Tier 3 - Low-risk pathogenic"
    elif prob >= 0.35:
        return "Tier 4 - Low-confidence benign"
    elif prob >= 0.15:
        return "Tier 5 - Medium-confidence benign"
    else:
        return "Tier 6 - High-confidence benign"


# =============================================================================
# 3. COLUMN ORDER
# Final column order for all data sheets.
# =============================================================================

COLUMN_ORDER = [
    "VariationID",
    "Name",
    "Cal_prob",
    "Base_prob",
    "Ensemble_prob",
    "Tier",
    "Predicted_class",
    "Confidence",
    "ACMG_codes",
    "Top5_SHAP",
]

# =============================================================================
# 4. STYLE DEFINITIONS
# =============================================================================

# Header row: dark blue background, white bold text
HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT  = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
HEADER_ALIGN = Alignment(
    horizontal="center", vertical="center", wrap_text=True
)

# Normal body text
NORMAL_FONT  = Font(size=10, name="Calibri")

# Alternating row fill for readability
ALT_FILL = PatternFill("solid", fgColor="EBF3FB")

# Thin border for header
THIN = Side(border_style="thin", color="1F4E79")
HEADER_BORDER = Border(bottom=THIN)

# Numeric columns (center-aligned)
NUMERIC_COLS = {"Cal_prob", "Base_prob", "Ensemble_prob"}

# Text columns (left-aligned)
TEXT_COLS = {
    "VariationID", "Name", "Tier", "Predicted_class",
    "Confidence", "ACMG_codes", "Top5_SHAP"
}


# =============================================================================
# 5. HELPER: Write a DataFrame to a worksheet with formatting
# =============================================================================

def write_sheet(ws, df):
    """
    Write a DataFrame to an openpyxl worksheet with professional formatting.
    Applies header styling, alternating row fills, column alignment,
    auto-width, freeze pane, and autofilter.
    """
    # Write header row
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = HEADER_ALIGN
        cell.border    = HEADER_BORDER

    # Write data rows
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        use_alt = (row_idx % 2 == 0)

        for col_idx, col_name in enumerate(df.columns, start=1):
            # Safe attribute access — replace spaces with underscores
            # for itertuples compatibility
            attr = col_name.replace(" ", "_").replace("-", "_")
            val  = getattr(row, attr, None)
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.font = NORMAL_FONT

            # Alternating fill
            if use_alt:
                cell.fill = ALT_FILL

            # Alignment and number format by column type
            if col_name in NUMERIC_COLS:
                cell.alignment = Alignment(
                    horizontal="center", vertical="top"
                )
                cell.number_format = "0.0000"
            else:
                cell.alignment = Alignment(
                    horizontal="left", vertical="top", wrap_text=True
                )

    # Auto-adjust column widths (capped at 80 characters)
    for col_idx, col_name in enumerate(df.columns, start=1):
        col_letter = get_column_letter(col_idx)
        max_len    = len(str(col_name))
        sample     = df[col_name].astype(str).head(500)
        for val in sample:
            measured = min(len(val), 80)
            max_len  = max(max_len, measured)
        ws.column_dimensions[col_letter].width = min(max_len + 4, 80)

    # Freeze first row
    ws.freeze_panes = "A2"

    # Enable autofilter across all columns
    ws.auto_filter.ref = ws.dimensions

    # Set header row height
    ws.row_dimensions[1].height = 30


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():

    # ── Create output directory if it does not exist ─────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory confirmed: {OUTPUT_DIR}")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nReading: {INPUT_CSV}")
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            "Please ensure results/vus/vus_predictions.csv exists "
            "in your project root."
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Total rows loaded: {len(df):,}")
    print(f"Columns found   : {list(df.columns)}")

    # ── Validate required columns ─────────────────────────────────────────────
    required = {
        "VariationID", "Name", "Cal_prob", "Base_prob",
        "Ensemble_prob", "Predicted_class", "Confidence",
        "Top5_SHAP", "ACMG_codes"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"The following required columns are missing from the CSV: "
            f"{missing}\nFound columns: {list(df.columns)}"
        )

    # ── Normalize Predicted_class if needed ───────────────────────────────────
    # Handles cases where values may be 0/1 or lowercase
    mapping = {
        "1": "Pathogenic", "0": "Benign",
         1 : "Pathogenic",  0 : "Benign",
        "pathogenic": "Pathogenic", "benign": "Benign"
    }
    df["Predicted_class"] = (
        df["Predicted_class"]
        .map(lambda x: mapping.get(x, mapping.get(str(x).lower(), x)))
    )

    # ── Generate Tier column ──────────────────────────────────────────────────
    print("\nAssigning tiers using manuscript probability thresholds...")
    df["Cal_prob"] = pd.to_numeric(df["Cal_prob"], errors="coerce")
    df["Tier"]     = df["Cal_prob"].apply(assign_tier)

    # ── Enforce column order ──────────────────────────────────────────────────
    # Append any extra columns that exist beyond the defined set
    extra_cols = [c for c in df.columns if c not in COLUMN_ORDER]
    final_cols = COLUMN_ORDER + extra_cols
    df = df[final_cols]

    # ── Build subsets ─────────────────────────────────────────────────────────

    # Sheet 1: Predicted Pathogenic — sorted by Cal_prob descending
    df_path = (
        df[df["Predicted_class"] == "Pathogenic"]
        .sort_values("Cal_prob", ascending=False)
        .reset_index(drop=True)
    )

    # Sheet 2: Predicted Benign — sorted by Cal_prob ascending
    df_benign = (
        df[df["Predicted_class"] == "Benign"]
        .sort_values("Cal_prob", ascending=True)
        .reset_index(drop=True)
    )

    # Sheet 3: Complete VUS — all 2,309 rows, original order
    df_all = df.reset_index(drop=True)

    # Sheet 4: Tier 1 only — sorted by Cal_prob descending
    df_tier1 = (
        df[df["Tier"].str.startswith("Tier 1")]
        .sort_values("Cal_prob", ascending=False)
        .reset_index(drop=True)
    )

    # Sheet 5: Tier 6 only — sorted by Cal_prob ascending
    df_tier6 = (
        df[df["Tier"].str.startswith("Tier 6")]
        .sort_values("Cal_prob", ascending=True)
        .reset_index(drop=True)
    )

    # ── Print tier distribution ───────────────────────────────────────────────
    tier_counts = {
        "pathogenic" : len(df_path),
        "benign"     : len(df_benign),
        "tier1"      : len(df_tier1),
        "tier2"      : len(df[df["Tier"].str.startswith("Tier 2")]),
        "tier3"      : len(df[df["Tier"].str.startswith("Tier 3")]),
        "tier4"      : len(df[df["Tier"].str.startswith("Tier 4")]),
        "tier5"      : len(df[df["Tier"].str.startswith("Tier 5")]),
        "tier6"      : len(df_tier6),
    }

    print("\nTier distribution:")
    for key, count in tier_counts.items():
        print(f"  {key:15s}: {count:,}")

    # ── Build workbook ────────────────────────────────────────────────────────
    print("\nBuilding workbook...")
    wb = Workbook()

    # Remove the default blank sheet created by openpyxl
    wb.remove(wb.active)

    # Define the five data sheets
    data_sheets = [
        ("VUS_Pathogenic", df_path,   "Predicted Pathogenic VUS"),
        ("VUS_Benign",     df_benign, "Predicted Benign VUS"),
        ("Complete_VUS",   df_all,    "All VUS"),
        ("Tier1_HighRisk", df_tier1,  "Tier 1 High-Risk VUS"),
        ("Tier6_Benign",   df_tier6,  "Tier 6 Benign VUS"),
    ]

    for sheet_name, data_df, label in data_sheets:
        ws = wb.create_sheet(sheet_name)
        write_sheet(ws, data_df)
        print(
            f"  {label:35s}: {len(data_df):,} rows "
            f"written to '{sheet_name}'"
        )

    # ── Save workbook ─────────────────────────────────────────────────────────
    wb.save(OUTPUT_FILE)
    print(f"\nWorkbook saved successfully.")
    print(f"Output path : {OUTPUT_FILE}")
    print(f"Total sheets: {len(wb.sheetnames)}")
    for name in wb.sheetnames:
        print(f"  - {name}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()