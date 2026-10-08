"""
combine_outputs_figure.py
BRCA-COMPASS — Combined Model Outputs Figure (6 panels, 2x3 grid)
Save to : M:\brca1_pathogenicity\combine_outputs_figure.py
Run     : python combine_outputs_figure.py
Output  : results\figures\manuscript\fig_combined_outputs.png
"""

import os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path("M:/brca1_pathogenicity")
FIG_DIR  = BASE_DIR / "results" / "figures"
MAN_DIR  = FIG_DIR / "manuscript"
OUT_PATH = MAN_DIR / "fig_combined_outputs.png"

PANELS = [
    {
        "label"  : "(a) SHAP Waterfall — Pathogenic Missense Variant",
        "caption": "Top drivers: is_likely_lof ↑  cons_snv ↑  gnomAD_AF ↓",
        "accent" : (41, 128, 185),
        "file"   : FIG_DIR / "shap_waterfall_missense_pathogenic.png",
    },
    {
        "label"  : "(b) SHAP Waterfall — Benign Variant",
        "caption": "Top drivers: gnomAD_AF ↑ (BA1)  is_synonymous ↑ (BP7)",
        "accent" : (41, 128, 185),
        "file"   : FIG_DIR / "shap_waterfall_missense_benign.png",
    },
    {
        "label"  : "(c) SHAP Waterfall — Variant of Uncertain Significance",
        "caption": "Mixed evidence — borderline pathogenicity probability",
        "accent" : (41, 128, 185),
        "file"   : FIG_DIR / "shap_waterfall_missense_uncertain.png",
    },
    {
        "label"  : "(d) VUS Confidence Distribution",
        "caption": "Calibrated pathogenicity probability across 2,309 held-out VUS",
        "accent" : (39, 174, 96),
        "file"   : MAN_DIR / "vus_fig2_confidence_distribution.png",
    },
    {
        "label"  : "(e) VUS Top 50 Ranking",
        "caption": "Top 50 high-risk VUS ranked by pathogenic probability",
        "accent" : (39, 174, 96),
        "file"   : MAN_DIR / "vus_fig3_top50_ranking.png",
    },
    {
        "label"  : "(f) External Validation — SGE Concordance",
        "caption": "BRCA-COMPASS vs Findlay SGE functional scores (AUC = 0.9041)",
        "accent" : (142, 68, 173),
        "file"   : FIG_DIR / "external_validation_sge_v2.png",
    },
]

PANEL_W    = 900
COLS       = 2
LABEL_H    = 38
SUBCAP_H   = 26
GAP_X      = 32
GAP_Y      = 42
PAD        = 50
TITLE_H    = 65
FOOTER_H   = 130
BG         = (255, 255, 255)
TITLE_CLR  = (15,  40,  80)
LABEL_CLR  = (25,  25,  25)
CAP_CLR    = (90,  90,  90)
BORDER_CLR = (205, 210, 215)

MAIN_TITLE = "BRCA-COMPASS: Model Output Visualisations"
FOOTER = (
    "Fig. Y | BRCA-COMPASS model output panels. "
    "(a) SHAP waterfall for a representative pathogenic missense variant: "
    "is_likely_lof and cons_snv drive pathogenic prediction; gnomAD_AF opposes it. "
    "(b) SHAP waterfall for a benign variant: high gnomAD_AF and is_synonymous "
    "generate strongly negative (benign) SHAP values consistent with ACMG BA1 and BP7. "
    "(c) SHAP waterfall for a VUS showing mixed evidence and borderline probability. "
    "(d) Calibrated pathogenicity probability distribution across 2,309 held-out VUS, "
    "illustrating three-tier confidence stratification. "
    "(e) Top 50 high-risk VUS candidates ranked by pathogenic probability for "
    "experimental follow-up prioritisation. "
    "(f) BRCA-COMPASS predictions versus Findlay SGE functional scores (missense AUC = 0.9041), "
    "confirming concordance with biologically derived ground truth."
)


def get_font(size, bold=False):
    paths = (
        ["C:/Windows/Fonts/Arialbd.ttf", "C:/Windows/Fonts/calibrib.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/Arial.ttf", "C:/Windows/Fonts/calibri.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def resize_w(img, w):
    return img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)


def placeholder(w, h, label):
    img  = Image.new("RGB", (w, h), (232, 232, 236))
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, w-3, h-3], outline=(180, 180, 185), width=2)
    draw.text((16, h//2 - 10), f"[Not found] {label}",
              font=get_font(13), fill=(140, 140, 150))
    return img


def wrap(text, font, max_w, draw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines


def build():
    MAN_DIR.mkdir(parents=True, exist_ok=True)
    panels, tallest = [], 0
    for src in PANELS:
        if src["file"].exists():
            img = resize_w(Image.open(src["file"]).convert("RGB"), PANEL_W)
            print(f"  ✓  {src['label'][:55]}")
        else:
            print(f"  ✗  {src['label'][:55]}  ← NOT FOUND: {src['file'].name}")
            img = placeholder(PANEL_W, 500, src["label"])
        panels.append(img)
        tallest = max(tallest, img.height)

    ROWS      = (len(panels) + COLS - 1) // COLS
    row_block = LABEL_H + tallest + SUBCAP_H
    total_w   = PAD*2 + COLS*PANEL_W + (COLS-1)*GAP_X
    total_h   = PAD + TITLE_H + ROWS*row_block + (ROWS-1)*GAP_Y + FOOTER_H + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)
    ft     = get_font(22, bold=True)
    fl     = get_font(14, bold=True)
    fs     = get_font(12)
    ff     = get_font(12)

    # Title
    draw.text((PAD, PAD+10), MAIN_TITLE, font=ft, fill=TITLE_CLR)
    tw = draw.textlength(MAIN_TITLE, font=ft)
    draw.line([(PAD, PAD+36), (PAD+tw, PAD+36)], fill=TITLE_CLR, width=2)

    y0 = PAD + TITLE_H
    for i, (img, src) in enumerate(zip(panels, PANELS)):
        row, col = i // COLS, i % COLS
        x = PAD + col*(PANEL_W + GAP_X)
        y = y0  + row*(row_block + GAP_Y)

        # Colour accent bar on left
        draw.rectangle([x-7, y, x-3, y+LABEL_H+tallest+SUBCAP_H],
                       fill=src["accent"])

        draw.text((x, y+4), src["label"], font=fl, fill=LABEL_CLR)
        iy = y + LABEL_H + (tallest - img.height)//2
        canvas.paste(img, (x, iy))
        draw.rectangle([x-1, iy-1, x+PANEL_W+1, iy+img.height+1],
                       outline=BORDER_CLR, width=1)
        draw.text((x, y+LABEL_H+tallest+4), src["caption"], font=fs, fill=CAP_CLR)

    # Footer
    fy = total_h - FOOTER_H - PAD + 12
    for line in wrap(FOOTER, ff, total_w - PAD*2, draw):
        draw.text((PAD, fy), line, font=ff, fill=(55, 55, 55))
        fy += 17

    canvas.save(str(OUT_PATH), dpi=(300, 300))
    print(f"\n✅  Saved → {OUT_PATH}  ({canvas.width}×{canvas.height} px)")


if __name__ == "__main__":
    print("="*60)
    print("BRCA-COMPASS | Combined Outputs Figure")
    print("="*60)
    build()