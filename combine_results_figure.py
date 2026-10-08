"""
combine_results_figure.py
BRCA-COMPASS — Combined Results Figure (6 panels, 3x2 grid)
Save to : M:\brca1_pathogenicity\combine_results_figure.py
Run     : python combine_results_figure.py
Output  : results\figures\manuscript\fig_combined_results.png
"""

import os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path("M:/brca1_pathogenicity")
FIG_DIR  = BASE_DIR / "results" / "figures"
MAN_DIR  = FIG_DIR / "manuscript"
OUT_PATH = MAN_DIR / "fig_combined_results.png"

PANELS = [
    {
        "label"  : "(a) ROC Curves — Internal & External Validation",
        "caption": "Internal AUC = 0.9987  |  External SGE AUC = 0.9041  |  CV = 0.9994 ± 0.0003",
        "file"   : MAN_DIR / "fig1_roc_curves.png",
    },
    {
        "label"  : "(b) Precision-Recall Curve",
        "caption": "PR-AUC = 0.9992 on held-out test set (n = 1,202)",
        "file"   : MAN_DIR / "fig2_pr_curves.png",
    },
    {
        "label"  : "(c) Confusion Matrix",
        "caption": "Threshold = 0.638  |  F1 = 0.9904  |  MCC = 0.9775",
        "file"   : MAN_DIR / "fig3_confusion_matrices.png",
    },
    {
        "label"  : "(d) Calibration Curve",
        "caption": "Brier Score = 0.0075  |  ECE = 0.0102 (post-calibration)",
        "file"   : MAN_DIR / "fig5_calibration.png",
    },
    {
        "label"  : "(e) Ablation Study — Metrics by Feature Set",
        "caption": "AUC across Full / No-LOF / dbNSFP-only / ClinVar-only / Conservative",
        "file"   : MAN_DIR / "fig4_metrics_bar.png",
    },
    {
        "label"  : "(f) VUS Tier Distribution",
        "caption": "83 high-risk pathogenic  |  651 high-confidence benign  |  1,575 uncertain",
        "file"   : MAN_DIR / "vus_fig5_summary_bar.png",
    },
]

PANEL_W     = 900
COLS        = 3
LABEL_H     = 38
SUBCAP_H    = 26
GAP_X       = 30
GAP_Y       = 40
PAD         = 50
TITLE_H     = 65
FOOTER_H    = 110
BG          = (255, 255, 255)
TITLE_CLR   = (15,  40,  80)
LABEL_CLR   = (25,  25,  25)
CAP_CLR     = (90,  90,  90)
BORDER_CLR  = (205, 210, 215)

MAIN_TITLE = "BRCA-COMPASS: Results Summary"
FOOTER = (
    "Fig. X | BRCA-COMPASS combined results. "
    "(a) ROC curves: internal test AUC = 0.9987, external SGE missense AUC = 0.9041, "
    "five-fold CV AUC = 0.9994 ± 0.0003. "
    "(b) Precision-recall curve, PR-AUC = 0.9992. "
    "(c) Confusion matrix at optimised threshold 0.638; F1 = 0.9904, MCC = 0.9775. "
    "(d) Calibration reliability diagram post-isotonic regression; Brier = 0.0075. "
    "(e) Ablation study: test AUC across full model and four ablated feature sets. "
    "(f) VUS prioritisation tier distribution across 2,309 held-out variants."
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
            print(f"  ✓  {src['label'][:50]}")
        else:
            print(f"  ✗  {src['label'][:50]}  ← NOT FOUND: {src['file'].name}")
            img = placeholder(PANEL_W, 480, src["label"])
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
    print("BRCA-COMPASS | Combined Results Figure")
    print("="*60)
    build()