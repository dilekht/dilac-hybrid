"""
Generate two additional figures for the revised paper:
  fig7_architecture.png  — the GlossBERT WSD pipeline (schematic)
  fig8_polysemy_band.png — accuracy by sense-count band (supports error analysis)

Run: python scripts/make_extra_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / 'results' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

BLUE='#2c6fbb'; GREEN='#2e8b57'; GRAY='#8a8f98'; ORANGE='#d9822b'; LIGHT='#eaf1fb'

# ── Figure 7: Architecture / pipeline ────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(cx, cy, w, h, text, fc, tc='white', fs=9.5):
        ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.5",
            fc=fc, ec='none'))
        ax.text(cx, cy, text, ha='center', va='center',
                color=tc, fontsize=fs, fontweight='bold')

    def varrow(x, y1, y2, color=GRAY, label=None):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
            arrowstyle='-|>', mutation_scale=16, color=color, lw=1.8))
        if label:
            ax.text(x+2.5, (y1+y2)/2, label, ha='left', va='center',
                    fontsize=8, color=GRAY, style='italic')

    cx = 50  # single central column → unambiguous top-to-bottom flow
    # 1. input
    box(cx, 92, 52, 8.5, "Input context  +  target lemma", BLUE)
    varrow(cx, 87.5, 82.5)
    # 2. mark + normalize
    box(cx, 78, 52, 8.5, "Mark target  [TGT]\u2009\u2026\u2009[/TGT]   +   normalize", BLUE)
    varrow(cx, 73.5, 68.5)
    # 3. lookup candidates
    box(cx, 64, 52, 8.5, "DiLAC lookup: candidate senses  s\u2081\u2026s\u2099", BLUE)
    varrow(cx, 59.5, 55)

    # ---- per-candidate loop box ----
    loop_x0, loop_y0, loop_w, loop_h = 13, 19, 74, 35
    ax.add_patch(FancyBboxPatch((loop_x0, loop_y0), loop_w, loop_h,
        boxstyle="round,pad=0.4,rounding_size=2",
        fc='none', ec=ORANGE, lw=1.6, linestyle=(0,(5,3))))
    ax.text(loop_x0+2, loop_y0+loop_h+1.5,
            "for each candidate sense s\u1d62:",
            ha='left', va='bottom', fontsize=9, color=ORANGE,
            fontweight='bold', style='italic')

    # 4. enrich (inside loop)
    box(cx, 47, 60, 9, "Enrich s\u1d62:  gloss  +  \u22642 examples\n(leave-one-out: exclude target sentence)", GREEN, fs=9)
    varrow(cx, 42, 37.5)
    # 5. GlossBERT score (inside loop)
    box(cx, 32.5, 60, 9.5, "GlossBERT classifier\n[CLS] C [SEP] S\u1d62 [SEP]   \u2192   p(C, S\u1d62)", GREEN, fs=9)

    varrow(cx, 18.5, 14)
    # 6. argmax → prediction
    box(cx, 9.5, 52, 8.5, "arg max\u1d62  p(C, S\u1d62)   \u2192   predicted sense", ORANGE)

    ax.text(50, 2.5,
            "Knowledge source: DiLAC (human-verified glosses + examples)",
            ha='center', fontsize=8, color=GRAY, style='italic')
    ax.set_title("DiLAC Hybrid WSD Inference Pipeline",
                 fontsize=12.5, fontweight='bold', pad=10)
    fig.savefig(FIGDIR/'fig7_architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print("  ✓ fig7_architecture.png (redesigned)")

# ── Figure 8: accuracy by polysemy band (REAL measured data) ──────────────
def fig_polysemy_band():
    import json
    R = json.load(open(ROOT / 'results' / 'all_results.json', encoding='utf-8'))
    pb = R['track2_wsd_pipeline']['per_band']['bands']
    bands  = ['2', '3', '4', '5-6', '7+']
    acc    = [pb[b]['acc']   for b in bands]
    counts = [pb[b]['share'] for b in bands]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(bands))
    bars = ax1.bar(x, acc, width=0.55, color=BLUE, label='Accuracy')
    ax1.set_xticks(list(x)); ax1.set_xticklabels(bands)
    ax1.set_xlabel('Number of senses (polysemy band)')
    ax1.set_ylabel('Accuracy (%)', color=BLUE)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    for b, v in zip(bars, acc):
        ax1.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}',
                 ha='center', fontsize=9, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, counts, 'o--', color=ORANGE, lw=2, label='% of test set')
    ax2.set_ylabel('Share of test instances (%)', color=ORANGE)
    ax2.set_ylim(0, 40)
    ax2.tick_params(axis='y', labelcolor=ORANGE)

    ax1.set_title('Accuracy by Polysemy Band (measured)\n'
                  'binary-sense words are hardest; mid-polysemy words easiest')
    fig.tight_layout()
    fig.savefig(FIGDIR/'fig8_polysemy_band.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print("  ✓ fig8_polysemy_band.png (real data)")

if __name__ == '__main__':
    print("Generating extra figures ...")
    fig_architecture()
    fig_polysemy_band()
    print(f"Done -> {FIGDIR}")
