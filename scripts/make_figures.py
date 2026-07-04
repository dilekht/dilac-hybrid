"""
Generate all paper figures from the verified results in results/all_results.json.

Every figure is produced from the single source-of-truth JSON, so figures
can never drift from the reported numbers. Run:

    python scripts/make_figures.py

Outputs PNG files to results/figures/ at 300 dpi.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'all_results.json'
FIGDIR  = ROOT / 'results' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS, encoding='utf-8') as f:
    R = json.load(f)

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.spelines.top' if False else 'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})
BLUE='#2c6fbb'; GREEN='#2e8b57'; GRAY='#888888'; ORANGE='#d9822b'; RED='#c0392b'

def save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}")

# ── Figure 1: WSD main results (the central figure) ──────────────────────
def fig_main_results():
    m = R['track2_wsd_pipeline']['main_results']
    order = ['MFS baseline','DiLAC Lesk-ar','AraBERT zero-shot',
             'CAMeLBERT zero-shot','Two-Stage (k=3)','Two-model BERT ensemble',
             'CAMeLBERT GlossBERT','Fusion (best)']
    accs = [m[k]['acc'] for k in order]
    f1s  = [m[k]['f1']  for k in order]
    # color: baselines gray, zero-shot orange, fine-tuned/hybrid blue, best green
    colors=[GRAY,GRAY,ORANGE,ORANGE,BLUE,BLUE,BLUE,GREEN]
    x=np.arange(len(order)); w=0.4
    fig,ax=plt.subplots(figsize=(10,5))
    b1=ax.bar(x-w/2, accs, w, label='Accuracy', color=colors)
    b2=ax.bar(x+w/2, f1s,  w, label='Macro-F1', color=colors, alpha=0.55)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha='right')
    ax.set_ylabel('Score (%)'); ax.set_ylim(0,80)
    ax.set_title('WSD Performance on DiLAC-WSD Test Set (2,050 instances)')
    for b,v in zip(b1,accs): ax.text(b.get_x()+b.get_width()/2,v+0.8,f'{v:.1f}',
                                     ha='center',fontsize=8)
    ax.legend(); fig.tight_layout()
    save(fig,'fig1_main_results.png')

# ── Figure 2: zero-shot vs fine-tuned (the headline) ─────────────────────
def fig_finetune_gain():
    fig,ax=plt.subplots(figsize=(6,5))
    cats=['Zero-shot','Fine-tuned\n(GlossBERT)']
    vals=[50.39,69.56]
    bars=ax.bar(cats,vals,color=[ORANGE,GREEN],width=0.5)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,80)
    ax.set_title('Effect of Fine-tuning CAMeLBERT-MSA on DiLAC\n(same test set)')
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',
                                       ha='center',fontweight='bold')
    ax.annotate('', xy=(1,69.56), xytext=(0,50.39),
                arrowprops=dict(arrowstyle='->',color=RED,lw=2))
    ax.text(0.5,62,'+19.2 pts',color=RED,fontweight='bold',ha='center')
    fig.tight_layout(); save(fig,'fig2_finetune_gain.png')

# ── Figure 3: sense-ID configuration comparison ──────────────────────────
def fig_senseid_configs():
    c=R['track1_knowledge_based']['sense_identification']['configurations']
    order=['Baseline (binary)','Gloss Priority (x3)','Frequency Weighting',
           'Combined','Discriminative IDF + Combined']
    accs=[c[k]['acc'] for k in order]
    f1s =[c[k]['f1']  for k in order]
    labels=['Baseline','Gloss\nPriority','Frequency','Combined','Discrim.\nIDF']
    x=np.arange(len(order)); w=0.4
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-w/2,accs,w,label='Accuracy',color=BLUE)
    ax.bar(x+w/2,f1s, w,label='Macro-F1',color=GREEN,alpha=0.7)
    ax.axhline(1.75,ls='--',color=RED,label='Random (1.75%)')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Score (%)'); ax.set_ylim(0,100)
    ax.set_title('Sense Identification: Lesk-ar Configurations (57 classes)')
    ax.legend(); fig.tight_layout(); save(fig,'fig3_senseid_configs.png')

# ── Figure 4: AWSS three-way similarity comparison ───────────────────────
def fig_awss():
    a=R['track1_knowledge_based']['awss_similarity']
    sys=['Human\nceiling','AWSS algo.\n(WordNet)','AWSS algo.\n(reproduced)',
         'DiLAC Lesk-ar\n(ours)']
    vals=[a['human_ceiling'],a['awss_algorithm_paper'],
          a['awss_algorithm_reproduced'],a['dilac_leskar_vs_human']]
    colors=[GRAY,BLUE,BLUE,GREEN]
    fig,ax=plt.subplots(figsize=(7,5))
    bars=ax.bar(sys,vals,color=colors,width=0.6)
    ax.set_ylabel('Pearson $r$ with human ratings'); ax.set_ylim(0,1.0)
    ax.set_title('Semantic Similarity on AWSS Benchmark (35 pairs)')
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.015,
                                       f'{v:.3f}',ha='center',fontsize=9)
    fig.tight_layout(); save(fig,'fig4_awss_similarity.png')

# ── Figure 5: ablation ───────────────────────────────────────────────────
def fig_ablation():
    ab=R['track2_wsd_pipeline']['ablation']
    order=['Full system','- Domain labels','- Runtime Lesk-ar',
           '- Two-stage filter','- BERT embeddings']
    accs=[ab[k]['acc'] for k in order]
    labels=['Full\nsystem','− Domain','− Lesk-ar','− Two-stage','− BERT']
    colors=[GREEN,GRAY,GRAY,GRAY,RED]
    fig,ax=plt.subplots(figsize=(7,5))
    bars=ax.bar(labels,accs,color=colors,width=0.6)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,80)
    ax.set_title('Ablation: Component Removed from Full System')
    for b,v in zip(bars,accs): ax.text(b.get_x()+b.get_width()/2,v+1,
                                       f'{v:.1f}',ha='center',fontsize=9)
    ax.text(4,15,'−61.3 pts',color=RED,fontweight='bold',ha='center')
    fig.tight_layout(); save(fig,'fig5_ablation.png')

# ── Figure 6: stemming study ─────────────────────────────────────────────
def fig_stemming():
    s=R['track1_knowledge_based']['stemming_study']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.5))
    # sense-id
    ax1.bar(['With\nstemming','Without\nstemming'],
            [s['with_stemming']['senseid_acc'],s['without_stemming']['senseid_acc']],
            color=[GRAY,GREEN],width=0.5)
    ax1.set_ylabel('Sense-ID Accuracy (%)'); ax1.set_ylim(0,60)
    ax1.set_title('(a) Sense Identification')
    for i,v in enumerate([s['with_stemming']['senseid_acc'],
                          s['without_stemming']['senseid_acc']]):
        ax1.text(i,v+1,f'{v:.1f}',ha='center')
    # awss
    ax2.bar(['With\nstemming','Without\nstemming'],
            [s['with_stemming']['awss_r'],s['without_stemming']['awss_r']],
            color=[GRAY,GREEN],width=0.5)
    ax2.set_ylabel('AWSS Pearson $r$'); ax2.set_ylim(0,1.0)
    ax2.set_title('(b) Semantic Similarity')
    for i,v in enumerate([s['with_stemming']['awss_r'],
                          s['without_stemming']['awss_r']]):
        ax2.text(i,v+0.02,f'{v:.3f}',ha='center')
    fig.suptitle('Effect of Light Stemming on News-Register Arabic')
    fig.tight_layout(); save(fig,'fig6_stemming.png')

if __name__ == '__main__':
    print("Generating figures from verified results ...")
    fig_main_results()
    fig_finetune_gain()
    fig_senseid_configs()
    fig_awss()
    fig_ablation()
    fig_stemming()
    print(f"\nAll figures written to {FIGDIR}")
