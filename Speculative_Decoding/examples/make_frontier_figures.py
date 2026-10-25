"""Original diagrams for the advanced survey; no paper figures are copied."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples.plotting import box, save, COLORS


def ngram_lookup():
    fig, ax = plt.subplots(figsize=(13,4.6))
    ax.set(xlim=(0,13),ylim=(0,4.7));ax.axis("off")
    ax.text(.1,4.25,"Prompt lookup: match the suffix, copy a known continuation",fontsize=17,weight="bold")
    words="the cat sat on the mat and the cat".split()
    ax.text(.1,3.3,"Committed history",color=COLORS['muted'])
    for i,w in enumerate(words):
        color=COLORS['accept'] if i<2 else COLORS['target'] if i<6 else COLORS['draft'] if i>6 else '#77879b'
        box(ax,1.75+i*1.17,2.55,1,.55,w,color)
        ax.text(2.25+i*1.17,2.32,str(i),ha="center",color=COLORS['muted'],fontsize=9)
    ax.text(10.45,3.42,"current suffix",ha="center",color=COLORS['draft'])
    ax.annotate("",xy=(2.75,3.28),xytext=(10.4,3.28),arrowprops={"arrowstyle":"->","color":COLORS['draft'],"lw":2})
    ax.text(5.5,3.45,"longest earlier match: k = 2",ha="center",fontsize=11)
    ax.text(.1,1.35,"Draft proposal",color=COLORS['muted'])
    for i,w in enumerate(words[2:6]):box(ax,4.09+i*1.17,1.05,1,.55,w,COLORS['target'])
    ax.annotate("",xy=(6.25,1.77),xytext=(6.25,2.2),arrowprops={"arrowstyle":"->","color":COLORS['target'],"lw":2})
    ax.text(4.0,.45,"Copy length = 4; match length = 2. The target still verifies every proposed token.",fontsize=11)
    ax.text(4.0,.04,"Known copied sequence => point-mass proposal, not a frequency distribution.",fontsize=11,color=COLORS['muted'])
    save(fig,'07_ngram_lookup')


def dependencies():
    fig,axes=plt.subplots(2,1,figsize=(12,6.7))
    for ax in axes:ax.set(xlim=(0,12),ylim=(0,3.3));ax.axis('off')
    ax=axes[0];ax.text(.1,2.98,"Autoregressive drafting: sampled tokens connect repeated backbone calls",fontsize=14,weight="bold")
    for i in range(4):
        x=2.15+i*2.15
        box(ax,x,1.45,1.55,.65,f"Draft call {i+1}",COLORS['draft'])
        ax.text(x+.77,.98,f"sample y{i+1}",ha="center",color=COLORS['draft'])
        if i<3:ax.annotate("",xy=(x+2.08,1.78),xytext=(x+1.65,1.78),arrowprops={'arrowstyle':'->'})
    ax.text(.1,1.75,"prefix s",fontsize=12)
    ax.text(2.15,.25,"q(y1..y4 | s) = q1(y1 | s) q2(y2 | s,y1) ...",fontsize=12)
    ax=axes[1];ax.text(.1,2.98,"Parallel block drafting: one backbone pass, multiple output positions",fontsize=14,weight="bold")
    box(ax,2.15,1.9,7.99,.55,"Known context features + anchor + MASK positions",COLORS['target'])
    for i in range(4):
        x=2.15+i*2.15
        box(ax,x,.88,1.55,.55,f"q{i+1} -> y{i+1}",COLORS['draft'])
        ax.annotate("",xy=(x+.77,1.52),xytext=(x+.77,1.85),arrowprops={'arrowstyle':'->'})
    ax.text(2.15,.19,"Independent slot draws give a factorized proposal. Correlated draws need their true joint law.",fontsize=10.5)
    ax.text(2.15,-.13,"This diagram describes dependency structure; box widths do not represent measured latency.",fontsize=10,color=COLORS['muted'])
    fig.tight_layout()
    save(fig,'08_draft_dependencies')


if __name__=='__main__':
    ngram_lookup();dependencies()
    print('Wrote mechanism figures 07 and 08 (SVG).')
