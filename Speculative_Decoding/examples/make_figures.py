"""Rebuild all figures from saved results; never rerun or overwrite experiments."""
from pathlib import Path
import json
import numpy as np
from examples.plotting import plt, box, save, COLORS
from examples.speculative_sampling import expected_tokens

ROOT = Path(__file__).resolve().parents[1]


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.7))
    ax.set(xlim=(0, 12), ylim=(0, 5)); ax.axis("off")
    ax.text(.1, 4.7, "One speculative round: propose, verify, commit", fontsize=17, weight="bold")
    ax.text(.1, 3.9, "1. Draft\n(serial)", color=COLORS["muted"])
    for i in range(4):
        box(ax, 2 + 1.8 * i, 3.55, 1.35, .65, f"q{i+1} -> y{i+1}", COLORS["draft"])
        if i < 3: ax.annotate("", (3.7+1.8*i, 3.87), (3.4+1.8*i, 3.87), arrowprops={"arrowstyle":"->"})
    ax.text(.1, 2.5, "2. Target\n(one causal pass)", color=COLORS["muted"])
    for i, label in enumerate(["last token", "y1", "y2", "y3", "y4"]):
        box(ax, 2 + 1.8*i, 2.25, 1.35, .6, label, COLORS["target"])
        ax.text(2.675+1.8*i, 1.95, f"logits -> p{i+1}", ha="center", color=COLORS["target"])
    ax.text(.1, .85, "3. Commit\n(example)", color=COLORS["muted"])
    for i, (label, color) in enumerate([("y1 accept",COLORS["accept"]),("y2 accept",COLORS["accept"]),
                                       ("y3 reject",COLORS["reject"]),("y4 discard","#9ca8b6")]):
        box(ax, 2 + 1.8*i, .65, 1.35, .65, label, color)
    box(ax, 9.2, .65, 2.15, .65, "repair z ~ residual", COLORS["target"])
    ax.text(2, .1, "Output: y1, y2, z.  If all 4 drafts pass: output y1..y4 plus a bonus token from p5.", fontsize=11)
    save(fig, "01_pipeline")


def probability_mass():
    p, q = np.array([.5,.3,.2]), np.array([.2,.5,.3])
    accepted = np.minimum(p,q)
    fig, axes = plt.subplots(1,2,figsize=(11,4.1), layout="constrained")
    x=np.arange(3)
    axes[0].bar(x-.18,p,.34,label="Target p",color=COLORS["target"])
    axes[0].bar(x+.18,q,.34,label="Proposal q",color=COLORS["draft"])
    axes[0].set(xticks=x,xticklabels=list("ABC"),ylim=(0,.65),ylabel="Probability",title="Two distributions; overlap = 0.70")
    axes[0].legend()
    axes[1].bar(x,accepted,label="Accepted mass min(p,q)",color=COLORS["accept"])
    axes[1].bar(x,p-accepted,bottom=accepted,label="Repair mass [p-q]+",color=COLORS["reject"])
    axes[1].set(xticks=x,xticklabels=list("ABC"),ylim=(0,.65),title="Accepted mass + repair mass = p")
    axes[1].legend()
    save(fig,"02_probability_mass")


def speed_model():
    gamma=np.arange(0,17)
    fig, axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    c=.08
    for alpha in [.5,.7,.85,.95]:
        e=np.array([expected_tokens(alpha,int(g)) for g in gamma])
        speed=e/(1+c*gamma)
        axes[0].plot(gamma,speed,"o-",ms=3,label=f"alpha = {alpha}")
    axes[0].axhline(1,color="gray",ls="--")
    axes[0].set(xlabel="Draft length gamma",ylabel="Modeled decode speedup",title="Ideal verifier: v = 1; draft cost c = 0.08")
    axes[0].legend()
    alphas=np.linspace(.3,.99,100); costs=np.linspace(.01,.4,100)
    best=np.ones((len(costs),len(alphas)))
    for g in range(1,17):
        v=1+.035*g
        speed=expected_tokens(alphas,g)[None,:]/(v+costs[:,None]*g)
        best=np.maximum(best,speed)
    im=axes[1].imshow(best,origin="lower",extent=[.3,.99,.01,.4],aspect="auto",cmap="viridis",vmin=1)
    fig.colorbar(im,ax=axes[1],label="Best modeled speedup, including baseline")
    axes[1].set(xlabel="Conditional acceptance alpha",ylabel="Draft cost c = Td / T1",title="Non-ideal verifier: v(gamma) = 1 + 0.035 gamma")
    fig.supxlabel("Analytical scenarios only. These are not measured GPU benchmarks.",color=COLORS["muted"],fontsize=10)
    save(fig,"03_speed_model")


def tree_mask():
    labels=["root", "A", "B", "C", "D", "E"]
    parents=[-1,0,0,1,1,2]
    xy=[(0,2),(1.6,3.2),(1.6,.8),(3.2,3.7),(3.2,2.6),(3.2,.8)]
    mask=np.zeros((6,6),dtype=int)
    for row in range(6):
        node=row
        while node>=0:
            mask[row,node]=1;node=parents[node]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    axes[0].set(xlim=(-.6,4),ylim=(0,4.3),title="Candidate tree (root = last committed token)"); axes[0].axis("off")
    for i,(x,y) in enumerate(xy):
        if parents[i]>=0:
            px,py=xy[parents[i]];axes[0].plot([px,x],[py,y],color="#b2bdca",zorder=0)
        axes[0].scatter(x,y,s=1300,color=COLORS["target"] if i==0 else COLORS["draft"])
        axes[0].text(x,y,labels[i],color="white",ha="center",va="center")
    axes[0].text(-.3,.05,"C sees root, A, C; it cannot see B, D, or E.",fontsize=10)
    axes[1].imshow(mask,cmap="Blues",vmin=0,vmax=1)
    for i in range(6):
        for j in range(6):axes[1].text(j,i,str(mask[i,j]),ha="center",va="center",color="white" if mask[i,j] else "#677")
    axes[1].set(xticks=range(6),xticklabels=labels,yticks=range(6),yticklabels=labels,
                xlabel="Key node",ylabel="Query node",title="Tree attention mask (1 = visible)")
    save(fig,"04_tree_attention")


def cache_rollback():
    fig,ax=plt.subplots(figsize=(12,4.3));ax.set(xlim=(0,12),ylim=(0,4.7));ax.axis("off")
    ax.text(.1,4.35,"KV cache after rejecting y3 (2 drafts accepted)",fontsize=17,weight="bold")
    rows=[(3.25,"Round start",["P1","P2","P3"],"last x is pending"),
          (2.1,"After verify",["P1","P2","P3","x","y1","y2","y3","y4"],""),
          (.95,"After crop",["P1","P2","P3","x","y1","y2"],"repair z is pending")]
    for y,label,tokens,note in rows:
        ax.text(.1,y+.25,label,va="center",color=COLORS["muted"])
        for i,token in enumerate(tokens):
            color=COLORS["reject"] if token in ["y3","y4"] else COLORS["target"] if i<4 else COLORS["accept"]
            box(ax,2+i*.9,y,.75,.5,token,color)
        if note:ax.text(2+len(tokens)*.9+.1,y+.25,note,va="center",fontsize=10)
    ax.text(2,.15,"Both target and draft caches keep only prefix + accepted drafts; correction/bonus enters next round.",fontsize=10)
    save(fig,"05_kv_rollback")


def empirical(result):
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    x=np.arange(3)
    for offset,key,label,color in [(-.25,"target","Exact target",COLORS["target"]),
                                    (0,"exact_empirical","Correct repair: empirical",COLORS["accept"]),
                                    (.25,"wrong_empirical","Wrong repair from p: empirical",COLORS["reject"])]:
        values = np.asarray(result[key])
        error = None if key == "target" else 1.96 * np.sqrt(values * (1-values) / result["samples"])
        ax.bar(x+offset, values, .24, label=label, color=color, yerr=error,
               capsize=3, error_kw={"elinewidth": 1})
    ax.set(xticks=x,xticklabels=list("ABC"),ylim=(0,.65),ylabel="Output probability",
           title=f"Monte Carlo validation: N = {result['samples']:,}, seed = {result['seed']}")
    ax.legend()
    fig.supxlabel("Whiskers: approximate 95% marginal Monte Carlo intervals (not simultaneous).",
                  color=COLORS["muted"], fontsize=9)
    save(fig,"06_empirical_validation")


def terminal_eos():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    ax.text(.1, 4.65, "Accepted EOS: commit it, stop, and emit no bonus", fontsize=17, weight="bold")
    rows = [
        (3.6, "Round start", [("P1", "target"), ("P2", "target")], "last x is pending"),
        (2.45, "Target input", [("x", "target"), ("y1", "draft"), ("EOS", "draft")], "proposer stops at EOS"),
        (1.3, "New output", [("y1", "accept"), ("EOS", "accept")], "accepted = 2; committed = 2"),
        (.15, "Final KV", [("P1", "target"), ("P2", "target"), ("x", "target"), ("y1", "accept")], "length = output length - 1"),
    ]
    for y, label, tokens, note in rows:
        ax.text(.1, y+.28, label, color=COLORS["muted"], va="center")
        for i, (token, color) in enumerate(tokens):
            box(ax, 2.0 + i*1.05, y, .85, .55, token, COLORS[color])
        ax.text(2.1+len(tokens)*1.05, y+.28, note, va="center", fontsize=11)
    save(fig, "10_terminal_eos")


if __name__ == "__main__":
    import argparse
    from examples import plotting
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="Also export Matplotlib figures as PNG (the technical map is SVG-only)")
    plotting.EXPORT_PNG = parser.parse_args().png
    from examples.make_frontier_figures import ngram_lookup, dependencies
    from examples.make_mainstream_figure import main as technical_map
    from examples.make_principle_figures import main as principle_figures
    from examples.make_forecast_figures import main as forecast_figures
    from examples.make_exactness_figures import make as exactness_figures
    from examples.make_training_review_figures import main as training_review_figures
    from examples.make_systems_figures import main as systems_figures
    from examples.make_cost_figures import make as cost_figures
    result = json.loads((ROOT / "data" / "experiment_results.json").read_text())["one_step"]
    pipeline()
    probability_mass()
    speed_model()
    tree_mask()
    cache_rollback()
    empirical(result)
    ngram_lookup()
    dependencies()
    technical_map()
    terminal_eos()
    principle_figures()
    forecast_figures()
    exactness_figures()
    training_review_figures()
    systems_figures()
    cost_figures()
    print(f"Rebuilt {len(list((ROOT/'docs/figures').glob('*.svg')))} figures; experiment JSON files were not changed.")
