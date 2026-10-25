"""Figures 11-21 for executable principle lessons; no GPU benchmark claims."""
from pathlib import Path
import json
import numpy as np
from .plotting import plt, box, save, COLORS

ROOT=Path(__file__).resolve().parents[1]


def matrix(ax, values, title, xlabels, ylabels, cmap='Blues', vmax=None):
    values=np.asarray(values)
    ax.imshow(values,cmap=cmap,vmin=0,vmax=vmax if vmax is not None else max(1e-9,values.max()))
    for (i,j),v in np.ndenumerate(values):
        text=f'{v:.2f}' if values.dtype.kind=='f' else str(v)
        ax.text(j,i,text,ha='center',va='center',fontsize=10,
                color='white' if v>values.max()*.55 else '#172b43')
    ax.set(xticks=range(len(xlabels)),xticklabels=xlabels,yticks=range(len(ylabels)),yticklabels=ylabels,title=title)


def coupling(r):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    joint=np.array(r['coupling']['joint'])
    matrix(axes[0],joint,'Maximal coupling: row sum = q, column sum = p',list('ABC'),list('ABC'))
    axes[0].set(xlabel='Final target token X',ylabel='Proposed token Y')
    x=np.arange(3);p=np.array(r['coupling']['target']);q=np.array(r['coupling']['proposal'])
    axes[1].bar(x,np.minimum(p,q),color=COLORS['accept'],label='Accepted diagonal mass')
    axes[1].bar(x,np.maximum(p-q,0),bottom=np.minimum(p,q),color=COLORS['reject'],label='Repair mass')
    axes[1].set(xticks=x,xticklabels=list('ABC'),ylim=(0,.7),title='Two routes reconstruct the target marginal')
    axes[1].legend(fontsize=9)
    fig.supxlabel('Off-diagonal mass moves from over-proposed B/C into under-proposed A.',fontsize=10)
    save(fig,'11_coupling_transport')


def processors(r):
    fig,axes=plt.subplots(1,5,figsize=(14,3.6),layout='constrained')
    titles={'raw':'Softmax T=1','cold':'Temperature T=0.5','top_k':'Top-k = 2','top_p':'Top-p = 0.8','grammar':'Mask out A'}
    for ax,(key,p) in zip(axes,r['processors'].items()):
        ax.bar(list('ABC'),p,color=COLORS['draft'])
        ax.set(ylim=(0,1),title=titles[key])
        for i,v in enumerate(p):ax.text(i,v+.03,f'{v:.3f}',ha='center',fontsize=9)
    axes[0].set_ylabel('Actual proposal probability')
    fig.supxlabel('Each panel applies its named change to logits [2, 1, 0]; these are different q distributions.',fontsize=10)
    save(fig,'12_actual_proposal')


def feature_flow():
    fig,axes=plt.subplots(3,1,figsize=(12,8.5),layout='constrained')
    descriptions=[('Independent draft','Token prefix','Small LM','Next token q','No target features'),
                  ('EAGLE-style step','Target feature + chosen token','Light decoder','State a + token q','Rollout reuses a, not unavailable future target features'),
                  ('DFlash-style block','Target context + anchor/MASK','Parallel backbone','Multiple slot logits','Context features remain visible at each draft layer')]
    for ax,(title,a,b,c,note) in zip(axes,descriptions):
        ax.set(xlim=(0,12),ylim=(0,2.5));ax.axis('off');ax.text(.1,2.1,title,fontsize=14,weight='bold')
        for x,w,label,color in [(0.2,3.4,a,'target'),(4.2,3,b,'draft'),(8,3.7,c,'accept')]:
            box(ax,x,1,w,.65,label,COLORS[color])
        for x in [3.7,7.35]:ax.annotate('',(x+.35,1.32),(x,1.32),arrowprops={'arrowstyle':'->'})
        ax.text(.2,.35,note,fontsize=11,color=COLORS['muted'])
    save(fig,'13_feature_conditioning')


def layer_reuse():
    fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    ax.text(.1,4.6,'Layer reuse: serial lower layers, batched upper layers',fontsize=17,weight='bold')
    for i in range(3):
        x=1.1+i*3.4
        box(ax,x,3.25,2.4,.55,f'Token {i+1}',COLORS['target'])
        box(ax,x,2.3,2.4,.55,'Early layers 1..E',COLORS['draft'])
        box(ax,x,1.35,2.4,.5,f'Saved activation a{i+1}',COLORS['accept'])
        for y in [2.95,2.0]:ax.annotate('',(x+1.2,y-.1),(x+1.2,y+.25),arrowprops={'arrowstyle':'->'})
    box(ax,1.1,.3,9.2,.55,'One causal pass through late layers E+1..L using saved activations',COLORS['target'])
    for i in range(3):ax.annotate('',(2.3+i*3.4,.9),(2.3+i*3.4,1.25),arrowprops={'arrowstyle':'->'})
    ax.text(.1,-.15,'Valid only when early-layer inputs, masks, positions and weights match the full target.',fontsize=10)
    save(fig,'14_layer_reuse')


def training_mask(r):
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    mask=np.array(r['block_mask'])
    matrix(ax,mask,'Two training blocks: legal target features + isolated anchor/MASK slots',
           ['g0','g1','g2','g3','g4','a2','M','M','a4','M','M'],['b0:a2','b0:M','b0:M','b1:a4','b1:M','b1:M'])
    ax.axvline(4.5,color=COLORS['reject'],lw=2);ax.axvline(7.5,color=COLORS['reject'],lw=2)
    ax.axhline(2.5,color=COLORS['reject'],lw=2)
    ax.set(xlabel='Keys: target context features | block 0 inputs | block 1 inputs',ylabel='Draft queries')
    fig.supxlabel('M means MASK input, never the true future token. 1 = visible, 0 = hidden.',fontsize=10)
    save(fig,'15_block_training_mask')


def dependence(r):
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['target','independent','conditional'],['Target: correlated modes','Independent slots','Conditional second slot']):
        matrix(ax,np.array(r['dependence'][key]),title,['cat','cats'],['a','two'],vmax=.5)
    fig.supxlabel('All three have [0.5,0.5] marginals. Independent slots waste 50% joint mass; conditional slots waste 5%.',fontsize=10)
    save(fig,'16_joint_dependence')


def tree_positions(r):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    matrix(axes[0],np.array(r['tree']['mask']),'Tree visibility by flattened node index',range(6),range(6))
    axes[0].set(xlabel='Key index',ylabel='Query index')
    axes[1].axis('off')
    data=[[i,parent,pos] for i,(parent,pos) in enumerate(zip(r['tree']['parents'],r['tree']['positions']))]
    table=axes[1].table(cellText=data,colLabels=['Flat index','Parent','Logical position'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,1.8)
    axes[1].set_title('Flattened order differs from token position')
    axes[1].text(.03,.04,'Select node 4: gather common prefix + [0, 1, 4].\nCropping flat rows [0,1,2,3,4] would include siblings.',transform=axes[1].transAxes,fontsize=10)
    save(fig,'17_tree_positions_gather')


def gradients(r):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(3)
    for offset,key,color in [(-.24,'kl','target'),(0,'tv','draft'),(.24,'hybrid','accept')]:
        axes[0].bar(x+offset,r['training'][key]['gradient'],.23,label=key.upper(),color=COLORS[color])
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xticks=x,xticklabels=list('ABC'),title='Logit gradients: target [.8,.15,.05], q uniform',ylabel='Gradient (descent moves opposite)');axes[0].legend()
    weights=r['training']['dpace']['weights']
    axes[1].bar([1,2,3],weights,color=COLORS['accept']);axes[1].set(xticks=[1,2,3],xlabel='Draft slot',ylabel='Detached CE weight',title='D-PACE weights from current token confidences')
    for i,w in enumerate(weights,1):axes[1].text(i,w+.03,f'{w:.3f}',ha='center')
    save(fig,'18_training_gradients')


def scheduler(r):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(1,4)
    for alpha,label,color in [([.9,.9,.9],'Request A','target'),([.6,.4,.2],'Request B','draft')]:
        axes[0].plot(x,np.cumprod(alpha),'o-',label=label,color=COLORS[color])
    axes[0].set(xticks=x,ylim=(0,1),xlabel='Candidate depth',ylabel='Prefix survival probability',title='Budget uses cumulative survival, not slot confidence');axes[0].legend()
    chosen=r['scheduler']['allocation'];axes[1].axis('off')
    axes[1].text(.02,.9,'Synthetic cost T(B) = 1 + 0.1 B',fontsize=14,weight='bold')
    lines=[f"Best lengths: A={chosen['lengths'][0]}, B={chosen['lengths'][1]}",f"Verified tokens (including anchors): {chosen['total_verify_tokens']}",f"Expected new output: {chosen['expected_emitted']:.3f}",f"Expected output / cost: {chosen['tokens_per_cost']:.3f}",'Scores must be known before current token sampling.','Retrospective selection example: P(A) .70 -> .85 (biased).']
    for i,line in enumerate(lines):axes[1].text(.02,.73-i*.12,line,fontsize=11)
    save(fig,'19_causal_budget')


def memory():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    lengths=np.array([1,2,4,8,16,32])*1024
    for kvheads,label in [(32,'MHA: 32 KV heads'),(8,'GQA: 8 KV heads')]:
        gib=2*lengths*32*kvheads*128*2/(1024**3)
        axes[0].plot(lengths/1024,gib,'o-',label=label)
    axes[0].set(xlabel='Context length (Ki tokens)',ylabel='KV memory (GiB)',title='B=1, 32 layers, head dim=128, 2 bytes/element');axes[0].legend()
    axes[1].axis('off');axes[1].text(.03,.9,'TriForce-style hierarchy',fontsize=15,weight='bold')
    for y,label,color in [(.67,'Small LM + short cache','draft'),(.42,'Target weights + retrieved KV','accept'),(.17,'Target weights + full KV','target')]:
        axes[1].text(.5,y,label,ha='center',bbox=dict(boxstyle='round,pad=.7',facecolor=COLORS[color],edgecolor='none'),color='white')
    for y in [.56,.31]:axes[1].annotate('',(.5,y-.07),(.5,y+.06),arrowprops={'arrowstyle':'->'})
    fig.supxlabel('Analytical memory sizes; sparse approximation stays on the proposal side.',fontsize=10)
    save(fig,'20_kv_hierarchy')


def survival():
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    sequences=[[.5,.9,.9],[.9,.9,.5]]
    for a,label,color in zip(sequences,['Weak early: [.5,.9,.9]','Weak late: [.9,.9,.5]'],['reject','accept']):
        s=np.cumprod(a)
        axes[0].plot([0,1,2,3],[1,*s],'o-',label=f'{label}; E[L]={1+s.sum():.3f}',color=COLORS[color])
    axes[0].set(xlabel='Accepted prefix depth',ylabel='Survival',xticks=[0,1,2,3],ylim=(0,1.05),title='Same slot-average acceptance, different useful progress');axes[0].legend(fontsize=9)
    p=np.array([.5,.3,.2]);q=np.array([.2,.5,.3]);common=np.minimum(p,q)
    accepted=common/common.sum();axes[1].bar(np.arange(3)-.18,p,.35,label='Unconditional target p',color=COLORS['target']);axes[1].bar(np.arange(3)+.18,accepted,.35,label='Token | accepted',color=COLORS['draft'])
    axes[1].set(xticks=[0,1,2],xticklabels=list('ABC'),ylim=(0,.65),title='Conditioning on acceptance changes the token law');axes[1].legend(fontsize=9)
    save(fig,'21_prefix_survival')


def main():
    r=json.loads((ROOT/'data/principle_results.json').read_text())
    coupling(r);processors(r);feature_flow();layer_reuse();training_mask(r);dependence(r)
    tree_positions(r);gradients(r);scheduler(r);memory();survival()
    print('Rebuilt principle figures 11-21 from saved numeric results and stated toy assumptions.')


if __name__=='__main__':main()
