"""Conditional forecast roadmap and analytical cost sensitivity, not benchmark data."""
import numpy as np
from .plotting import plt, save, COLORS
from .forecast_contract import load_forecasts, month_label


def roadmap():
    ledger=load_forecasts()
    forecasts=ledger['forecasts']
    windows=[f["window_months_after_as_of"] for f in forecasts]
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    color_by_conf={'高':COLORS['target'],'中高':COLORS['accept'],'中':COLORS['draft']}
    for i,(f,(start,end)) in enumerate(zip(forecasts,windows)):
        y=len(forecasts)-1-i;color=color_by_conf[f['confidence']]
        ax.plot([start,end],[y,y],lw=8,color=color,solid_capstyle='round')
        ax.scatter([start,end],[y,y],s=35,color='white',edgecolor=color,zorder=3)
    ticks=sorted({0, *[offset for window in windows for offset in window]})
    ax.set(yticks=range(len(forecasts)),yticklabels=[f"{f['id']}: {f['plot_label']}" for f in forecasts][::-1],
           xticks=ticks,xticklabels=[month_label(ledger['as_of'],offset) for offset in ticks],
           xlim=(-.5,max(ticks)+1),ylim=(-.7,len(forecasts)-.3),
           title=f"Forecast evaluation windows (reviewed {ledger['reviewed_on']})")
    ax.grid(axis='x',alpha=.25)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color=color_by_conf[k],lw=5,label=v) for k,v in
                       [('高','High'),('中高','Medium-high'),('中','Medium')]
                       if any(f['confidence']==k for f in forecasts)],loc='lower right',fontsize=9)
    fig.supxlabel('Windows are review horizons, not arrival dates. Confidence concerns each measurable forecast, not only the underlying mechanism.',fontsize=9)
    save(fig,'22_forecast_roadmap')


def cost_sensitivity():
    progress=np.linspace(.8,1.8,120);round_cost=np.linspace(.8,1.8,120)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,length in zip(axes,[1.,1.5]):
        speed=progress[None,:]/(length*round_cost[:,None])
        im=ax.imshow(speed,origin='lower',extent=[.8,1.8,.8,1.8],aspect='auto',vmin=.4,vmax=2,cmap='RdYlGn')
        ax.contour(progress,round_cost,speed,levels=[1],colors=['#172b43'],linewidths=1.5)
        ax.scatter([1.3],[1.0],color=COLORS['target'],s=40)
        ax.annotate(f'{1.3/length:.2f}x',xy=(1.3,1.0),xytext=(1.36,.91),fontsize=11,color=COLORS['target'])
        ax.set(xlabel='Relative committed tokens / round',ylabel='Relative time / round',
               title=f'Output-length ratio = {length:g}')
    fig.colorbar(im,ax=axes,label='Modeled serialized generation speedup')
    fig.supxlabel('S = committed-progress ratio / (round-time ratio x output-length ratio). Includes repair/bonus/EOS in counts; excludes prefill, overlap and tools.',fontsize=9)
    save(fig,'23_acceptance_cost_forecast')


def main():
    roadmap();cost_sensitivity()
    print('Built forecast figures 22-23; no measured performance is implied.')


if __name__=='__main__':main()
