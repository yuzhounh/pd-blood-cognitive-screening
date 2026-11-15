"""Descriptive calibration display for the three predefined nested LR strategies."""
from pathlib import Path
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.stats.proportion import proportion_confint
from common_preprocessing import sha,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/results/bayes100'
FIG=ROOT/'manuscript/figures'
PRIVATE=ROOT/'data/derived/bayes100'
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Arial','font.size':10.5,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def main(reuse_bins=False):
    assert json.loads((OUT/'summary_validation.json').read_text())['status']=='computed_and_numerically_checked'
    names=['Age_LR','Basic_LR','LR'];labels=[MODEL_LABELS[name] for name in names]
    colors=['#5b6770','#285f8f','#c7652e']
    inputs={str((PRIVATE/(n+'.npz')).relative_to(ROOT)):sha(PRIVATE/(n+'.npz')) for n in names}
    retained=pd.read_csv(OUT/'calibration_bins.csv') if reuse_bins else None
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.4));rows=[]
    for ax,(cohort,yk,pk,title) in zip(axes,[('PPMI_internal','yi','pi','A. Internal PPMI (n=416)'),
                                          ('FAHHAUST_external','ye','p','B. External LY-PD (n=197)')]):
        ax.plot([0,1],[0,1],ls='--',color='#8b959c',lw=1,zorder=0)
        for name,label,color in zip(names,labels,colors):
            d=np.load(PRIVATE/(name+'.npz'));y,p=d[yk],d[pk]
            points=[];covered=0
            if reuse_bins:
                selected=retained.loc[retained.cohort.eq(cohort)&retained.model.eq(name)].sort_values('bin')
                covered=int(selected.n.sum())
                points=selected[['mean_probability','observed_fraction','low','high']].to_numpy().tolist()
            else:
                edges=np.unique(np.quantile(p,np.linspace(0,1,6)))
                group=np.digitize(p,edges[1:-1],right=True)
                for g in np.unique(group):
                    selected=group==g;n=int(selected.sum());positive=int(y[selected].sum());covered+=n
                    mean=float(p[selected].mean());fraction=positive/n
                    low,high=proportion_confint(positive,n,alpha=.05,method='wilson')
                    points.append([mean,fraction,float(low),float(high)])
                    rows.append(dict(cohort=cohort,model=name,bin=int(g),n=n,positive=positive,
                                     mean_probability=mean,observed_fraction=fraction,low=float(low),high=float(high)))
            assert covered==len(y) and len(points)<=5
            a=np.asarray(points)
            ax.errorbar(a[:,0],a[:,1],yerr=np.vstack([a[:,1]-a[:,2],a[:,3]-a[:,1]]),
                        fmt='o-',color=color,lw=1,elinewidth=.8,capsize=2,ms=4,label=label,alpha=.85)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed COG-positive fraction')
        ax.set_box_aspect(4/5)
        ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
        ax.set_title(title,loc='left',fontsize=11.5,weight='bold',pad=10)
        ax.grid(color='#e6eaed',lw=.6,zorder=0)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.0))
    fig.subplots_adjust(left=.085,right=.985,top=.87,bottom=.22,wspace=.33)
    fig.canvas.draw()
    assert all(np.isclose(ax.get_window_extent().height/ax.get_window_extent().width,4/5)
               for ax in axes)
    for extension in ['pdf','png']:
        fig.savefig(FIG/('figure4_calibration.'+extension),dpi=600,bbox_inches='tight',pad_inches=.08)
    fig.savefig(ROOT/'tmp/preview_figure4_calibration.png',dpi=150,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    if not reuse_bins:
        pd.DataFrame(rows).to_csv(OUT/'calibration_bins.csv',index=False)
    write_json(OUT/'calibration_figure_manifest.json',dict(status='generated',code_sha256=sha(__file__),
               input_sha256=inputs,maximum_bins=5,interval='Wilson 95%; descriptive evaluation sample',
               model_or_threshold_updates=0,retained_bins_reused=reuse_bins,
               output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
                   [FIG/'figure4_calibration.pdf',FIG/'figure4_calibration.png']}))
    print(json.dumps(dict(event='BAYES100_CALIBRATION_FIGURE_GENERATED',models=3,cohorts=2,model_refits=0)),flush=True)


if __name__=='__main__':main()
