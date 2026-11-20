"""Vector result figures from the 18 prediction strategies."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve,auc
from statsmodels.stats.proportion import proportion_confint
from common_preprocessing import clean_sheet,sha,write_json
from factorial_protocol import ROOT,OUT,WORK,STRATEGIES,ALGORITHMS,SETS,paths
from display_labels import display_text, PREDICTOR_SET_DISPLAY
FIG=ROOT/'outputs/figures';COLORS={'Set 1':'#596775','Set 2':'#285f8f','Full set':'#c7652e'}
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
outputs=[];roc_checks=[];all_bins=[]
def export(fig,stem):
    for ext in ['pdf','png']:
        f=FIG/(stem+'.'+ext);fig.savefig(f,dpi=600,bbox_inches='tight',pad_inches=.08);outputs.append(f)
    fig.savefig(WORK/(stem+'_preview.png'),dpi=150,bbox_inches='tight',pad_inches=.08);plt.close(fig)
def legend(fig,y=.0):
    handles=[plt.Line2D([],[],color=COLORS[s],marker='o',lw=1.4,label=display_text(s)) for s in SETS]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,y))
def auroc(perf):
    fig,axes=plt.subplots(1,2,figsize=(9.4,5.22));y=np.arange(6)[::-1]
    for ax,cohort,title in zip(axes,['PPMI_internal','FAHHAUST_external'],['A. Internal PPMI (n=416)','B. External LY-PD (n=197)']):
        for s,offset in zip(SETS,[.22,0,-.22]):
            selected=perf.loc[perf.cohort.eq(cohort)&perf.predictor_set.eq(s)].set_index('algorithm').loc[ALGORITHMS]
            ax.errorbar(selected.AUROC,y+offset,xerr=np.vstack([selected.AUROC-selected.AUROC_low,selected.AUROC_high-selected.AUROC]),fmt='o',ms=4,capsize=2,color=COLORS[s],elinewidth=1)
        lo=min(.4,float(np.floor(perf.AUROC_low.min()*20)/20));hi=max(.85,float(np.ceil(perf.AUROC_high.max()*20)/20))
        ax.set(xlim=(lo,hi),ylim=(-.6,5.6),xlabel='AUROC (95% CI)');ax.set_yticks(y,ALGORITHMS);ax.axvline(.5,color='#8b959c',ls='--',lw=.8)
        ax.set_title(title,loc='left',fontsize=11,weight='bold');ax.grid(axis='x',color='#e6eaed',lw=.6);ax.set_box_aspect(9/8)
    fig.subplots_adjust(left=.08,right=.99,bottom=.17,top=.9,wspace=.28);legend(fig);export(fig,'figure2_predictor_set_auroc')
def increments(delta):
    fig,axes=plt.subplots(2,2,figsize=(9.3,7.8));y=np.arange(6)[::-1]
    for col,metric in enumerate(['AUROC','Brier']):
        sub=delta.loc[delta.metric.eq(metric)&delta.primary_comparison];span=max(abs(sub.low).max(),abs(sub.high).max())*1.13
        for row,cohort in enumerate(['PPMI_internal','FAHHAUST_external']):
            ax=axes[row,col];d=sub.loc[sub.cohort.eq(cohort)].set_index('algorithm').loc[ALGORITHMS]
            ax.errorbar(d.difference,y,xerr=np.vstack([d.difference-d.low,d.high-d.difference]),fmt='o',color=COLORS['Full set'],capsize=3,ms=4)
            ax.axvline(0,color='#7c878f',ls='--',lw=.9);ax.set(xlim=(-span,span),ylim=(-.6,5.6));ax.set_yticks(y,ALGORITHMS);ax.grid(axis='x',color='#e6eaed',lw=.6)
            ax.set_title(('ABCD'[row*2+col])+'. '+('Internal PPMI' if row==0 else 'External LY-PD'),loc='left',fontsize=11,weight='bold')
            ax.set_xlabel(('AUROC difference' if metric=='AUROC' else 'Brier difference')+' (Full set - Basic set)\n'+('Positive favors Full set' if metric=='AUROC' else 'Negative favors Full set'),fontsize=9.3)
    fig.subplots_adjust(left=.08,right=.99,bottom=.1,top=.95,hspace=.5,wspace=.28);export(fig,'figure3_blood_panel_increment')
def age_figure():
    full,_=clean_sheet(pd.read_csv(ROOT/'data/derived/cog_pm3/ppmi_cog.csv'));ext,_=clean_sheet(pd.read_csv(ROOT/'data/derived/cog_v1/fahhaust.csv'));split=np.load(ROOT/'data/derived/cog_pm3/frozen_split.npz')
    internal=full.set_index('PATNO').loc[split['internal_id']]
    tests=pd.read_csv(ROOT/'results/manuscript_statistics/figure2_age_tests.csv')
    arrays=[d.loc[d.COG.eq(c),'age'].dropna().to_numpy() for d in [internal,ext] for c in [0,1]]
    fig,ax=plt.subplots(figsize=(6.6,5.2));pos=[.8,1.6,3,3.8]
    b=ax.boxplot(arrays,positions=pos,widths=.5,patch_artist=True,showfliers=False,medianprops=dict(color='#263541',lw=1.5))
    for patch,color,alpha in zip(b['boxes'],['#285f8f','#285f8f','#c7652e','#c7652e'],[.3,.75,.3,.75]):patch.set(facecolor=color,edgecolor=color,alpha=alpha)
    for j,test in enumerate(tests.itertuples()):
        left,right=pos[2*j:2*j+2];height=94
        ax.plot([left,left,right,right],[height-1,height,height,height-1],color='#263541',lw=1)
        ax.text((left+right)/2,height+1,'P < 0.001' if test.p_holm<.001 else f'P = {test.p_holm:.3f}',ha='center')
        gap=arrays[2*j+1].mean()-arrays[2*j].mean();ax.text((left+right)/2,103,f'Mean gap {gap:.2f} years',ha='center')
    ax.set(xlim=(.2,4.4),ylim=(30,108),ylabel='Age (years)');ax.set_xticks(pos,['COG-negative','COG-positive']*2,fontsize=9)
    ax.text(1.2,-.14,'Internal PPMI',transform=ax.get_xaxis_transform(),ha='center');ax.text(3.4,-.14,'External LY-PD',transform=ax.get_xaxis_transform(),ha='center')
    ax.grid(axis='y',color='#e6eaed',lw=.6);fig.subplots_adjust(bottom=.2);export(fig,'figure_s1_age_case_mix')
def roc_panels(packets,perf):
    for cohort,yk,pk,stem,label in [('PPMI_internal','yi','pi','figure_s2_internal_roc','Internal PPMI'),('FAHHAUST_external','ye','p','figure_s3_external_roc','External LY-PD')]:
        fig,axes=plt.subplots(3,2,figsize=(8,9.2))
        for panel,(ax,algorithm) in enumerate(zip(axes.flat,ALGORITHMS)):
            legend_handles=[];legend_names=[];legend_values=[]
            for spec in [s for s in STRATEGIES if s['algorithm']==algorithm]:
                a=packets[spec['strategy_id']];fpr,tpr,_=roc_curve(a[yk],a[pk]);value=auc(fpr,tpr)
                target=perf.loc[perf.cohort.eq(cohort)&perf.strategy_id.eq(spec['strategy_id'])].iloc[0].AUROC
                assert abs(value-target)<1e-12;roc_checks.append(dict(cohort=cohort,strategy_id=spec['strategy_id'],AUROC=float(value)))
                line=ax.plot(fpr,tpr,color=COLORS[spec['predictor_set']],lw=1.25)[0]
                legend_handles.append(line);legend_names.append(display_text(spec['predictor_set']));legend_values.append(f'{value:.3f}')
            ax.plot([0,1],[0,1],color='#a7b0b8',ls='--',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 - specificity',ylabel='Sensitivity');ax.set_title(f'{chr(65+panel)}. {algorithm}',loc='left',weight='bold')
            # Column-major legend entries: names at one left edge, values at another.
            blanks=[plt.Line2D([],[],lw=0,alpha=0) for _ in legend_values]
            lg=ax.legend(legend_handles+blanks,legend_names+legend_values,ncols=2,
                         frameon=False,fontsize=8,loc='lower right',alignment='left',
                         handlelength=1.6,handletextpad=.35,columnspacing=.25)
            for text in lg.get_texts():text.set_ha('left')
            ax.grid(color='#e6eaed',lw=.5)
        fig.suptitle(label,fontsize=12,weight='bold');fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.06,hspace=.42,wspace=.28);export(fig,stem)
def calibration_panels(packets, retained_bins=None):
    for cohort,yk,pk,stem in [('PPMI_internal','yi','pi','figure_s4_internal_calibration'),('FAHHAUST_external','ye','p','figure_s5_external_calibration')]:
        fig,axes=plt.subplots(3,2,figsize=(8.3,9.4))
        for panel,(ax,algorithm) in enumerate(zip(axes.flat,ALGORITHMS[1:])):
            ax.plot([0,1],[0,1],color='#a7b0b8',ls='--',lw=.8)
            for spec in [s for s in STRATEGIES if s['algorithm']==algorithm]:
                if retained_bins is not None:
                    selected=retained_bins.loc[retained_bins.cohort.eq(cohort)&retained_bins.strategy_id.eq(spec['strategy_id'])].sort_values('bin')
                    assert int(selected.n.sum())==(416 if cohort=='PPMI_internal' else 197)
                    points=selected[['mean_probability','observed_fraction','low','high']].to_numpy()
                else:
                    a=packets[spec['strategy_id']];p,y=a[pk],a[yk];edges=np.unique(np.quantile(p,np.linspace(0,1,6)));group=np.digitize(p,edges[1:-1],right=True);points=[]
                    for g in np.unique(group):
                        mask=group==g;n=int(mask.sum());k=int(y[mask].sum());lo,hi=proportion_confint(k,n,alpha=.05,method='wilson');mean=float(p[mask].mean());fraction=k/n
                        points.append([mean,fraction,lo,hi]);all_bins.append(dict(cohort=cohort,strategy_id=spec['strategy_id'],bin=int(g),n=n,positive=k,mean_probability=mean,observed_fraction=fraction,low=float(lo),high=float(hi)))
                v=np.array(points);ax.errorbar(v[:,0],v[:,1],yerr=np.vstack([v[:,1]-v[:,2],v[:,3]-v[:,1]]),fmt='o-',ms=3.5,lw=1,capsize=2,color=COLORS[spec['predictor_set']])
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed fraction');ax.set_title(f'{chr(65+panel)}. {algorithm}',loc='left',weight='bold');ax.grid(color='#e6eaed',lw=.5)
        axes.flat[-1].set_visible(False);fig.suptitle('Internal PPMI' if cohort=='PPMI_internal' else 'External LY-PD',fontsize=12,weight='bold');fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.09,hspace=.42,wspace=.3);legend(fig);export(fig,stem)
def main(presentation_only=False):
    outputs.clear();roc_checks.clear();all_bins.clear()
    WORK.mkdir(exist_ok=True,parents=True)
    perf=pd.read_csv(OUT/'performance.csv');delta=pd.read_csv(OUT/'paired_differences.csv');packets={}
    for spec in STRATEGIES:
        with np.load(paths(spec['strategy_id'])[1]) as z:packets[spec['strategy_id']]={k:z[k].copy() for k in ['yi','pi','ye','p']}
    FIG.mkdir(parents=True,exist_ok=True);auroc(perf);increments(delta)
    if presentation_only:
        outputs.extend(FIG/('figure_s1_age_case_mix.'+ext) for ext in ['pdf','png'])
        retained=pd.read_csv(OUT/'supplementary_calibration_bins.csv')
    else:
        age_figure();retained=None
    roc_panels(packets,perf);calibration_panels(packets,retained_bins=retained)
    if not presentation_only:
        pd.DataFrame(all_bins).to_csv(OUT/'supplementary_calibration_bins.csv',index=False)
    from calibration_bayes100_figure import main as lr_calibration
    lr_calibration(reuse_bins=presentation_only,manifest_path=OUT/'lr_calibration_display_manifest.json')
    outputs.extend(FIG/('figure4_calibration.'+ext) for ext in ['pdf','png'])
    manifest=dict(status='generated',font_family='Arial',predictive_models_refitted=0,ROC_checks=roc_checks,supplementary_ROC_curves=36,output_sha256={str(f.relative_to(ROOT)):sha(f) for f in outputs},input_sha256={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['performance.csv','paired_differences.csv','supplementary_calibration_bins.csv']},main_LR_calibration_reused=True,presentation_only=presentation_only,predictor_set_display=PREDICTOR_SET_DISPLAY,calibration_bins_reused=presentation_only,roc_legend_alignment={'name_column':'left','AUROC_column':'left'},supplementary_panel_labels={kind:[f'{chr(65+i)}. {a}' for i,a in enumerate(order)] for kind,order in [('S2/S3',ALGORITHMS),('S4/S5',ALGORITHMS[1:])]})
    write_json(OUT/'figure_manifest.json',manifest);print(json.dumps(dict(status='generated',main_figures=[2,3,4],main_figure4_reused=True,supplementary_figures=5,ROC_curves=36,presentation_only=presentation_only)))
if __name__=='__main__':main(presentation_only='--presentation-only' in sys.argv)
