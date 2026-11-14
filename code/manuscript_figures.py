"""Publication figures from frozen model results and within-cohort age comparisons."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from common_preprocessing import clean_sheet, sha, write_json

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'manuscript/figures'
OUT=ROOT/'manuscript/results/manuscript_statistics'
BLUE,ORANGE,INK='#285f8f','#c7652e','#263541'
FONT_FAMILY='Arial'
font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),fallback_to_default=False)
plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':10.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':INK,
                     'xtick.color':INK,'ytick.color':INK,'text.color':INK,
                     'savefig.facecolor':'white'})


def export(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.08)
    fig.savefig(ROOT/'tmp'/('preview_'+name+'.png'),dpi=150,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def flow_figure():
    fig,ax=plt.subplots(figsize=(9.4,7.4))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color=BLUE,fill='#f0f5f9',size=12):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',
                                   linewidth=1.1,edgecolor=color,facecolor=fill))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.3)
    def arrow(start,end):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',lw=1.15,color='#536575',shrinkA=1,shrinkB=1))
    ppmi_center,training_center,internal_center,external_center=.3125,.1775,.4825,.815
    ax.text(ppmi_center,1.015,'PPMI development',ha='center',va='top',weight='bold',fontsize=14)
    ax.text(external_center,1.015,'External validation',ha='center',va='top',weight='bold',fontsize=14)
    box(.025,.88,.575,.08,'Updated raw PPMI records\n2,263 participants classified as PD')
    box(.655,.88,.32,.08,'Published FAHHAUST\n197 participants with PD',color=ORANGE,fill='#fbf2ec')
    arrow((ppmi_center,.872),(ppmi_center,.848))
    box(.025,.75,.575,.09,'Eligible cognitive and blood matches: 1,385\nCOG+: 679 / COG−: 706\n878 without eligible matches')
    arrow((ppmi_center,.742),(ppmi_center,.718))
    box(.025,.63,.575,.08,'Outcome-stratified 70:30 split\nTraining: 969 / Internal test: 416')
    ax.plot([ppmi_center,ppmi_center],[.622,.595],lw=1.15,color='#536575')
    ax.plot([training_center,internal_center],[.595,.595],lw=1.15,color='#536575')
    arrow((training_center,.595),(training_center,.568))
    arrow((internal_center,.595),(internal_center,.568))
    box(.025,.47,.305,.09,'PPMI training sample\nn = 969\nCOG+: 475 / COG−: 494',size=11.8)
    box(.365,.47,.235,.09,'Internal PPMI test\nn = 416\nCOG+: 204 / COG−: 212',size=11.2)
    box(.655,.6925,.32,.055,'External records reserved for evaluation',color=ORANGE,fill='#fbf2ec',size=11.5)
    arrow((external_center,.872),(external_center,.7555))
    arrow((external_center,.6845),(external_center,.568))
    box(.655,.47,.32,.09,'External FAHHAUST\nn = 197\nCOG+: 122 / COG−: 75',color=ORANGE,fill='#fbf2ec',size=11.8)
    arrow((training_center,.462),(training_center,.413))
    box(.025,.29,.305,.115,'Training-only development\nFold-fitted preprocessing\nMean five-fold AUROC tuning\nSix algorithms + LR references',size=11.5)
    arrow((training_center,.282),(training_center,.243))
    box(.025,.125,.305,.11,'Final fit on all 969 participants\nTraining OOF Youden thresholds\nFrozen models and preprocessing',size=11.2)
    box(.365,.125,.235,.11,'Internal evaluation\nn = 416',size=12)
    box(.655,.125,.32,.11,'External validation\nn = 197',color=ORANGE,fill='#fbf2ec',size=12)
    arrow((internal_center,.462),(internal_center,.243))
    arrow((external_center,.462),(external_center,.243))
    arrow((.338,.18),(.357,.18))
    ax.plot([training_center,training_center,external_center],[.117,.075,.075],lw=1.15,color='#536575')
    arrow((external_center,.075),(external_center,.117))
    ax.text(.5,.025,'COG = MoCA <26 OR cognitive-item score ≥1; both components observed.',
            ha='center',va='center',fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    export(fig,'figure1_study_design')


def comparison_figure():
    path=ROOT/'manuscript/results/bayes100/all_model_performance.csv'
    performance=pd.read_csv(path)
    names=['Age_LR','Basic_LR','LR','SVM','RF','XGBoost','CatBoost','NN']
    assert len(performance)==16 and set(performance.model)==set(names)
    fig=plt.figure(figsize=(8.6,6.0))
    grid=fig.add_gridspec(1,2,width_ratios=[1.5,1.25],left=.17,right=.99,bottom=.19,top=.9,wspace=.32)
    ax=fig.add_subplot(grid[0,0]);age_ax=fig.add_subplot(grid[0,1])
    y=np.arange(len(names))[::-1]
    for value in y[:2]:
        ax.axhspan(value-.44,value+.44,color='#eef2f5',zorder=0)
    for cohort,color,offset in [('PPMI_internal',BLUE,.15),('FAHHAUST_external',ORANGE,-.15)]:
        ordered=performance.loc[performance.cohort.eq(cohort)].set_index('model').loc[names]
        ax.errorbar(ordered.AUROC,y+offset,
                    xerr=np.vstack([ordered.AUROC-ordered.AUROC_low,ordered.AUROC_high-ordered.AUROC]),
                    fmt='o',ms=4.6,color=color,elinewidth=1.15,capsize=2.2,capthick=1,zorder=3)
    ax.axvline(.5,lw=.9,ls='--',color='#8b959c',zorder=1)
    ax.set_yticks(y,['Age-only LR','Basic LR (5)','LR','SVM','RF','XGBoost','CatBoost','NN'],fontsize=10.5)
    ax.set(xlim=(.45,.85),ylim=(-.7,len(names)-.2),xlabel='AUROC (95% CI)')
    ax.set_xticks([.5,.6,.7,.8]);ax.grid(axis='x',color='#e6eaed',lw=.65,zorder=0)
    ax.set_title('A. Discrimination',loc='left',fontsize=12,weight='bold',pad=12)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=7)
    handles=[Line2D([0],[0],color=BLUE,marker='o',lw=1,label='Internal PPMI (n=416)'),
             Line2D([0],[0],color=ORANGE,marker='o',lw=1,label='FAHHAUST (n=197)')]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.05,-.16),frameon=False,fontsize=9.7,handlelength=1.5)
    ppmi,_=clean_sheet(pd.read_csv(ROOT/'data/derived/cog_pm3/ppmi_cog.csv'))
    external,_=clean_sheet(pd.read_csv(ROOT/'data/derived/cog_v1/fahhaust.csv'))
    split=np.load(ROOT/'data/derived/cog_pm3/frozen_split.npz')
    internal=ppmi.set_index('PATNO').loc[split['internal_id']]
    summary=[];arrays=[]
    for cohort,d in [('PPMI_internal',internal),('FAHHAUST_external',external)]:
        for cog in [0,1]:
            a=d.loc[d.COG.eq(cog),'age'].dropna().to_numpy()
            arrays.append(a)
            summary.append(dict(cohort=cohort,COG=cog,n=int(len(a)),mean=float(a.mean()),
                                median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75))))
    tests=[]
    for i,cohort in enumerate(['PPMI_internal','FAHHAUST_external']):
        negative,positive=arrays[2*i:2*i+2]
        result=ttest_ind(positive,negative,equal_var=False,alternative='two-sided')
        tests.append(dict(cohort=cohort,n_negative=len(negative),n_positive=len(positive),
                          mean_negative=float(negative.mean()),mean_positive=float(positive.mean()),
                          mean_difference=float(positive.mean()-negative.mean()),
                          statistic=float(result.statistic),df=float(result.df),p_raw=float(result.pvalue)))
    adjusted=multipletests([r['p_raw'] for r in tests],method='holm')[1]
    for row,p in zip(tests,adjusted): row['p_holm']=float(p)
    pd.DataFrame(tests).to_csv(OUT/'figure2_age_tests.csv',index=False)
    positions=[.8,1.6,3.0,3.8]
    bp=age_ax.boxplot(arrays,positions=positions,widths=.52,patch_artist=True,showfliers=False,
                      medianprops=dict(color=INK,lw=1.6),whiskerprops=dict(lw=1),capprops=dict(lw=1))
    for patch,color,alpha in zip(bp['boxes'],[BLUE,BLUE,ORANGE,ORANGE],[.30,.72,.30,.72]):
        patch.set(facecolor=color,edgecolor=color,alpha=alpha,linewidth=1.1)
    max_age=max(float(a.max()) for a in arrays)
    lower=5*np.floor(min(float(a.min()) for a in arrays)/5)
    upper=5*np.ceil(max_age/5)+18
    age_ax.set(ylim=(lower,upper),xlim=(.2,4.4),ylabel='Age, years')
    age_ax.set_xticks(positions,['COG−','COG+','COG−','COG+'],fontsize=10.2)
    age_ax.text(1.2,-.095,'Internal\nPPMI',transform=age_ax.get_xaxis_transform(),ha='center',va='top',fontsize=10.2)
    age_ax.text(3.4,-.095,'External\nFAHHAUST',transform=age_ax.get_xaxis_transform(),ha='center',va='top',fontsize=10.2)
    age_ax.set_title('B. Age by COG status',loc='left',fontsize=12,weight='bold',pad=12)
    age_ax.grid(axis='y',color='#e6eaed',lw=.65);age_ax.set_axisbelow(True)
    gap_internal=summary[1]['mean']-summary[0]['mean'];gap_external=summary[3]['mean']-summary[2]['mean']
    for x,gap in [(1.2,gap_internal),(3.4,gap_external)]:
        age_ax.text(x,upper-2,f'Mean gap\n{gap:.2f} years',ha='center',va='top',fontsize=10.2,linespacing=1.35)
    bracket_y=max_age+3.5
    for (left,right),row in zip([positions[:2],positions[2:]],tests):
        age_ax.plot([left,left,right,right],[bracket_y-.9,bracket_y,bracket_y,bracket_y-.9],
                    color=INK,lw=.9)
        label='P < 0.001' if row['p_holm']<.001 else f"P = {row['p_holm']:.3f}"
        age_ax.text((left+right)/2,bracket_y+1.1,label,ha='center',va='bottom',fontsize=9.5)
    assert np.isclose(gap_internal,2.34105090311988) and np.isclose(gap_external,8.06863387978142)
    assert sum(r['n'] for r in summary if r['cohort']=='PPMI_internal')==413
    assert sum(r['n'] for r in summary if r['cohort']=='FAHHAUST_external')==197
    pd.DataFrame(summary).to_csv(OUT/'figure2_age_summary.csv',index=False)
    export(fig,'figure2_performance_case_mix')


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'manuscript/results/bayes100/all_model_performance.csv',ROOT/'data/derived/cog_pm3/ppmi_cog.csv',
             ROOT/'data/derived/cog_v1/fahhaust.csv',ROOT/'data/derived/cog_pm3/frozen_split.npz']
    before={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    flow_figure();comparison_figure()
    assert all(sha(ROOT/rel)==expected for rel,expected in before.items())
    write_json(OUT/'figure_manifest.json',dict(status='generated',code_sha256=sha(__file__),
                input_sha256=before,model_refits=0,new_statistical_tests=2,
                age_tests_sha256=sha(OUT/'figure2_age_tests.csv'),
                figure2_points=16,figure2_age_observations=610,font_family=FONT_FAMILY,
                output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(FIG.glob('*'))
                               if p.suffix.lower() in {'.pdf','.png'}}))
    print(json.dumps(dict(event='MANUSCRIPT_FIGURES_COMPLETE',figures=2,model_refits=0)),flush=True)


if __name__=='__main__':
    main()
