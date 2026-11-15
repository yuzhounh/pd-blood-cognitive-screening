"""Independent B129 metric, paired-interval, presentation and PDF checks."""
import json,re,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from common_preprocessing import sha,write_json
from factorial_protocol import ROOT,OUT,STRATEGIES,ALGORITHMS,SEED,paths
from factorial_results import load_verified
from factorial_tables import difference_interval
M=ROOT/'manuscript'
def main():
    packets,receipts=load_verified();perf=pd.read_csv(OUT/'performance.csv');metrics=pd.read_csv(OUT/'metrics.csv');delta=pd.read_csv(OUT/'paired_differences.csv');absolute=pd.read_csv(OUT/'absolute_intervals.csv');cal=pd.read_csv(OUT/'calibration.csv');cl=pd.read_csv(OUT/'classification_intervals.csv')
    assert [len(x) for x in [perf,metrics,delta,absolute,cal,cl]]==[36,72,72,72,36,72]
    assert len(perf.groupby(['algorithm','predictor_set','cohort']))==36
    points={};rng=np.random.default_rng(SEED);ci_checks=[]
    for cohort,yk,pk in [('PPMI_internal','yi','pi'),('FAHHAUST_external','ye','p')]:
        y=packets['Age_LR'][yk];n1=int(y.sum());n0=len(y)-n1
        strata=[np.flatnonzero(y==v) for v in [0,1]]
        indices=np.c_[rng.choice(strata[0],(2000,len(strata[0])),replace=True),rng.choice(strata[1],(2000,len(strata[1])),replace=True)]
        checked_draws={}
        for spec in STRATEGIES:
            key=spec['strategy_id'];p=packets[key][pk]
            auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0));brier=float(np.mean((y-p)**2));points[(cohort,key)]={'AUROC':auc,'Brier':brier}
            r=perf.loc[perf.cohort.eq(cohort)&perf.strategy_id.eq(key)].iloc[0]
            assert abs(r.AUROC-auc)<1e-12 and abs(r.Brier-brier)<1e-12
            for operating,cut in [('OOF_Youden',receipts[key]['oof_threshold']),('fixed_0.5',.5)]:
                q=p>=cut;expected=dict(accuracy=float(np.mean(q==y)),sensitivity=float(q[y==1].mean()),specificity=float((~q[y==0]).mean()))
                row=metrics.loc[metrics.cohort.eq(cohort)&metrics.strategy_id.eq(key)&metrics.operating_point.eq(operating)].iloc[0]
                interval=cl.loc[cl.cohort.eq(cohort)&cl.strategy_id.eq(key)&cl.operating_point.eq(operating)].iloc[0]
                assert abs(row.threshold-cut)<1e-12
                for name,value in expected.items():assert abs(row[name]-value)<1e-12 and interval[name+'_low']<=value<=interval[name+'_high']
            if key in ['Basic_LR','LR','NN_Set2','NN']:
                # Different implementation: sklearn AUROC in explicit resamples.
                checked_draws[key]=np.array([[roc_auc_score(y[ix],p[ix]),np.mean((p[ix]-y[ix])**2)] for ix in indices])
        for full,reference in [('LR','Basic_LR'),('NN','NN_Set2')]:
            bounds=np.quantile(checked_draws[full]-checked_draws[reference],[.025,.975],axis=0)
            for j,metric in enumerate(['AUROC','Brier']):
                row=delta.loc[delta.cohort.eq(cohort)&delta.strategy_id.eq(full)&delta.reference_id.eq(reference)&delta.metric.eq(metric)].iloc[0]
                assert np.allclose([row.low,row.high],bounds[:,j],rtol=0,atol=1e-12)
                ci_checks.append(dict(cohort=cohort,algorithm=full,metric=metric))
    for r in delta.itertuples():
        expected=points[(r.cohort,r.strategy_id)][r.metric]-points[(r.cohort,r.reference_id)][r.metric]
        assert abs(r.difference-expected)<1e-12 and r.low<=r.high
    assert (cal.joint_bootstrap_successes==2000).all() and (cal.offset_bootstrap_successes==2000).all()
    primary=delta.loc[delta.primary_comparison]
    extauc=primary.loc[primary.cohort.eq('FAHHAUST_external')&primary.metric.eq('AUROC')]
    assert (extauc.difference<0).all() and set(extauc.loc[extauc.high<0,'algorithm'])=={'LR','SVM','XGBoost','NN'}
    extb=primary.loc[primary.cohort.eq('FAHHAUST_external')&primary.metric.eq('Brier')]
    assert set(extb.loc[extb.high<0,'algorithm'])=={'LR','RF','CatBoost'}
    assert set(extb.loc[extb.low>0,'algorithm'])=={'XGBoost','NN'}
    # Tables must present the complete result collection, with correct rounding.
    table2=(M/'tables/table2.tex').read_text();table3=(M/'tables/table3.tex').read_text()
    for r in perf.loc[perf.predictor_set.eq('Full set')].itertuples():
        for k,d in [('AUROC',3),('AUROC_low',3),('AUROC_high',3),('Brier',4),('Brier_low',4),('Brier_high',4)]:assert f'{getattr(r,k):.{d}f}' in table2
    for r in primary.itertuples():
        d=3 if r.metric=='AUROC' else 4
        assert difference_interval(r.difference,r.low,r.high,d) in table3
    expected_table_rows={'factorial_search':6,'factorial_selected':18,'factorial_performance_internal':18,'factorial_performance_external':18,'factorial_calibration':36,'factorial_classification_internal':36,'factorial_classification_external':36,'factorial_secondary':24,'factorial_anthropometry':16}
    for name,n in expected_table_rows.items():assert len(pd.read_csv(OUT/(name+'_formatted.csv')))==n,(name,n)
    fig=json.loads((OUT/'figure_manifest.json').read_text());assert len(fig['ROC_checks'])==36
    assert all(sha(ROOT/rel)==digest for rel,digest in fig['output_sha256'].items())
    for r in fig['ROC_checks']:assert abs(r['AUROC']-points[(r['cohort'],r['strategy_id'])]['AUROC'])<1e-12
    old=json.loads((ROOT/'manuscript/results/bayes100/calibration_figure_manifest.json').read_text())
    assert all(sha(ROOT/rel)==digest for rel,digest in old['output_sha256'].items())
    bins=pd.read_csv(OUT/'supplementary_calibration_bins.csv');assert len(bins.groupby(['cohort','strategy_id']))==30
    assert all(n in [416,197] for n in bins.groupby(['cohort','strategy_id']).n.sum())
    methods=(M/'sections/methods.tex').read_text();results=(M/'sections/results.tex').read_text();discussion=(M/'sections/discussion.tex').read_text();abstract=(M/'sections/abstract.tex').read_text();conclusion=(M/'sections/conclusion.tex').read_text()
    assert '18 strategies' in methods and '10 startup trials' in methods and 'other ten reduced-input' in methods
    assert 'after the initial held-out results had been examined' in methods
    assert 'not treated as an independent nested estimate' in methods and 'without a paired effect test' in methods
    for r in extauc.itertuples():
        for k in ['difference','low','high']:assert f'{getattr(r,k):.3f}' in results
    for r in extb.itertuples():
        for k in ['difference','low','high']:assert f'{getattr(r,k):.4f}' in results
    assert 'all six algorithms' in abstract and 'all six algorithms' in conclusion
    assert 'not each paired with a basic-input counterpart' not in discussion
    assert 'Table~S9' in discussion and 'Tables~S7--S8' in discussion
    build=json.loads((OUT/'build_validation.json').read_text())
    assert all(sha(ROOT/rel)==digest for rel,digest in build['source_sha256'].items())
    for name,record in build['documents'].items():
        assert sha(ROOT/record['pdf'])==record['sha256'] and record['compiled_without_errors_or_undefined_references']
        assert not record['layout_warnings']
        text=(ROOT/f'tmp/latex/{name}/extracted.txt').read_text();aux=(ROOT/f'tmp/latex/{name}/{name}.aux').read_text()
        labels=set(re.findall(r'\\newlabel\{tab:[^}]+\}\{\{(S?[^}]+)\}',aux))
        if name=='manuscript':
            assert labels=={'1','2','3','4'},labels
            assert all(x in text for x in ['18','0.607','0.687','0.701','0.728','0.0163'])
            assert set(re.findall(r'^\s*Figure (\d+)\.',text,re.M))=={'1','2','3','4'}
        else:
            assert labels=={'S1','S2A','S2B','S3A','S3B','S4','S5A','S5B','S6','S7','S8','S9'},labels
            assert set(re.findall(r'^\s*Figure S(\d+)\.',text,re.M))=={'1','2','3','4','5'}
    record=dict(status='passed',strategies=18,reused_strategies=8,new_strategies=10,total_trial_records=1800,new_trials=1000,performance_rows=36,paired_rows=72,classification_rows=72,calibration_rows=36,all_calibration_bootstrap_fits_estimable=True,independent_paired_CI_checks=ci_checks,baseline_both_conditions_unchanged=True,main_tables=4,main_figures=4,supplementary_table_groups=9,supplementary_figures=5,source_results_figures_and_PDFs_consistent=True)
    write_json(OUT/'manuscript_consistency.json',record);print(json.dumps(record,indent=2))
if __name__=='__main__':main()
