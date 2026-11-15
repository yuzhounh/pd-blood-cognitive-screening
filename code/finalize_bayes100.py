"""Summarize frozen Bayesian-search predictions; no fitting of predictive models."""
from pathlib import Path
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize,minimize_scalar
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from threadpoolctl import threadpool_limits
from common_preprocessing import sha,write_json,youden

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/results/bayes100'
PRIVATE=ROOT/'data/derived/bayes100'
NAMES=['Age_LR','Basic_LR','LR','SVM','RF','XGBoost','CatBoost','NN']
SEED=20261014


def measures(y,p,cut):
    tn,fp,fn,tp=confusion_matrix(y,p>=cut,labels=[0,1]).ravel()
    return dict(AUROC=float(roc_auc_score(y,p)),AUPRC=float(average_precision_score(y,p)),
                Brier=float(brier_score_loss(y,p)),accuracy=float((tp+tn)/len(y)),
                sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),
                balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2))


def calibration(y,p):
    from calibration_likelihood import fit_calibration
    z=logit(np.clip(p,1e-6,1-1e-6))
    base=dict(mean_probability=float(p.mean()),observed_fraction=float(y.mean()))
    if float(z.std()) == 0:
        raise ValueError('Calibration slope is not identifiable for constant log odds')
    offset,slope,intercept=fit_calibration(y,z,np.ones(len(y)),float(z.mean()),float(z.std()))
    return dict(**base,calibration_intercept_offset=offset,calibration_fit_success=True,
                calibration_intercept=intercept,calibration_slope=slope,
                calibration_note='Centered/scaled unpenalized binomial likelihood')


def bootstrap(y,p,indices):
    yy=y[indices];pp=p[indices]
    n1=int(y.sum());n0=len(y)-n1
    auc=(np.sum(rankdata(pp,axis=1)*yy,axis=1)-n1*(n1+1)/2)/(n1*n0)
    brier=np.mean((pp-yy)**2,axis=1)
    for k in [0,1,2]:
        assert abs(auc[k]-roc_auc_score(yy[k],pp[k]))<1e-12
        assert abs(brier[k]-brier_score_loss(yy[k],pp[k]))<1e-12
    return np.c_[auc,brier]


def main():
    assert json.loads((OUT/'completion.json').read_text())['status']=='completed'
    packets={};records={}
    for name in NAMES:
        r=json.loads((OUT/(name+'.json')).read_text())
        assert r['status']=='completed' and r['trials_completed']==100
        assert r['n_train']==969 and r['n_features']==(1 if name=='Age_LR' else 5 if name=='Basic_LR' else 21)
        assert r['class_weight'] is None and r['resampling'] is None
        assert r['final_refit']['all_training_rows_used'] and r['final_refit']['n_final_training']==969
        assert sha(PRIVATE/(name+'.npz'))==r['predictions_sha256']
        assert len(r['tuning'])==100 and r['selection_score']==max(t['value'] for t in r['tuning'])
        assert all(abs(np.mean(t['fold_AUROC'])-t['value'])<1e-12 for t in r['tuning'])
        a=np.load(PRIVATE/(name+'.npz'));packets[name]=a;records[name]=r
        assert abs(youden(a['y'],a['oof'])-r['oof_threshold'])<1e-12
        for key in ['train_id','y','internal_id','yi','external_id','ye']:
            assert np.array_equal(a[key],packets[NAMES[0]][key]),(name,key)
    rng=np.random.default_rng(SEED);metrics=[];cal=[];absolute=[];paired=[]
    for cohort,yk,pk,n,positives in [('PPMI_internal','yi','pi',416,204),('FAHHAUST_external','ye','p',197,122)]:
        y=packets[NAMES[0]][yk]
        assert len(y)==n and int(y.sum())==positives
        strata=[np.flatnonzero(y==k) for k in [0,1]]
        indices=np.c_[rng.choice(strata[0],(2000,len(strata[0])),replace=True),
                      rng.choice(strata[1],(2000,len(strata[1])),replace=True)]
        draws={};points={}
        for name in NAMES:
            p=packets[name][pk];cut=records[name]['oof_threshold']
            draws[name]=bootstrap(y,p,indices);intervals=np.quantile(draws[name],[.025,.975],axis=0)
            points[name]=measures(y,p,cut)
            for operating,threshold in [('OOF_Youden',cut),('fixed_0.5',.5)]:
                metrics.append(dict(cohort=cohort,model=name,n_features=records[name]['n_features'],
                                    operating_point=operating,threshold=threshold,**measures(y,p,threshold),
                                    AUROC_low=intervals[0,0],AUROC_high=intervals[1,0],
                                    Brier_low=intervals[0,1],Brier_high=intervals[1,1]))
            cal.append(dict(cohort=cohort,model=name,**points[name],**calibration(y,p)))
            if name in ['Age_LR','Basic_LR','LR']:
                for j,metric in enumerate(['AUROC','Brier']):
                    absolute.append(dict(cohort=cohort,model=name,metric=metric,value=points[name][metric],
                                         low=intervals[0,j],high=intervals[1,j],source='bayes100_saved_prediction_bootstrap'))
        for model,ref in [('Basic_LR','Age_LR'),('LR','Age_LR'),('LR','Basic_LR')]:
            intervals=np.quantile(draws[model]-draws[ref],[.025,.975],axis=0)
            for j,metric in enumerate(['AUROC','Brier']):
                paired.append(dict(cohort=cohort,model=model,reference=ref,metric=metric,
                                   difference=points[model][metric]-points[ref][metric],
                                   low=intervals[0,j],high=intervals[1,j],source='bayes100_saved_prediction_bootstrap'))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(metrics).query("operating_point=='OOF_Youden'").to_csv(OUT/'all_model_performance.csv',index=False)
    pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False)
    pd.DataFrame(absolute).to_csv(OUT/'absolute_intervals.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_lr.csv',index=False)
    write_json(OUT/'summary_validation.json',dict(status='computed_and_numerically_checked',strategies=8,
               main_algorithms=6,trials_per_strategy=100,training_rows=969,metrics_rows=len(metrics),
               calibration_rows=len(cal),absolute_intervals=len(absolute),paired_intervals=len(paired),
               bootstrap_draws=2000,predictive_models_refitted_by_this_script=0,
               inputs={str((PRIVATE/(n+'.npz')).relative_to(ROOT)):sha(PRIVATE/(n+'.npz')) for n in NAMES}))
    print(pd.DataFrame(metrics).query("operating_point=='OOF_Youden'")[['cohort','model','AUROC','AUPRC','Brier','accuracy','sensitivity','specificity']].to_string(index=False))
    print(json.dumps(dict(event='BAYES100_SUMMARY_COMPUTED',models=8,metrics_rows=len(metrics))),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--prediction-dir',type=Path)
    args=parser.parse_args()
    if args.results_dir: OUT=args.results_dir.resolve()
    if args.prediction_dir: PRIVATE=args.prediction_dir.resolve()
    with threadpool_limits(limits=1):main()
