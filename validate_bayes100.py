"""Independent checks of completed searches, frozen participants and reported metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from common_preprocessing import sha,write_json,youden

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/bayes100'
PRIVATE=ROOT/'data/derived/bayes100'
INPUT=ROOT/'data/derived/bayes100_input'
NAMES=['Age_LR','Basic_LR','LR','SVM','RF','XGBoost','CatBoost','NN']


def main():
    split=np.load(ROOT/'data/derived/cog_pm3/frozen_split.npz')
    external=pd.read_csv(ROOT/'data/derived/cog_v1/fahhaust.csv')
    manifest=json.loads((INPUT/'input_manifest.json').read_text())
    metrics=pd.read_csv(OUT/'metrics.csv');absolute=pd.read_csv(OUT/'absolute_intervals.csv')
    paired=pd.read_csv(OUT/'paired_lr.csv');cal=pd.read_csv(OUT/'calibration.csv')
    assert (len(metrics),len(absolute),len(paired),len(cal))==(32,12,12,16)
    assert set(metrics.model)==set(NAMES)
    distinct={};points={};selections={}
    for name in NAMES:
        r=json.loads((OUT/(name+'.json')).read_text())
        assert r['model']==name and r['status']=='completed' and r['trials_completed']==100
        assert r['class_weight'] is None and r['resampling'] is None
        assert r['n_features']==(1 if name=='Age_LR' else 5 if name=='Basic_LR' else 21)
        assert r['signature_inputs']['code']==sha(ROOT/'bayes100.py')
        assert r['signature_inputs']['preprocessing']==sha(ROOT/'common_preprocessing.py')
        assert r['signature_inputs']['body_rules']==sha(ROOT/'body_rules.py')
        assert r['signature_inputs']['inputs']==manifest
        signature=hashlib.sha256(json.dumps(r['signature_inputs'],sort_keys=True).encode()).hexdigest()
        assert signature==r['signature']
        trials=r['tuning'];assert len(trials)==100
        assert len(set(t['number'] for t in trials))==100
        assert all(len(t['fold_AUROC'])==5 for t in trials)
        assert all(abs(np.mean(t['fold_AUROC'])-t['value'])<1e-12 for t in trials)
        best=next(t for t in trials if t['number']==r['best_trial'])
        assert best['params']==r['params'] and best['value']==max(t['value'] for t in trials)==r['selection_score']
        distinct[name]=len(set(json.dumps(t['params'],sort_keys=True) for t in trials))
        assert r['final_refit']['n_final_training']==969 and r['final_refit']['all_training_rows_used']
        a=np.load(PRIVATE/(name+'.npz'))
        assert sha(PRIVATE/(name+'.npz'))==r['predictions_sha256']
        for key in ['train_id','internal_id','y','yi']:assert np.array_equal(a[key],split[key]),(name,key)
        assert np.array_equal(a['external_id'],external.PATNO.to_numpy())
        assert np.array_equal(a['ye'],external.COG.to_numpy(int))
        assert not np.intersect1d(a['train_id'],a['internal_id']).size
        folds=list(StratifiedKFold(5,shuffle=True,random_state=20261014).split(np.zeros((969,1)),a['y']))
        scores=[]
        for _,va in folds:
            yy,pp=a['y'][va],a['oof'][va];n1=int(yy.sum());n0=len(yy)-n1
            scores.append(float((rankdata(pp)[yy==1].sum()-n1*(n1+1)/2)/(n1*n0)))
        assert np.allclose(scores,r['fold_AUROC'],atol=1e-12,rtol=0)
        assert abs(np.mean(scores)-r['selection_score'])<1e-12
        assert abs(youden(a['y'],a['oof'])-r['oof_threshold'])<1e-12
        selections[name]=dict(trials=100,distinct_parameters=distinct[name],best_mean_AUROC=r['selection_score'],
                              final_training_rows=969,final_rounds=r['final_refit']['fixed_rounds'])
        for cohort,yk,pk in [('PPMI_internal','yi','pi'),('FAHHAUST_external','ye','p')]:
            y,p=a[yk],a[pk];n1=int(y.sum());n0=len(y)-n1
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            auc=float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0));brier=float(np.mean((p-y)**2))
            points[(cohort,name)]=dict(AUROC=auc,Brier=brier)
            for operating,cut in [('OOF_Youden',r['oof_threshold']),('fixed_0.5',.5)]:
                row=metrics.loc[metrics.cohort.eq(cohort)&metrics.model.eq(name)&metrics.operating_point.eq(operating)].iloc[0]
                pred=p>=cut
                expected=dict(AUROC=auc,Brier=brier,accuracy=float(np.mean(pred==y)),
                              sensitivity=float(pred[y==1].mean()),specificity=float((~pred[y==0]).mean()))
                assert abs(row.threshold-cut)<1e-12
                for key,value in expected.items():assert abs(row[key]-value)<1e-12,(name,cohort,key)
    for _,r in absolute.iterrows():
        assert abs(r.value-points[(r.cohort,r.model)][r.metric])<1e-12
        assert np.isfinite([r.low,r.high]).all() and r.low<=r.high
    for _,r in paired.iterrows():
        assert abs(r.difference-(points[(r.cohort,r.model)][r.metric]-points[(r.cohort,r.reference)][r.metric]))<1e-12
        assert np.isfinite([r.low,r.high]).all() and r.low<=r.high
    # A large calibration slope can reflect a compressed score scale; inspect its information.
    xgb=np.load(PRIVATE/'XGBoost.npz');z=logit(np.clip(xgb['p'],1e-6,1-1e-6))
    row=cal.loc[cal.cohort.eq('FAHHAUST_external')&cal.model.eq('XGBoost')].iloc[0]
    design=np.c_[np.ones(len(z)),z];beta=np.array([row.calibration_intercept,row.calibration_slope])
    fitted=expit(design@beta);information=design.T@((fitted*(1-fitted))[:,None]*design)
    eigen=np.linalg.eigvalsh(information);gradient=design.T@(fitted-xgb['ye'])/len(z)
    assert np.linalg.matrix_rank(design)==2 and eigen.min()>1e-8 and np.max(np.abs(gradient))<1e-5
    diagnostic=dict(score_range=[float(xgb['p'].min()),float(xgb['p'].max())],
                    score_sd=float(xgb['p'].std()),calibration_slope=float(beta[1]),
                    information_min_eigenvalue=float(eigen.min()),information_condition=float(eigen.max()/eigen.min()),
                    mean_gradient_max_abs=float(np.max(np.abs(gradient))))
    result=dict(status='passed',scope='search records, frozen inputs, OOF selection and saved-prediction numerical results',
                strategies=8,completed_trials=800,trial_fold_scores=4000,test_metric_rows=32,
                frozen_split_and_labels_unchanged=True,predictive_model_refits=0,
                selections=selections,XGBoost_external_calibration_diagnostic=diagnostic)
    write_json(OUT/'validation.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--prediction-dir',type=Path)
    parser.add_argument('--input-dir',type=Path)
    args=parser.parse_args()
    if args.results_dir: OUT=args.results_dir.resolve()
    if args.prediction_dir: PRIVATE=args.prediction_dir.resolve()
    if args.input_dir: INPUT=args.input_dir.resolve()
    main()
