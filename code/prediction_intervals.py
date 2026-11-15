"""Calibration bootstrap and Wilson intervals using frozen evaluation predictions."""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
import time
import warnings

for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.special import logit
from calibration_likelihood import fit_calibration
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/results/bayes100'
PRIVATE=ROOT/'data/derived/bayes100'
NAMES=['Age_LR','Basic_LR','LR','SVM','RF','XGBoost','CatBoost','NN']
SEED=20261014
N_BOOT=2000



def classification(y,p,threshold):
    positive=p>=threshold
    tp=int(np.sum(positive & (y==1)));tn=int(np.sum(~positive & (y==0)))
    n=len(y);n1=int(y.sum());n0=n-n1
    values=dict(n=n,n_positive=n1,n_negative=n0,tp=tp,tn=tn,fp=n0-tn,fn=n1-tp)
    for key,k,den in [('accuracy',tp+tn,n),('sensitivity',tp,n1),('specificity',tn,n0)]:
        lo,hi=proportion_confint(k,den,alpha=.05,method='wilson')
        values.update({key:k/den,key+'_low':float(lo),key+'_high':float(hi)})
    return values


def main():
    started=time.perf_counter()
    old_cal=pd.read_csv(OUT/'calibration.csv').set_index(['cohort','model'])
    old_metrics=pd.read_csv(OUT/'metrics.csv').set_index(['cohort','model','operating_point'])
    packets={}
    for name in NAMES:
        with np.load(PRIVATE/(name+'.npz')) as saved:
            packets[name]={key:saved[key].copy() for key in ['yi','pi','ye','p','internal_id','external_id']}
    reference=packets[NAMES[0]]
    for packet in packets.values():
        for key in ['yi','ye','internal_id','external_id']:
            assert np.array_equal(packet[key],reference[key]),key
    cal_rows=[];class_rows=[];failures=[]
    validation=[]
    rng=np.random.default_rng(SEED)
    for cohort,yk,pk,n,n1 in [('PPMI_internal','yi','pi',416,204),
                               ('FAHHAUST_external','ye','p',197,122)]:
        y=reference[yk].astype(float)
        assert len(y)==n and y.sum()==n1 and set(y)=={0,1}
        # Ordinary participant bootstrap; counts equal an expanded resample.
        indices=rng.integers(0,n,size=(N_BOOT,n))
        counts=np.array([np.bincount(row,minlength=n) for row in indices])
        for name in NAMES:
            p=packets[name][pk]
            assert len(p)==n and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            z=logit(np.clip(p,1e-6,1-1e-6));center=float(z.mean());scale=float(z.std())
            assert scale>0
            point=fit_calibration(y,z,np.ones(n),center,scale)
            retained=old_cal.loc[(cohort,name)]
            # Summary and bootstrap use the identical stable likelihood parameterization.
            assert abs(point[0]-retained.calibration_intercept_offset)<1e-5,(cohort,name,'offset')
            assert abs(point[1]-retained.calibration_slope)<.002,(cohort,name,'slope',point[1],retained.calibration_slope)
            draws=np.full((N_BOOT,3),np.nan)
            for b,weights in enumerate(counts):
                try:
                    draws[b]=fit_calibration(y,z,weights,center,scale)
                except (RuntimeError,ValueError,np.linalg.LinAlgError,PerfectSeparationWarning) as error:
                    failures.append(dict(cohort=cohort,model=name,replicate=b,error=str(error)))
            # Check weighted bootstrap likelihood against explicit repeated rows.
            for b in [0,N_BOOT-1]:
                expanded=fit_calibration(y[indices[b]],z[indices[b]],np.ones(n),center,scale)
                assert np.allclose(draws[b],expanded,rtol=1e-6,atol=1e-6)
            finite=np.isfinite(draws).all(axis=1)
            if not finite.all():
                (OUT/'uncertainty_fit_failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
                raise RuntimeError('Unestimable calibration resamples recorded; no intervals published')
            low,high=np.quantile(draws,[.025,.975],axis=0)
            cal_rows.append(dict(cohort=cohort,model=name,
                calibration_intercept_offset=float(retained.calibration_intercept_offset),
                calibration_slope=float(retained.calibration_slope),
                offset_low=low[0],offset_high=high[0],slope_low=low[1],slope_high=high[1],
                calibration_intercept=float(retained.calibration_intercept),
                intercept_low=low[2],intercept_high=high[2],
                bootstrap_attempts=N_BOOT,bootstrap_successes=int(finite.sum())))
            for operating in ['OOF_Youden','fixed_0.5']:
                original=old_metrics.loc[(cohort,name,operating)]
                values=classification(y,p,float(original.threshold))
                for metric in ['accuracy','sensitivity','specificity']:
                    assert abs(values[metric]-original[metric])<1e-12,(cohort,name,operating,metric)
                class_rows.append(dict(cohort=cohort,model=name,operating_point=operating,
                                       threshold=float(original.threshold),**values))
            validation.append(dict(cohort=cohort,model=name,
                offset_solver_difference=float(point[0]-retained.calibration_intercept_offset),
                slope_solver_difference=float(point[1]-retained.calibration_slope),
                bootstrap_successes=int(finite.sum())))
            print(json.dumps(dict(cohort=cohort,model=name,completed_bootstrap=N_BOOT,
                                  seconds=round(time.perf_counter()-started,1))),flush=True)
    assert len(cal_rows)==16 and len(class_rows)==32 and not failures
    pd.DataFrame(cal_rows).to_csv(OUT/'calibration_intervals.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'classification_intervals.csv',index=False)
    record=dict(status='passed',seed=SEED,calibration_bootstrap='ordinary participant percentile',
        bootstrap_attempts_per_fit=N_BOOT,calibration_rows=16,classification_rows=32,
        failed_bootstrap_fits=0,classification_method='Wilson score, two-sided 95%',
        prediction_models_refitted=0,threshold_updates=0,input_updates=0,
        original_point_estimates_preserved=True,weighted_and_expanded_bootstrap_agree=True,
        calibration_point_checks=validation,
        evaluation_sizes=dict(PPMI_internal=416,FAHHAUST_external=197),
        software=dict(python=sys.version.split()[0],
                      **{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','statsmodels']}),
        seconds=time.perf_counter()-started)
    (OUT/'uncertainty_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=record['status'],seconds=record['seconds'])),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path)
    parser.add_argument('--prediction-dir',type=Path)
    args=parser.parse_args()
    if args.results_dir: OUT=args.results_dir.resolve()
    if args.prediction_dir: PRIVATE=args.prediction_dir.resolve()
    with threadpool_limits(limits=1):main()
