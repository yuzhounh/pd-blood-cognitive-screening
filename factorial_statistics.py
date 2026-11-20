"""Same-cohort performance, within-algorithm increments, and calibration across all 18 strategies."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,time,warnings,sys,importlib.metadata as md
import numpy as np
import pandas as pd
from scipy.special import logit
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from threadpoolctl import threadpool_limits
from common_preprocessing import write_json,sha
from factorial_protocol import ROOT,OUT,STRATEGIES,ALGORITHMS,SEED
from factorial_results import load_verified
from finalize_bayes100 import measures,bootstrap
from prediction_intervals import classification
from calibration_likelihood import fit_calibration
N_BOOT=2000

def calibration_draw(y,z,weights,center,scale):
    if np.ptp(z)==0:
        fraction=float(np.average(y,weights=weights))
        if not 0<fraction<1:raise ValueError('Bootstrap contains one outcome class')
        return np.array([float(logit(fraction)-z[0]),np.nan,np.nan]),'Constant predicted log odds; slope and free intercept not identifiable'
    try:return fit_calibration(y,z,weights,center,scale),''
    except (RuntimeError,ValueError,np.linalg.LinAlgError,PerfectSeparationWarning) as error:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',PerfectSeparationWarning)
                fit=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=z,freq_weights=weights).fit(maxiter=100,tol=1e-10)
            offset=float(fit.params[0]) if fit.converged else np.nan
        except Exception:offset=np.nan
        return np.array([offset,np.nan,np.nan]),str(error)

def main():
    started=time.time();packets,receipts=load_verified()
    performance=[];metrics=[];absolute=[];differences=[];calibration=[];classes=[];failures=[]
    perf_rng=np.random.default_rng(SEED);cal_rng=np.random.default_rng(SEED)
    old_cal=pd.read_csv(ROOT/'results/bayes100/calibration_intervals.csv').set_index(['cohort','model'])
    for cohort,yk,pk,n,n1 in [('PPMI_internal','yi','pi',416,204),('FAHHAUST_external','ye','p',197,122)]:
        y=packets['Age_LR'][yk];assert len(y)==n and int(y.sum())==n1
        strata=[np.flatnonzero(y==v) for v in [0,1]]
        indices=np.c_[perf_rng.choice(strata[0],(N_BOOT,len(strata[0])),replace=True),perf_rng.choice(strata[1],(N_BOOT,len(strata[1])),replace=True)]
        ordinary=cal_rng.integers(0,n,size=(N_BOOT,n));weights=np.array([np.bincount(row,minlength=n) for row in ordinary])
        draws={};points={}
        for spec in STRATEGIES:
            key=spec['strategy_id'];p=packets[key][pk];r=receipts[key]
            meta=dict(cohort=cohort,strategy_id=key,algorithm=spec['algorithm'],predictor_set=spec['predictor_set'],n_features=spec['n_features'],reused=spec['reused'])
            cut=r['oof_threshold'];point=measures(y,p,cut);points[key]=point
            draws[key]=bootstrap(y,p,indices);bounds=np.quantile(draws[key],[.025,.975],axis=0)
            performance.append(dict(**meta,threshold=cut,**point,AUROC_low=bounds[0,0],AUROC_high=bounds[1,0],Brier_low=bounds[0,1],Brier_high=bounds[1,1]))
            for j,name in enumerate(['AUROC','Brier']):absolute.append(dict(**meta,metric=name,value=point[name],low=bounds[0,j],high=bounds[1,j]))
            for operating,threshold in [('OOF_Youden',cut),('fixed_0.5',.5)]:
                metrics.append(dict(**meta,operating_point=operating,threshold=threshold,**measures(y,p,threshold)))
                classes.append(dict(**meta,operating_point=operating,threshold=threshold,**classification(y,p,threshold)))
            z=logit(np.clip(p,1e-6,1-1e-6));center=float(z.mean());scale=float(z.std())
            estimate,note=calibration_draw(y,z,np.ones(n),center,scale)
            if spec['reused']:
                old=old_cal.loc[(cohort,key)]
                assert np.allclose(estimate,[old.calibration_intercept_offset,old.calibration_slope,old.calibration_intercept],rtol=0,atol=1e-8,equal_nan=True)
                low=np.array([old.offset_low,old.slope_low,old.intercept_low]);high=np.array([old.offset_high,old.slope_high,old.intercept_high])
                success=np.array([2000,2000,2000]);source='v3.2 identical ordinary-bootstrap protocol and frozen predictions'
            else:
                samples=np.full((N_BOOT,3),np.nan)
                for b,w in enumerate(weights):
                    samples[b],reason=calibration_draw(y,z,w,center,scale)
                    if reason and np.ptp(z)>0:failures.append(dict(cohort=cohort,strategy_id=key,replicate=b,reason=reason))
                success=np.isfinite(samples).sum(axis=0)
                low=np.full(3,np.nan);high=np.full(3,np.nan)
                for j in range(3):
                    if success[j]==N_BOOT:low[j],high[j]=np.quantile(samples[:,j],[.025,.975])
                if np.ptp(z)>0:
                    for b in [0,N_BOOT-1]:
                        expanded,_=calibration_draw(y[ordinary[b]],z[ordinary[b]],np.ones(n),center,scale)
                        assert np.allclose(samples[b],expanded,rtol=1e-6,atol=1e-6,equal_nan=True)
                source='new ordinary-bootstrap estimates'
                if success[1]<N_BOOT and not note:note='At least one joint bootstrap fit failed; joint intervals withheld'
            calibration.append(dict(**meta,mean_probability=float(p.mean()),observed_fraction=float(y.mean()),score_min=float(p.min()),score_max=float(p.max()),calibration_intercept_offset=estimate[0],calibration_slope=estimate[1],calibration_intercept=estimate[2],offset_low=low[0],offset_high=high[0],slope_low=low[1],slope_high=high[1],intercept_low=low[2],intercept_high=high[2],bootstrap_attempts=N_BOOT,offset_bootstrap_successes=int(success[0]),joint_bootstrap_successes=int(success[1]),calibration_note=note,interval_source=source))
            print(json.dumps(dict(cohort=cohort,strategy=key,joint_successes=int(success[1]),seconds=round(time.time()-started,1))),flush=True)
        for algorithm in ALGORITHMS:
            keys=[s['strategy_id'] for s in STRATEGIES if s['algorithm']==algorithm]
            for full,reference,label in [(keys[2],keys[1],'Full set - Set 2'),(keys[1],keys[0],'Set 2 - Set 1'),(keys[2],keys[0],'Full set - Set 1')]:
                bounds=np.quantile(draws[full]-draws[reference],[.025,.975],axis=0)
                for j,metric in enumerate(['AUROC','Brier']):
                    differences.append(dict(cohort=cohort,algorithm=algorithm,comparison=label,strategy_id=full,reference_id=reference,metric=metric,difference=points[full][metric]-points[reference][metric],low=bounds[0,j],high=bounds[1,j],primary_comparison=label=='Full set - Set 2'))
    frames={'performance.csv':performance,'metrics.csv':metrics,'absolute_intervals.csv':absolute,'paired_differences.csv':differences,'calibration.csv':calibration,'classification_intervals.csv':classes}
    for name,rows in frames.items():pd.DataFrame(rows).to_csv(OUT/name,index=False)
    assert list(map(len,[performance,metrics,absolute,differences,calibration,classes]))==[36,72,72,72,36,72]
    write_json(OUT/'calibration_failures.json',failures)
    record=dict(status='computed',strategies=18,new_trials=1000,reused_trials=800,performance_rows=36,classification_rows=72,calibration_rows=36,absolute_interval_rows=72,paired_interval_rows=72,bootstrap_replicates=2000,calibration_interval_rule='Report interval only when all 2000 fits for that parameter succeeded; failures are not discarded',calibration_nonidentified_or_incomplete_rows=sum(r['joint_bootstrap_successes']<2000 for r in calibration),new_calibration_failures=len(failures),software=dict(python=sys.version.split()[0],**{n:md.version(n) for n in ['numpy','pandas','scipy','scikit-learn','statsmodels']}),seconds=time.time()-started)
    write_json(OUT/'statistics_validation.json',record);print(json.dumps(record),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
