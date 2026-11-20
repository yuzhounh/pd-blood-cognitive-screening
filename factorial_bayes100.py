"""Train the ten reduced-input strategies alongside the eight baseline fits."""
from pathlib import Path
import argparse,functools,gc,hashlib,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import bayes100 as base
from common_preprocessing import sha,write_json,youden
from factorial_protocol import ROOT,INPUT,OUT,PRIVATE,WORK,BY_ID,NEW,SEED,N_TRIALS
from factorial_preprocessing import SubsetPreprocessor

def configure(n):
    base.CappedImputer=functools.partial(SubsetPreprocessor,sex_index=1 if n>=5 else None)
    base.feature_indices=lambda unused:list(range(n))

def signature(spec):
    inputs=json.loads((INPUT/'input_manifest.json').read_text())
    for name,digest in inputs['files_sha256'].items():assert sha(INPUT/name)==digest
    record=dict(strategy=spec,input_manifest=inputs,packages=base.package_versions(),seed=SEED,trials=N_TRIALS,
        code_sha256={n:sha(ROOT/n) for n in ['bayes100.py','common_preprocessing.py','body_rules.py','factorial_protocol.py','factorial_preprocessing.py','factorial_bayes100.py']},
        search='baseline algorithm-specific space, independent TPE per strategy',selection='mean_five_fold_AUROC')
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),record

def run_strategy(key):
    spec=BY_ID[key];assert key in NEW and not spec['reused']
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    algorithm=spec['algorithm'];n=spec['n_features'];configure(n)
    sig,record=signature(spec);receipt=OUT/(key+'.json');prediction=PRIVATE/(key+'.npz')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old.get('signature')==sig and old.get('status')=='completed':
            assert sha(prediction)==old['predictions_sha256'];return old
        raise RuntimeError('Existing incompatible B129 receipt; do not overwrite '+key)
    started=time.time();d=np.load(INPUT/'training.npz');X=d['X'][:,spec['columns']];y=d['y']
    assert X.shape==(969,n) and int(y.sum())==475
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(storage='sqlite:///'+(PRIVATE/(key+'.sqlite3')).as_posix(),study_name=key+'_'+sig[:16],direction='maximize',load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=10))
    def objective(trial):
        params=base.search_space(algorithm,trial)
        _,scores,rounds=base.cv_predictions(algorithm,params,X,y,folds)
        trial.set_user_attr('fold_AUROC',scores);trial.set_user_attr('fold_rounds',rounds)
        return float(np.mean(scores))
    done=study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    with threadpool_limits(limits=1):
        if len(done)<N_TRIALS:study.optimize(objective,n_trials=N_TRIALS-len(done))
        done=study.get_trials(states=[optuna.trial.TrialState.COMPLETE]);assert len(done)==N_TRIALS
        params=study.best_trial.params
        oof,scores,rounds=base.cv_predictions(algorithm,params,X,y,folds)
        assert np.isclose(np.mean(scores),study.best_value,rtol=0,atol=1e-10)
        threshold=youden(y,oof);fixed=None
        if algorithm in ['XGBoost','CatBoost','NN']:
            model,prep,fixed=base.fit_learner(algorithm,params,X,y,SEED)
            del model,prep;gc.collect();assert fixed>0
        model,prep,_=base.fit_learner(algorithm,params,X,y,SEED,fixed)
        # Open held-out arrays only after configuration, stopping and OOF threshold are fixed.
        internal=np.load(INPUT/'internal.npz');external=np.load(INPUT/'external.npz')
        assert (len(internal['y']),int(internal['y'].sum()),len(external['y']),int(external['y'].sum()))==(416,204,197,122)
        pi=base.predict(algorithm,model,prep,internal['X'][:,spec['columns']]);pe=base.predict(algorithm,model,prep,external['X'][:,spec['columns']])
    np.savez_compressed(prediction,train_id=d['id'],y=y,oof=oof,internal_id=internal['id'],yi=internal['y'],pi=pi,external_id=external['id'],ye=external['y'],p=pe)
    trials=[dict(number=t.number,value=t.value,params=t.params,fold_AUROC=t.user_attrs['fold_AUROC'],fold_rounds=t.user_attrs['fold_rounds'],seconds=t.duration.total_seconds()) for t in done]
    result=dict(status='completed',model=key,**{k:v for k,v in spec.items() if k not in ['strategy_id','reused']},strategy_id=key,reused=False,signature=sig,signature_inputs=record,params=params,tuning=trials,trials_completed=len(done),selection_metric='mean_five_fold_AUROC',selection_score=float(study.best_value),best_trial=study.best_trial.number,oof_threshold=threshold,fold_AUROC=scores,n_train=969,n_internal=416,n_external=197,class_weight=None,resampling=None,final_refit=dict(n_final_training=969,all_training_rows_used=True,fixed_rounds=fixed),seconds=time.time()-started,predictions_sha256=sha(prediction))
    write_json(OUT/(key+'_trials.json'),trials);write_json(receipt,result)
    print(json.dumps(dict(event='B129_STRATEGY_COMPLETE',strategy=key,seconds=result['seconds'])),flush=True)
    return result

def main(workers):
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    lock=PRIVATE/'active.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    started=time.time();states={}
    order=sorted(NEW,key=lambda k:({'NN':0,'RF':1,'CatBoost':2,'XGBoost':3,'SVM':4}[BY_ID[k]['algorithm']],k))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(run_strategy,k):k for k in order}
            for future in as_completed(futures):
                key=futures[future]
                try:r=future.result();states[key]=dict(status='completed',seconds=r['seconds'])
                except Exception:states[key]=dict(status='failed',traceback=traceback.format_exc())
                write_json(OUT/'progress.json',states)
        result=dict(status='completed' if len(states)==10 and all(v['status']=='completed' for v in states.values()) else 'needs_attention',new_strategies=10,reused_strategies=8,models=states,seconds=time.time()-started)
        write_json(OUT/'completion.json',result);print(json.dumps(result),flush=True)
        return 0 if result['status']=='completed' else 1
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args();raise SystemExit(main(a.workers))
