"""Training-only TPE search on the frozen PPMI split; independent held-out evaluation."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from common_preprocessing import CappedImputer, sha, write_json, youden

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/derived/bayes100_input'
OUT=ROOT/'manuscript/results/bayes100'
PRIVATE=ROOT/'data/derived/bayes100'
SEED=20261014
NAMES=['NN','CatBoost','RF','XGBoost','SVM','LR','Age_LR','Basic_LR']
TREE_CAP=5000
TREE_PATIENCE=50
NN_CAP=300
NN_PATIENCE=30


def package_versions():
    result={}
    for p in ['numpy','pandas','scipy','scikit-learn','xgboost','catboost','optuna']:
        try:result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:result[p]=None
    return result


def search_space(name,trial):
    if name in ['LR','Age_LR','Basic_LR']:
        return dict(C=trial.suggest_float('C',1e-4,1e3,log=True))
    if name=='SVM':
        p=dict(kernel=trial.suggest_categorical('kernel',['linear','rbf']),
               C=trial.suggest_float('C',1e-3,1e3,log=True))
        if p['kernel']=='rbf':p['gamma']=trial.suggest_float('gamma',1e-4,1.,log=True)
        return p
    if name=='RF':
        return dict(n_estimators=trial.suggest_int('n_estimators',100,1000,step=50),
                    max_depth=trial.suggest_categorical('max_depth',[None,2,3,4,5,6,8,10,12,16,20]),
                    min_samples_split=trial.suggest_int('min_samples_split',2,20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf',1,100,log=True),
                    max_features=trial.suggest_float('max_features',.03,1.))
    if name=='XGBoost':
        return dict(learning_rate=trial.suggest_float('learning_rate',.001,.3,log=True),
                    max_depth=trial.suggest_int('max_depth',1,8),
                    min_child_weight=trial.suggest_float('min_child_weight',1.,20.,log=True),
                    subsample=trial.suggest_float('subsample',.3,1.),
                    colsample_bytree=trial.suggest_float('colsample_bytree',.5,1.),
                    reg_alpha=trial.suggest_float('reg_alpha',1e-4,10.,log=True),
                    reg_lambda=trial.suggest_float('reg_lambda',1e-3,10.,log=True),
                    gamma=trial.suggest_float('gamma',0.,5.))
    if name=='CatBoost':
        return dict(learning_rate=trial.suggest_float('learning_rate',.005,.3,log=True),
                    depth=trial.suggest_int('depth',1,8),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1.,30.,log=True),
                    random_strength=trial.suggest_float('random_strength',1e-3,10.,log=True),
                    subsample=trial.suggest_float('subsample',.5,1.),
                    rsm=trial.suggest_float('rsm',.5,1.))
    if name=='NN':
        return dict(width1=trial.suggest_categorical('width1',[64,128,256,512]),
                    width2=trial.suggest_categorical('width2',[32,64,128,256]),
                    width3=trial.suggest_categorical('width3',[16,32,64,128]),
                    learning_rate=trial.suggest_float('learning_rate',1e-4,.03,log=True),
                    alpha=trial.suggest_float('alpha',1e-6,10.,log=True),
                    batch_size=trial.suggest_categorical('batch_size',[32,64,128,256]))
    raise ValueError(name)


def feature_indices(name):
    return [0] if name=='Age_LR' else list(range(5 if name=='Basic_LR' else 21))


def nn_fit(x,y,vx,vy,params,seed,fixed_rounds):
    net=MLPClassifier(hidden_layer_sizes=tuple(params[k] for k in ['width1','width2','width3']),
                      activation='relu',solver='adam',alpha=params['alpha'],
                      learning_rate_init=params['learning_rate'],batch_size=params['batch_size'],
                      random_state=np.random.RandomState(seed),max_iter=1,warm_start=True,
                      early_stopping=False,tol=0.,n_iter_no_change=NN_CAP+1)
    best_loss=float('inf');best_model=None;best_epoch=0;stale=0
    for epoch in range(1,(fixed_rounds or NN_CAP)+1):
        net.partial_fit(x,y,classes=np.array([0,1]))
        if not np.isfinite(net.loss_):raise FloatingPointError('Nonfinite NN training loss')
        if fixed_rounds is None:
            value=float(log_loss(vy,net.predict_proba(vx),labels=[0,1]))
            if not np.isfinite(value):raise FloatingPointError('Nonfinite NN stopping loss')
            if value<best_loss-1e-5:
                best_loss=value;best_epoch=epoch;best_model=copy.deepcopy(net);stale=0
            else:stale+=1
            if stale>=NN_PATIENCE:break
    if fixed_rounds is None:
        assert best_epoch>0
        net=best_model
    return net,int(fixed_rounds or best_epoch)


def fit_learner(name,params,X,y,seed,fixed_rounds=None):
    indices=np.arange(len(y));stop=None
    if name in ['XGBoost','CatBoost','NN'] and fixed_rounds is None:
        indices,stop=train_test_split(indices,test_size=.1,random_state=seed,stratify=y)
    prep=CappedImputer().fit(X[indices])
    cols=feature_indices(name)
    x=prep.transform(X[indices])[:,cols]
    vx=prep.transform(X[stop])[:,cols] if stop is not None else None
    assert np.isfinite(x).all()
    if name in ['LR','Age_LR','Basic_LR']:
        model=LogisticRegression(C=params['C'],solver='lbfgs',penalty='l2',max_iter=3000,
                                 class_weight=None,random_state=seed).fit(x,y[indices]);rounds=None
    elif name=='SVM':
        model=SVC(**params,probability=True,class_weight=None,random_state=seed,cache_size=512).fit(x,y[indices]);rounds=None
    elif name=='RF':
        model=RandomForestClassifier(**params,n_jobs=1,class_weight=None,random_state=seed).fit(x,y[indices]);rounds=None
    elif name=='XGBoost':
        from xgboost import XGBClassifier
        model=XGBClassifier(**params,n_estimators=fixed_rounds or TREE_CAP,n_jobs=1,tree_method='hist',
                            objective='binary:logistic',eval_metric='auc',random_state=seed,
                            **({'early_stopping_rounds':TREE_PATIENCE} if stop is not None else {}))
        model.fit(x,y[indices],**({'eval_set':[(vx,y[stop])],'verbose':False} if stop is not None else {}))
        rounds=int(model.best_iteration)+1 if stop is not None else int(fixed_rounds)
    elif name=='CatBoost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**params,iterations=fixed_rounds or TREE_CAP,thread_count=1,
                                 bootstrap_type='Bernoulli',loss_function='Logloss',eval_metric='AUC',
                                 random_seed=seed,allow_writing_files=False,verbose=False)
        model.fit(x,y[indices],**({'eval_set':(vx,y[stop]),'early_stopping_rounds':TREE_PATIENCE,
                                 'use_best_model':True} if stop is not None else {}))
        rounds=int(model.get_best_iteration())+1 if stop is not None else int(fixed_rounds)
    elif name=='NN':
        model,rounds=nn_fit(x,y[indices],vx,y[stop] if stop is not None else None,params,seed,fixed_rounds)
    else:raise ValueError(name)
    return model,prep,rounds


def predict(name,model,prep,X):
    x=prep.transform(X)[:,feature_indices(name)]
    p=np.asarray(model.predict_proba(x)[:,1],dtype=float)
    assert p.shape==(len(X),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def cv_predictions(name,params,X,y,folds):
    oof=np.full(len(y),np.nan);coverage=np.zeros(len(y),int);scores=[];rounds=[]
    for i,(tr,va) in enumerate(folds):
        assert not np.intersect1d(tr,va).size
        model,prep,count=fit_learner(name,params,X[tr],y[tr],SEED+i)
        oof[va]=predict(name,model,prep,X[va]);coverage[va]+=1
        scores.append(float(roc_auc_score(y[va],oof[va])));rounds.append(count)
        del model,prep;gc.collect()
    assert (coverage==1).all() and np.isfinite(oof).all()
    return oof,scores,rounds


def signature(trials):
    record=dict(code=sha(__file__),preprocessing=sha(ROOT/'code/common_preprocessing.py'),
                inputs=json.loads((DATA/'input_manifest.json').read_text()),
                trials=trials,seed=SEED,packages=package_versions())
    for filename,expected in record['inputs']['files_sha256'].items():
        assert sha(DATA/filename)==expected,filename
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),record


def run_model(name,trials):
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sig,record=signature(trials);path=OUT/(name+'.json')
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('signature')==sig and cached.get('status')=='completed':return cached
    started=time.time();training=np.load(DATA/'training.npz')
    X,y=training['X'],training['y']
    assert X.shape==(969,21) and y.sum()==475 and len(np.unique(training['id']))==969
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    study=optuna.create_study(storage='sqlite:///'+(PRIVATE/(name+'.sqlite3')).as_posix(),
                              study_name=name+'_'+sig[:16],direction='maximize',load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=10))
    def objective(trial):
        params=search_space(name,trial)
        _,scores,rounds=cv_predictions(name,params,X,y,folds)
        trial.set_user_attr('fold_AUROC',scores);trial.set_user_attr('fold_rounds',rounds)
        return float(np.mean(scores))
    completed=len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    with threadpool_limits(limits=1):
        if completed<trials:study.optimize(objective,n_trials=trials-completed)
        done=study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        assert len(done)==trials
        params=study.best_trial.params
        oof,scores,rounds=cv_predictions(name,params,X,y,folds)
        assert np.isclose(np.mean(scores),study.best_value,rtol=0,atol=1e-10)
        cut=youden(y,oof);fixed_rounds=None
        if name in ['XGBoost','CatBoost','NN']:
            pilot,pilot_prep,fixed_rounds=fit_learner(name,params,X,y,SEED)
            del pilot,pilot_prep;gc.collect()
            assert fixed_rounds>0
        model,prep,_=fit_learner(name,params,X,y,SEED,fixed_rounds)
        # Held-out evaluation files are opened only after all choices above are fixed.
        internal=np.load(DATA/'internal.npz');external=np.load(DATA/'external.npz')
        assert (len(internal['y']),int(internal['y'].sum()))==(416,204)
        assert (len(external['y']),int(external['y'].sum()))==(197,122)
        pi=predict(name,model,prep,internal['X']);pe=predict(name,model,prep,external['X'])
    np.savez_compressed(PRIVATE/(name+'.npz'),train_id=training['id'],y=y,oof=oof,
                        internal_id=internal['id'],yi=internal['y'],pi=pi,
                        external_id=external['id'],ye=external['y'],p=pe)
    rows=[dict(number=t.number,value=t.value,params=t.params,fold_AUROC=t.user_attrs['fold_AUROC'],
               fold_rounds=t.user_attrs['fold_rounds'],seconds=t.duration.total_seconds()) for t in done]
    write_json(OUT/(name+'_trials.json'),rows)
    result=dict(status='completed',model=name,signature=sig,signature_inputs=record,params=params,
                tuning=rows,trials_completed=len(done),selection_metric='mean_five_fold_AUROC',
                selection_score=float(study.best_value),best_trial=study.best_trial.number,
                oof_threshold=cut,fold_AUROC=scores,n_train=969,n_internal=416,n_external=197,
                n_features=len(feature_indices(name)),class_weight=None,resampling=None,
                final_refit=dict(n_final_training=969,all_training_rows_used=True,fixed_rounds=fixed_rounds),
                seconds=time.time()-started,predictions_sha256=sha(PRIVATE/(name+'.npz')))
    write_json(path,result)
    print(json.dumps(dict(event='BAYES100_MODEL_COMPLETE',model=name,trials=len(done),seconds=result['seconds'])),flush=True)
    return result


def pilot(models):
    OUT.mkdir(parents=True,exist_ok=True)
    d=np.load(DATA/'training.npz');X,y=d['X'],d['y']
    tr,va=next(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    defaults=dict(LR={'C':.1},SVM={'C':1.,'kernel':'rbf','gamma':.03},
                  RF={'n_estimators':500,'max_depth':8,'min_samples_split':2,'min_samples_leaf':5,'max_features':.5},
                  XGBoost={'learning_rate':.03,'max_depth':4,'min_child_weight':3.,'subsample':.8,'colsample_bytree':.8,'reg_alpha':.01,'reg_lambda':1.,'gamma':0.},
                  CatBoost={'learning_rate':.03,'depth':4,'l2_leaf_reg':3.,'random_strength':.1,'subsample':.8,'rsm':.8},
                  NN={'width1':128,'width2':64,'width3':32,'learning_rate':.001,'alpha':.0001,'batch_size':128})
    rows=[]
    with threadpool_limits(limits=1):
        for name in ['LR','SVM','RF','XGBoost','CatBoost','NN']:
            if name not in models:continue
            started=time.time();model,prep,rounds=fit_learner(name,defaults[name],X[tr],y[tr],SEED)
            p=predict(name,model,prep,X[va]);elapsed=time.time()-started
            rows.append(dict(model=name,one_fold_seconds=elapsed,selected_rounds=rounds,
                             training_fold_AUROC=float(roc_auc_score(y[va],p)),rough_100_trial_seconds=500*elapsed))
            del model,prep;gc.collect()
            print(json.dumps(rows[-1]),flush=True)
    write_json(OUT/'pilot.json',dict(packages=package_versions(),evaluation_samples_read=False,timings=rows))


def controller(trials,workers,models):
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    started=time.time();statuses={};lock=PRIVATE/'active.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(run_model,n,trials):n for n in models}
            for future in as_completed(futures):
                name=futures[future]
                try:
                    r=future.result();statuses[name]=dict(status='completed',seconds=r['seconds'])
                except Exception:
                    statuses[name]=dict(status='failed',traceback=traceback.format_exc())
                write_json(OUT/'progress.json',statuses)
        result=dict(event='BAYES100_COMPLETION',status='completed' if all(s['status']=='completed' for s in statuses.values()) else 'needs_attention',models=statuses,seconds=time.time()-started)
    except Exception:
        result=dict(event='BAYES100_COMPLETION',status='failed',traceback=traceback.format_exc(),seconds=time.time()-started)
    finally:lock.unlink(missing_ok=True)
    write_json(OUT/'completion.json',result);print(json.dumps(result),flush=True)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--model',choices=NAMES)
    parser.add_argument('--trials',type=int,default=100);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--models',nargs='+',choices=NAMES,default=NAMES)
    args=parser.parse_args()
    if args.pilot:pilot(args.models)
    elif args.model:run_model(args.model,args.trials)
    else:sys.exit(controller(args.trials,args.workers,args.models))
