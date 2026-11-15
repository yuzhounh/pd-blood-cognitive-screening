"""Validate reuse and low-dimensional behavior before the B129 full search."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import bayes100 as base
from common_preprocessing import CappedImputer,sha,write_json
from factorial_protocol import ROOT,INPUT,OUT,PRIVATE,WORK,STRATEGIES,BY_ID,BASELINE,SEED,paths
from factorial_preprocessing import SubsetPreprocessor
from factorial_bayes100 import configure

def main():
    OUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    assert len(STRATEGIES)==len(BY_ID)==18 and len({(s['algorithm'],s['predictor_set']) for s in STRATEGIES})==18
    input_manifest=json.loads((INPUT/'input_manifest.json').read_text())
    for filename,digest in input_manifest['files_sha256'].items():assert sha(INPUT/filename)==digest
    packages=base.package_versions();reused=[];protection={}
    reference=None
    for spec in STRATEGIES:
        if not spec['reused']:continue
        receipt,prediction=paths(spec['strategy_id']);r=json.loads(receipt.read_text())
        assert r['status']=='completed' and r['trials_completed']==100
        assert r['n_features']==spec['n_features']
        assert r['signature_inputs']['packages']==packages
        assert r['signature_inputs']['inputs']==input_manifest
        for key,name in [('code','bayes100.py'),('preprocessing','common_preprocessing.py'),('body_rules','body_rules.py')]:assert r['signature_inputs'][key]==sha(ROOT/'code'/name)
        assert sha(prediction)==r['predictions_sha256']
        with np.load(prediction) as z:
            if reference is None:reference={k:z[k].copy() for k in ['train_id','internal_id','external_id','y','yi','ye']}
            for key,value in reference.items():assert np.array_equal(z[key],value)
        reused.append(dict(**spec,receipt=str(receipt.relative_to(ROOT)),prediction=str(prediction.relative_to(ROOT)),receipt_sha256=sha(receipt),predictions_sha256=sha(prediction)))
    assert len(reused)==8
    # Protect both previous experiments, including their aggregate receipts, from writes.
    for directory in ['data/derived/bayes100_input','data/derived/B128 Retained Inputs','data/derived/bayes100','data/derived/B128 Retained Predictions','manuscript/results/bayes100','manuscript/results/B128 Retained Measurement']:
        for f in (ROOT/directory).glob('*'):
            if f.is_file() and f.suffix in ['.npz','.json','.csv']:protection[str(f.relative_to(ROOT))]=sha(f)
    write_json(OUT/'baseline_reuse.json',dict(status='passed',baseline_version='v3.2',baseline_commit='6d1d20fbfacb952d55ff9f8ea791f5338e3a9866',reused_strategies=reused,packages=packages,protected_sha256=protection,input_manifest=input_manifest))
    d=np.load(INPUT/'training.npz');X=d['X'];y=d['y'];tr,va=next(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    for n in [1,5,21]:
        old=CappedImputer().fit(X[tr]);new=SubsetPreprocessor(sex_index=1 if n>=5 else None).fit(X[tr,:n])
        assert np.allclose(old.transform(X[va])[:,:n],new.transform(X[va,:n]),rtol=1e-12,atol=1e-12)
    defaults=dict(SVM=dict(C=1.,kernel='rbf',gamma=.03),RF=dict(n_estimators=100,max_depth=2,min_samples_split=5,min_samples_leaf=12,max_features=.03),XGBoost=dict(learning_rate=.03,max_depth=4,min_child_weight=3.,subsample=.8,colsample_bytree=.5,reg_alpha=.01,reg_lambda=1.,gamma=0.),CatBoost=dict(learning_rate=.03,depth=4,l2_leaf_reg=3.,random_strength=.1,subsample=.8,rsm=.5),NN=dict(width1=128,width2=64,width3=32,learning_rate=.001,alpha=.0001,batch_size=128))
    checks=[]
    with threadpool_limits(limits=1):
        for algorithm,params in defaults.items():
            for n in [1,5]:
                configure(n);start=time.time()
                model,prep,rounds=base.fit_learner(algorithm,params,X[tr,:n],y[tr],SEED)
                predictions=base.predict(algorithm,model,prep,X[va,:n])
                assert predictions.shape==(len(va),) and np.isfinite(predictions).all() and ((predictions>=0)&(predictions<=1)).all()
                checks.append(dict(algorithm=algorithm,n_features=n,status='passed',selected_rounds=rounds,seconds=time.time()-start))
                print(json.dumps(checks[-1]),flush=True)
    assert all(sha(ROOT/rel)==digest for rel,digest in protection.items())
    write_json(OUT/'preflight.json',dict(status='passed',strategy_count=18,reused=8,new=10,columnwise_equivalence_rtol=1e-12,held_out_evaluation_files_used_for_smoke=False,checks=checks,protected_files_unchanged=True))
    print(json.dumps(dict(status='passed',reused=8,new=10,low_dimensional_fits=len(checks))))
if __name__=='__main__':main()
