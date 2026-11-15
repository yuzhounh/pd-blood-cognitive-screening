"""Read and independently verify all 18 retained prediction packets."""
import json,hashlib
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from common_preprocessing import sha,youden,write_json
from factorial_protocol import ROOT,INPUT,OUT,STRATEGIES,SEED,paths

def load_verified():
    baseline=json.loads((OUT/'baseline_reuse.json').read_text())
    assert all(sha(ROOT/rel)==digest for rel,digest in baseline['protected_sha256'].items())
    split=np.load(ROOT/'data/derived/cog_pm3/frozen_split.npz');external=np.load(INPUT/'external.npz')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(np.zeros((969,1)),split['y']))
    packets={};receipts={};sources={}
    for spec in STRATEGIES:
        key=spec['strategy_id'];rp,pp=paths(key);r=json.loads(rp.read_text())
        assert r['status']=='completed' and r['trials_completed']==100 and r['n_features']==spec['n_features']
        assert r['class_weight'] is None and r['resampling'] is None
        assert r['final_refit']['n_final_training']==969 and r['final_refit']['all_training_rows_used']
        sig=r['signature_inputs'];assert sig['packages']==baseline['packages']
        if not spec['reused']:
            assert sig['strategy']==spec and sig['trials']==100 and sig['seed']==SEED
            assert all(sha(ROOT/'code'/n)==v for n,v in sig['code_sha256'].items())
            assert sig['input_manifest']==baseline['input_manifest']
        assert hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()==r['signature']
        assert sha(pp)==r['predictions_sha256']
        with np.load(pp) as z:a={k:z[k].copy() for k in z.files}
        for name in ['train_id','internal_id','y','yi']:assert np.array_equal(a[name],split[name]),(key,name)
        assert np.array_equal(a['external_id'],external['id']) and np.array_equal(a['ye'],external['y'])
        for name in ['oof','pi','p']:assert np.isfinite(a[name]).all() and ((a[name]>=0)&(a[name]<=1)).all()
        assert abs(youden(a['y'],a['oof'])-r['oof_threshold'])<1e-12
        scores=[]
        for _,va in folds:
            y=a['y'][va];n1=int(y.sum());n0=len(y)-n1
            scores.append((rankdata(a['oof'][va])[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
        assert np.allclose(scores,r['fold_AUROC'],atol=1e-12,rtol=0)
        assert abs(np.mean(scores)-r['selection_score'])<1e-12
        trials=r['tuning'];assert len(trials)==100 and len({t['number'] for t in trials})==100
        assert all(len(t['fold_AUROC'])==5 and abs(np.mean(t['fold_AUROC'])-t['value'])<1e-12 for t in trials)
        best=next(t for t in trials if t['number']==r['best_trial'])
        assert best['params']==r['params'] and best['value']==max(t['value'] for t in trials)==r['selection_score']
        packets[key]=a;receipts[key]=r;sources[key]=dict(receipt=str(rp.relative_to(ROOT)),prediction=str(pp.relative_to(ROOT)),reused=spec['reused'],receipt_sha256=sha(rp),prediction_sha256=sha(pp))
    write_json(OUT/'prediction_validation.json',dict(status='passed',strategies=18,reused=8,new=10,new_trials=1000,total_retained_trials=1800,verified_trial_fold_scores=9000,ids_labels_split_and_training_thresholds_verified=True,baseline_conditions_unchanged=True,sources=sources))
    return packets,receipts
if __name__=='__main__':
    load_verified();print('B129: all 18 prediction packets and 1800 trial records verified.')
