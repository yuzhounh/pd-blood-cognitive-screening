"""B129 registry: six algorithms crossed with three frozen predictor sets."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data/derived/bayes100_input'
OUT=ROOT/'manuscript/results/B129 Full Factorial'
PRIVATE=ROOT/'data/derived/B129 Full Factorial Predictions'
WORK=ROOT/'tmp/B129 Full Factorial'
BASE_RESULTS=ROOT/'manuscript/results/bayes100'
BASE_PREDICTIONS=ROOT/'data/derived/bayes100'
ALGORITHMS=['LR','SVM','RF','XGBoost','CatBoost','NN']
# Immutable analysis keys; display_labels maps these to semantic manuscript labels.
SETS=['Set 1','Set 2','Full set']
BASELINE={'Age_LR','Basic_LR','LR','SVM','RF','XGBoost','CatBoost','NN'}
STRATEGIES=[]
for algorithm in ALGORITHMS:
    for index,(label,n) in enumerate(zip(SETS,[1,5,21]),1):
        key=algorithm if index==3 else ('Age_LR' if index==1 else 'Basic_LR') if algorithm=='LR' else algorithm+'_Set'+str(index)
        STRATEGIES.append(dict(strategy_id=key,algorithm=algorithm,predictor_set=label,n_features=n,columns=list(range(n)),reused=key in BASELINE))
BY_ID={s['strategy_id']:s for s in STRATEGIES}
NEW=[s['strategy_id'] for s in STRATEGIES if not s['reused']]
SEED=20261014
N_TRIALS=100

def paths(strategy_id):
    spec=BY_ID[strategy_id]
    return ((BASE_RESULTS if spec['reused'] else OUT)/(strategy_id+'.json'),
            (BASE_PREDICTIONS if spec['reused'] else PRIVATE)/(strategy_id+'.npz'))
