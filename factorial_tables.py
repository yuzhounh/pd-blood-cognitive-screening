"""Render main and supplementary tables from the 18-strategy results."""
import json,math
import numpy as np
import pandas as pd
import six_model_tables as display
from manuscript_statistics import tex
from factorial_protocol import ROOT,OUT,STRATEGIES,ALGORITHMS,paths
TABLES=ROOT/'outputs/tables'
# B134: content-width measurements include padding and unbroken header words.
CALIBRATION_WIDTHS=[.0847984058,.1124547419,.1177138904,.2164005216,.2164005440,.2522318964]
CLASSIFICATION_WIDTHS=[.1310422367,.1368741608,.1340309262,.1921265721,.1921265721,.2137995321]
SECONDARY_WIDTHS=[.0993646823,.1297246955,.2403355598,.2545600048,.2760150577]
SELECTED_WIDTHS=[.1217560179,.1272235461,.1027893365,.0932345945,.0939180740,.4610784311]
SENSITIVITY_WIDTHS=[.1258034678,.1642416217,.1715510564,.1388854253,.1388854253,.1247349770,.1358980265]
def interval(v,lo,hi,d=3):
    return 'NA' if not np.isfinite([v,lo,hi]).all() else f'{v:.{d}f} ({lo:.{d}f} to {hi:.{d}f})'

def difference_interval(v,lo,hi,d=3):
    """Retain the sign of a nonzero difference or bound that rounds to zero."""
    if not np.isfinite([v,lo,hi]).all():
        return 'NA'
    def signed(value):
        precision=d
        while value != 0 and float(f'{value:.{precision}f}') == 0 and precision < 12:
            precision += 1
        return f'{value:.{precision}f}' if value == 0 or float(f'{value:.{precision}f}') != 0 else f'{value:.3e}'
    return f'{signed(v)} ({signed(lo)} to {signed(hi)})'
def table(name,title,headers,rows,widths,notes,long=True,**layout):
    display.OUT=OUT
    display.table(name,title,headers,rows,''.join('P{'+f'{w:.10f}'+r'\textwidth}' for w in widths),notes,long=long,group_first=False,**layout)
def main():
    perf=pd.read_csv(OUT/'performance.csv');delta=pd.read_csv(OUT/'paired_differences.csv');cal=pd.read_csv(OUT/'calibration.csv');classification=pd.read_csv(OUT/'classification_intervals.csv')
    def get(frame,cohort,key):return frame.loc[frame.cohort.eq(cohort)&frame.strategy_id.eq(key)].iloc[0]
    note='Intervals use 2,000 outcome-stratified participant bootstrap samples, conditional on fitted models. Higher AUROC and lower Brier are favorable. '+r'\textbf{Abbreviations:} AUROC, area under the receiver operating characteristic curve; CI, confidence interval; LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network.'
    rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for alg in ALGORITHMS:
            r=get(perf,cohort,alg)
            rows.append([label,alg,interval(r.AUROC,r.AUROC_low,r.AUROC_high),interval(r.Brier,r.Brier_low,r.Brier_high,4)])
    table('table2','Performance of the six algorithms using Full set',['Sample','Algorithm','AUROC (95% CI)','Brier (95% CI)'],rows,[.140,.184,.315,.361],note,long=False)
    def pairrows(frame):
        rows=[]
        for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
            for alg in ALGORITHMS:
                a=frame.loc[frame.cohort.eq(cohort)&frame.algorithm.eq(alg)&frame.metric.eq('AUROC')].iloc[0]
                b=frame.loc[frame.cohort.eq(cohort)&frame.algorithm.eq(alg)&frame.metric.eq('Brier')].iloc[0]
                rows.append([label,alg,difference_interval(a.difference,a.low,a.high),difference_interval(b.difference,b.low,b.high,4)])
        return rows
    table('table3','Increment from adding blood predictors within each algorithm',['Sample','Algorithm','AUROC difference (95% CI)','Brier difference (95% CI)'],pairrows(delta.loc[delta.primary_comparison]),[.116,.152,.386,.346],
        'Differences are Full set minus Basic set on the same evaluation participants. Positive AUROC and negative Brier differences favor Full set. Paired intervals use shared outcome-stratified bootstrap indices and are conditional on fitted models. Comparisons are exploratory and unadjusted for multiplicity; intervals crossing zero do not establish equivalence. Paired differences were calculated before rounding. Additional decimal places preserve the sign of nonzero values that would otherwise round to zero.',long=False)
    # Search descriptions are read from the unchanged baseline algorithm implementation.
    from paper_bayes100_tables import search_descriptions
    table('factorial_search','Shared search spaces across the three predictor sets',['Algorithm','Trials per set','Search space'],search_descriptions(),[.13,.16,.71],
        'Each algorithm used the same applicable search space in Age-only set, Basic set and Full set, with independent 100-trial TPE searches, 10 startup trials and seed 20261014. Early stopping and preprocessing were training-only; no class weighting or outcome resampling was used.',header_lines={'Trials per set':('Trials per set',)})
    rows=[]
    for spec in STRATEGIES:
        r=json.loads(paths(spec['strategy_id'])[0].read_text());params='; '.join(f'{k}={v:.6g}' if isinstance(v,float) else f'{k}={v}' for k,v in r['params'].items())
        rounds=r['final_refit']['fixed_rounds'];rows.append([spec['algorithm'],spec['predictor_set'],f'{r["selection_score"]:.3f}',f'{r["oof_threshold"]:.3f}',str(rounds) if rounds is not None else '--',params])
    table('factorial_selected','Selected configurations for all 18 strategies',['Algorithm','Set','CV AUROC','OOF cut-off','Rounds','Selected parameters'],rows,SELECTED_WIDTHS,
        'CV AUROC is the arithmetic mean of five training validation folds. Rounds are the final tree or epoch counts determined by training-only stopping; a dash denotes an algorithm without early stopping. Each full-precision OOF Youden cut-off was applied unchanged to both evaluation cohorts. CV AUROC and cut-offs are rounded to three decimal places for display; operating-point statistics use the full-precision stored cut-offs.',tabcolsep=3,header_lines={'CV AUROC':('CV','AUROC'),'OOF cut-off':('OOF','cut-off')})
    for cohort,code,label in [('PPMI_internal','internal','Internal PPMI'),('FAHHAUST_external','external','External LY-PD')]:
        rows=[]
        for spec in STRATEGIES:
            r=get(perf,cohort,spec['strategy_id']);rows.append([spec['algorithm'],spec['predictor_set'],interval(r.AUROC,r.AUROC_low,r.AUROC_high),interval(r.Brier,r.Brier_low,r.Brier_high,4)])
        table('factorial_performance_'+code,label+': performance of all 18 strategies',['Algorithm','Set','AUROC (95% CI)','Brier (95% CI)'],rows,[.175,.182,.300,.343],note)
        rows=[]
        for spec in STRATEGIES:
            for operating in ['OOF_Youden','fixed_0.5']:
                r=classification.loc[classification.cohort.eq(cohort)&classification.strategy_id.eq(spec['strategy_id'])&classification.operating_point.eq(operating)].iloc[0]
                rows.append([spec['algorithm'],spec['predictor_set'],('OOF '+f'{r.threshold:.4f}') if operating=='OOF_Youden' else '0.5',*[interval(r[k]*100,r[k+'_low']*100,r[k+'_high']*100,1) for k in ['accuracy','sensitivity','specificity']]])
        table('factorial_classification_'+code,label+': classification at training-derived and fixed thresholds',['Algorithm','Set','Cut-off','Accuracy, % (95% CI)','Sensitivity, % (95% CI)','Specificity, % (95% CI)'],rows,CLASSIFICATION_WIDTHS,
            'OOF denotes the training-derived Youden cut-off, fixed for both evaluation cohorts; 0.5 is the reference operating point. Intervals are Wilson score intervals. Denominators are all participants, positive participants and negative participants, respectively. AUROC and Brier are threshold-independent and are reported in Table S3. For XGBoost with Basic set and Full set, compressed predicted probabilities were entirely below 0.5 in both evaluation cohorts, giving 0% sensitivity and 100% specificity at the 0.5 reference threshold.',nowrap_body=True,header_break='percent_ci')
    rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for spec in STRATEGIES:
            r=get(cal,cohort,spec['strategy_id']);rows.append([label,spec['algorithm'],spec['predictor_set'],interval(r.calibration_intercept_offset,r.offset_low,r.offset_high),interval(r.calibration_intercept,r.intercept_low,r.intercept_high),interval(r.calibration_slope,r.slope_low,r.slope_high)])
    complete=bool((cal.joint_bootstrap_successes==2000).all())
    table('factorial_calibration','Calibration of all 18 strategies in both evaluation cohorts',['Sample','Algorithm','Set','Offset intercept (95% CI)','Free intercept (95% CI)','Slope (95% CI)'],rows,CALIBRATION_WIDTHS,
        'The offset intercept estimates calibration-in-the-large with slope fixed at 1. Free intercept and slope are estimated together using centered/scaled log odds and transformed back to the original scale. Intervals use 2,000 ordinary participant bootstrap samples per cohort, shared across strategies. '+('All 2,000 joint and offset fits succeeded in every row. ' if complete else 'NA denotes an unidentified estimate or an interval withheld because not all bootstrap fits succeeded; success counts and reasons are supplied in the accompanying results. ')+
        'These evaluation-sample fits did not update predictions. Large slopes can arise from compressed probability ranges and are not clinical effect sizes.',font_size=(8.5,10.5),tabcolsep=2.5,nowrap_body=True,header_break='ci')
    rows=[]
    for comparison in ['Set 2 - Set 1','Full set - Set 1']:
        selected=delta.loc[delta.comparison.eq(comparison)]
        for row in pairrows(selected):rows.append(row[:2]+[comparison]+row[2:])
    table('factorial_secondary','Secondary within-algorithm predictor-set comparisons',['Sample','Algorithm','Comparison','AUROC difference (95% CI)','Brier difference (95% CI)'],rows,SECONDARY_WIDTHS,
        'Differences are the first named predictor set minus the second named set: the richer set minus the reference. Positive AUROC and negative Brier favor the richer set. Intervals use the same 2,000 outcome-stratified participant bootstrap samples as the principal comparisons, conditional on fitted models and without multiplicity adjustment. Additional decimal places preserve the sign of nonzero values that would otherwise round to zero.',nowrap_body=True,header_break='ci')
    # Preserve the previously completed eight-strategy measurement sensitivity analysis.
    old=pd.read_csv(ROOT/'results/retained_measurement_sensitivity/clean_retained_historical_comparison.csv');rows=[]
    lookup={s['strategy_id']:s for s in STRATEGIES}
    for r in old.itertuples():
        spec=lookup[r.model];rows.append(['Internal' if r.cohort=='PPMI_internal' else 'External',spec['algorithm'],spec['predictor_set'],f'{r.AUROC_clean:.3f}',f'{r.AUROC_retained:.3f}',f'{r.Brier_clean:.4f}',f'{r.Brier_retained:.4f}'])
    table('factorial_anthropometry','Anthropometric-handling sensitivity in the eight baseline strategies',['Sample','Algorithm','Set','AUROC cleaned','AUROC retained','Brier cleaned','Brier retained'],rows,SENSITIVITY_WIDTHS,
        'The sensitivity analysis covered LR with all three predictor sets and the five other algorithms with Full set. Only three body-measurement input cells in one training participant differed. Both conditions repeated preprocessing, tuning, stopping and refitting. All other inputs and evaluation participants were identical. These descriptive differences do not measure repeated-study training uncertainty.',header_lines={k:tuple(k.split(' ',1)) for k in ['AUROC cleaned','AUROC retained','Brier cleaned','Brier retained']})
    # Keep the software statement next to the training tables.
    env=json.loads((OUT/'statistics_validation.json').read_text())['software'];training=json.loads((OUT/'baseline_reuse.json').read_text())['packages']
    text=f"Training used Python 3.12.15, NumPy {training['numpy']}, pandas {training['pandas']}, SciPy {training['scipy']}, scikit-learn {training['scikit-learn']}, Optuna {training['optuna']}, XGBoost {training['xgboost']} and CatBoost {training['catboost']}. "
    text+=f"Performance and interval calculations were computed from the retained prediction arrays using Python {env['python']}, NumPy {env['numpy']}, pandas {env['pandas']}, SciPy {env['scipy']}, scikit-learn {env['scikit-learn']} and statsmodels {env['statsmodels']}. This statistical postprocessing did not deserialize fitted estimators or generate new model predictions in the later environment."
    (TABLES/'factorial_software.tex').write_text(tex(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='generated',main_tables_replaced=[2,3],strategies=18,supplementary_table_parts=10)))
if __name__=='__main__':main()
