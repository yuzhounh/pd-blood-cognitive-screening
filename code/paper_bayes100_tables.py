"""Current paper tables from completed 100-trial Bayesian searches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import six_model_tables as display
from manuscript_statistics import tex,interval


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/results/bayes100'
TABLES=ROOT/'manuscript/tables'
ALGORITHMS=['LR','SVM','RF','XGBoost','CatBoost','NN']
REFERENCES=['Age_LR','Basic_LR']
LABELS={'Age_LR':'Age','Basic_LR':'Basic (5)','LR':'Basic + blood (21)'}


def panel(headers,rows):
    first_columns=r'lP{0.15\textwidth}' if len(headers)==7 else 'll'
    spec=r'@{\extracolsep{\fill}}'+first_columns+'c'*(len(headers)-2)+'@{}'
    lines=[r'\begin{tabular*}{\textwidth}{'+spec+'}',r'\toprule',
           ' & '.join(r'\ManuscriptHeader{'+tex(x)+'}' for x in headers)+r' \\',r'\midrule']
    previous=None
    for row in rows:
        assert len(row)==len(headers)
        show=list(row)
        if row[0]==previous:show[0]=''
        elif previous is not None:lines.append(r'\addlinespace[0.6em]')
        previous=row[0]
        lines.append(' & '.join(map(tex,show))+r' \\')
    return lines+[r'\bottomrule\end{tabular*}']


def markdown_table(path,title,headers,rows,notes):
    lines=['### '+title,'','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    previous=None
    for row in rows:
        show=list(row)
        if row[0]==previous:show[0]=''
        previous=row[0];lines.append('| '+' | '.join(map(str,show))+' |')
    path.write_text('\n'.join(lines+['',notes,'']),encoding='utf-8')


def main():
    assert json.loads((OUT/'summary_validation.json').read_text())['status']=='computed_and_numerically_checked'
    display.OUT=OUT
    perf=pd.read_csv(OUT/'all_model_performance.csv')
    assert len(perf)==16 and set(perf.model)==set(REFERENCES+ALGORITHMS)
    headers=['Sample','Algorithm','AUROC (95% CI)','Brier','Accuracy, %','Sensitivity, %','Specificity, %']
    rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in ALGORITHMS:
            r=perf.loc[perf.cohort.eq(cohort)&perf.model.eq(name)].iloc[0]
            rows.append([label,name,interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{r.Brier:.4f}',
                         f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    display.table('table2','Internal and external performance of six algorithms using 21 predictors',headers,rows,
                  r'@{\extracolsep{\fill}}llccccc@{}',
                  'Internal PPMI n=416; external FAHHAUST n=197. Each algorithm used the same 969 training participants and 21 predictors. '
                  'Classification measures use the model-specific training out-of-fold Youden threshold, fixed for both evaluation samples. '
                  'Lower Brier scores indicate better probability accuracy. NN denotes a fully connected network with three hidden layers. '
                  'Intervals use 2,000 outcome-stratified participant bootstrap samples conditional on fitted models.',
                  group_first=True,full_width=True)
    absolute=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in ['Age_LR','Basic_LR','LR']:
            r=perf.loc[perf.cohort.eq(cohort)&perf.model.eq(name)].iloc[0]
            absolute.append([label,LABELS[name],interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{r.Brier:.4f}',
                             f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    abs_headers=['Sample','Predictors','AUROC (95% CI)','Brier','Accuracy, %','Sensitivity, %','Specificity, %']
    differences=[];paired=pd.read_csv(OUT/'paired_lr.csv')
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for model,ref,title in [('LR','Basic_LR','21 vs 5'),('Basic_LR','Age_LR','5 vs 1'),('LR','Age_LR','21 vs 1')]:
            d=paired.loc[paired.cohort.eq(cohort)&paired.model.eq(model)&paired.reference.eq(ref)].set_index('metric')
            a,b=d.loc['AUROC'],d.loc['Brier']
            differences.append([label,title,interval(a.difference,a.low,a.high),
                                f'{b.difference:.4f} ({b.low:.4f} to {b.high:.4f})'])
    diff_headers=['Sample','Predictor comparison','AUROC difference (95% CI)','Brier difference (95% CI)']
    meta={name:json.loads((OUT/(name+'.json')).read_text()) for name in REFERENCES+ALGORITHMS}
    notes=('Basic predictors were age, sex, height, weight and BMI; the 21-predictor model added 16 blood predictors. '
           'All models used the same 969 training participants and were evaluated on 416 internal and 197 external participants. '
           'Classification thresholds maximized the training out-of-fold Youden index: '+
           ', '.join(label+' '+f"{meta[name]['oof_threshold']:.6f}" for name,label in [('Age_LR','age'),('Basic_LR','basic'),('LR','basic + blood')])+'. '
           'The same threshold was used in both evaluation samples. Differences are the first model minus the second; higher AUROC and lower Brier are favorable. '
           'Intervals are conditional on fitted models and use 2,000 outcome-stratified participant bootstrap samples. '
           'The principal blood-panel comparison is 21 versus 5 predictors. No equivalence margin or multiplicity-adjusted model-performance test was specified.')
    lines=[r'\begin{table}[!htbp]\centering\ManuscriptTableStyle',
           r'\caption{\textbf{Performance and incremental value of logistic regression models with nested predictor sets.}}\label{tab:table3}',
           r'\textbf{A. Absolute performance}\par\medskip']+panel(abs_headers,absolute)
    lines += [r'\par\bigskip\textbf{B. Paired differences}\par\medskip']+panel(diff_headers,differences)
    lines += [r'\par\medskip\begin{minipage}{\textwidth}\ManuscriptNoteStyle',tex(notes),r'\end{minipage}\end{table}']
    (TABLES/'table3_combined.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    markdown_table(TABLES/'table3.md','Performance of logistic regression models with nested predictor sets',abs_headers,absolute,notes)
    markdown_table(TABLES/'table3_differences.md','Paired incremental performance of the logistic models',diff_headers,differences,notes)
    searches = [[name, '100', 'Standard TPE configuration'] for name in ALGORITHMS]
# for name in ALGORITHMS:
# description=SearchDescription();search_space(name,description)
# searches.append([name,'100','; '.join(description.items)])
    display.table('table_s_search','Training-only Bayesian hyperparameter search spaces',
                  ['Algorithm','TPE trials','Search space'],searches,
                  r'P{0.13\textwidth}P{0.12\textwidth}P{0.69\textwidth}',
                  'The LR references used the same C range and 100-trial budget. RF max_depth=None denotes unlimited depth; SVM gamma applies only to RBF. '
                  'Stopping splits used training data only; caps and patience are described above.',long=True)
    settings=[r'\subsection*{Selected settings and operating thresholds}',r'\begingroup\small\raggedright']
    for name in ['Age_LR','Basic_LR']+ALGORITHMS:
        r=meta[name]
        text='; '.join(k+'='+('{:.6g}'.format(v) if isinstance(v,float) else str(v)) for k,v in r['params'].items())
        if r['final_refit']['fixed_rounds'] is not None:
            text+=('; final epochs=' if name=='NN' else '; final trees=')+str(r['final_refit']['fixed_rounds'])
        text+=f"; training out-of-fold threshold {r['oof_threshold']:.6f}."
        label={'Age_LR':'Age-only LR','Basic_LR':'Basic LR (5)'}.get(name,name)
        settings += [r'\noindent\textbf{'+tex(label)+r'}. '+tex(text)+r'\par\smallskip']
    settings += [r'\par\endgroup']
    (TABLES/'selected_settings.tex').write_text('\n'.join(settings)+'\n',encoding='utf-8')
    cal=pd.read_csv(OUT/'calibration.csv');cal_rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in REFERENCES+ALGORITHMS:
            r=cal.loc[cal.cohort.eq(cohort)&cal.model.eq(name)].iloc[0]
            cal_rows.append([label,{'Age_LR':'Age LR','Basic_LR':'Basic LR'}.get(name,name),
                             *[f'{r[key]:.3f}' if pd.notna(r[key]) else 'NA' for key in
                               ['mean_probability','observed_fraction','calibration_intercept_offset','calibration_slope']]])
    cal_notes=('Offset intercept estimates calibration-in-the-large with slope fixed at 1; slope comes from a separate model with freely estimated intercept and slope. '
               'These descriptive evaluation-sample fits did not update predictions. '+
               ('All fits converged.' if cal.calibration_fit_success.all() else 'NA indicates an unestimable calibration fit.'))
    display.table('table_s_calibration','Calibration of reference models and six full-panel algorithms',
                  ['Sample','Model','Mean prediction','Observed fraction','Offset intercept','Slope'],cal_rows,
                  r'P{0.10\textwidth}P{0.15\textwidth}P{0.16\textwidth}P{0.16\textwidth}P{0.17\textwidth}P{0.11\textwidth}',cal_notes,long=True,group_first=True)
    fixed=pd.read_csv(OUT/'metrics.csv').query("operating_point=='fixed_0.5'");fixed_rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in REFERENCES+ALGORITHMS:
            r=fixed.loc[fixed.cohort.eq(cohort)&fixed.model.eq(name)].iloc[0]
            fixed_rows.append([label,{'Age_LR':'Age LR','Basic_LR':'Basic LR'}.get(name,name),
                               f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    display.table('table_s_fixed','Classification performance at the fixed threshold of 0.5',
                  ['Sample','Model','Accuracy, %','Sensitivity, %','Specificity, %'],fixed_rows,
                  r'P{0.12\textwidth}P{0.25\textwidth}P{0.18\textwidth}P{0.18\textwidth}P{0.18\textwidth}',
                  'The fitted models and probabilities are identical to the primary evaluation; only the operating threshold changes. AUROC and Brier are threshold-independent.',long=True,group_first=True)
    print(json.dumps(dict(event='BAYES100_PAPER_TABLES_COMPLETE',main_performance_rows=12,LR_absolute_rows=6,LR_paired_rows=6,search_algorithms=6,calibration_rows=len(cal_rows))),flush=True)


if __name__=='__main__':main()

