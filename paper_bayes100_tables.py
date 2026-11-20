"""Current paper tables from completed 100-trial Bayesian searches."""
from pathlib import Path
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS
import json
import ast
import numpy as np
import pandas as pd
import six_model_tables as display
from manuscript_statistics import tex,interval


ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/bayes100'
TABLES=ROOT/'outputs/tables'
ALGORITHMS=['LR','SVM','RF','XGBoost','CatBoost','NN']
REFERENCES=['Age_LR','Basic_LR']
LABELS=PREDICTOR_SET_LABELS


def search_descriptions():
    """Read the actual search function without importing or running learners."""
    tree=ast.parse((ROOT/'bayes100.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='search_space')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'bayes100.search_space','exec'),namespace)

    class Description:
        def __init__(self,model):self.items=[];self.model=model
        def suggest_float(self,key,low,high,log=False):
            self.items.append(f'{key}: {low:g} to {high:g} ({"log" if log else "linear"})'+(' if kernel=rbf' if key=='gamma' and self.model=='SVM' else ''))
            return low
        def suggest_int(self,key,low,high,step=1,log=False):
            scale='log integer' if log else f'integer, step {step}'
            self.items.append(f'{key}: {low} to {high} ({scale})')
            return low
        def suggest_categorical(self,key,choices):
            self.items.append(key+': ['+', '.join(map(str,choices))+']')
            return choices[-1]  # Include conditional RBF gamma.

    rows=[]
    for name in ALGORITHMS:
        description=Description(name)
        namespace['search_space'](name,description)
        rows.append([name,'100','; '.join(description.items)])
    return rows


def write_software_versions(meta):
    snapshot=json.loads((ROOT/'environment_snapshot.json').read_text(encoding='utf-8'))
    training=meta['LR']['signature_inputs']['packages']
    xgb=meta['XGBoost']['signature_inputs']['packages']
    cat=meta['CatBoost']['signature_inputs']['packages']
    post=snapshot['postprocessing_environment'];packages=post['packages']
    text=(f'Model training used Python {snapshot["training_python"]}, NumPy {training["numpy"]}, pandas {training["pandas"]}, '
          f'SciPy {training["scipy"]}, and scikit-learn {training["scikit-learn"]}. '
          f'XGBoost used version {xgb["xgboost"]} with Optuna {xgb["optuna"]}; '
          f'the other strategies used Optuna {training["optuna"]}, and CatBoost used version {cat["catboost"]}. '
          'The NN used scikit-learn MLPClassifier. The cleaned and retained-measurement analyses used the same training environment. '
          f'The retained performance summaries were produced using Python {post["python"].split()[0]}, '
          f'NumPy {packages["numpy"]}, pandas {packages["pandas"]}, SciPy {packages["scipy"]}, '
          f'and scikit-learn {packages["scikit-learn"]}.')
    uncertainty=json.loads((OUT/'uncertainty_validation.json').read_text(encoding='utf-8'))['software']
    text+=(f' Calibration-bootstrap and Wilson interval calculations used Python {uncertainty["python"]}, '
           f'NumPy {uncertainty["numpy"]}, pandas {uncertainty["pandas"]}, SciPy {uncertainty["scipy"]}, '
           f'and statsmodels {uncertainty["statsmodels"]}.')
    (TABLES/'software_versions.tex').write_text(tex(text)+'\n',encoding='utf-8')


def write_missing_table():
    """Reformat retained counts without rereading participant data."""
    frame=pd.read_csv(ROOT/'results/manuscript_statistics/table_s_missing_formatted.csv',
                      dtype=str,keep_default_na=False)
    frame.iloc[:,0]=frame.iloc[:,0].replace({'age':'Age','sex':'Sex','height':'Height','weight':'Weight'})
    display.table('table_s_missing','Missing predictor values before training-derived imputation',
                  frame.columns.tolist(),frame.values.tolist(),
                  r'P{0.2880504511\textwidth}P{0.2828726787\textwidth}P{0.2801764718\textwidth}P{0.1489003985\textwidth}',
                  'Counts include missing values and invalid observations set to missing by uniform validity rules. '
                  'No outcome component was imputed. '+r'\textbf{Abbreviations:} BMI, body mass index; NLR, neutrophil-to-lymphocyte ratio.',
                  long=False,width_fraction=.75)


def panel(headers,rows):
    first_columns='ll'  # Predictor labels stay on one line within the full-width table.
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


def write_case_mix_table():
    """Promote the retained six-row descriptive table without recalculation."""
    display.OUT=OUT
    frame=pd.read_csv(ROOT/'results/manuscript_statistics/table_s_case_mix_formatted.csv',
                      dtype=str,keep_default_na=False)
    assert len(frame)==6
    display.table('table4','Age and cognitive-component distributions by outcome',
                  frame.columns.tolist(),frame.values.tolist(),
                  r'P{0.134\textwidth}P{0.100\textwidth}P{0.069\textwidth}P{0.261\textwidth}P{0.215\textwidth}P{0.221\textwidth}',
                  'Both positive means MoCA <26 and cognitive-item score >=1. Percentages in this column use the number of COG-positive participants as the denominator. '
                  'Age summaries use available values before imputation or capping; MoCA was complete. \\textbf{Abbreviations:} COG, composite cognitive-screening outcome; MoCA, Montreal Cognitive Assessment; SD, standard deviation.',long=False)


def write_anthropometry_control():
    comparison=pd.read_csv(ROOT/'results/retained_measurement_sensitivity/clean_retained_historical_comparison.csv')
    display.OUT=OUT
    rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in REFERENCES+ALGORITHMS:
            r=comparison.loc[comparison.cohort.eq(cohort)&comparison.model.eq(name)].iloc[0]
            rows.append([label,MODEL_LABELS[name],f'{r.AUROC_clean:.3f}',f'{r.AUROC_retained:.3f}',f'{r.Brier_clean:.4f}',f'{r.Brier_retained:.4f}'])
    notes=('Cleaned denotes the primary valid-measurement analysis; retained denotes the separate analysis with the original invalid training measurements. '
           'Only height, weight, and BMI cells in one training participant differed; all other inputs, labels, split, software and predictive fitting code were identical. '
           'All eight 100-trial searches and training-only stopping procedures were repeated in each condition. Higher AUROC and lower Brier are favorable. '
           'These descriptive differences do not include repeated-study model-development uncertainty. '+r'\textbf{Abbreviations:} AUROC, area under the receiver operating characteristic curve; BMI, body mass index; LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network.')
    display.table('table_s_anthropometry','Sensitivity to retaining the invalid anthropometric measurements',
        ['Sample','Model','AUROC cleaned','AUROC retained','Brier cleaned','Brier retained'],rows,
        r'P{0.10\textwidth}P{0.18\textwidth}P{0.18\textwidth}P{0.18\textwidth}P{0.18\textwidth}P{0.18\textwidth}',notes,long=False,group_first=True)


def main():
    assert json.loads((OUT/'summary_validation.json').read_text())['status']=='computed_and_numerically_checked'
    display.OUT=OUT
    write_anthropometry_control()
    write_case_mix_table()
    write_missing_table()
    perf=pd.read_csv(OUT/'all_model_performance.csv')
    assert len(perf)==16 and set(perf.model)==set(REFERENCES+ALGORITHMS)
    headers=['Sample','Algorithm','AUROC (95% CI)','Brier','Accuracy, %','Sensitivity, %','Specificity, %']
    rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in ALGORITHMS:
            r=perf.loc[perf.cohort.eq(cohort)&perf.model.eq(name)].iloc[0]
            rows.append([label,MODEL_LABELS[name],interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{r.Brier:.4f}',
                         f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    display.table('table2','Internal and external performance of six algorithms using 21 predictors',headers,rows,
                  r'@{\extracolsep{\fill}}llccccc@{}',
                  'Internal PPMI n=416; external LY-PD n=197. Each algorithm used the same 969 training participants and 21 predictors. '
                  'Classification measures use the model-specific training out-of-fold Youden threshold, fixed for both evaluation samples. '
                  'Lower Brier scores indicate better probability accuracy. Supplementary Table S4 reports absolute Brier intervals and Wilson classification intervals at both operating points. NN denotes a fully connected network with three hidden layers. '
                  'AUROC and Brier intervals use 2,000 outcome-stratified participant bootstrap samples conditional on fitted models. \\textbf{Abbreviations:} PPMI, Parkinson\'s Progression Markers Initiative; LY-PD, Luoyang PD cohort; LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network; AUROC, area under the receiver operating characteristic curve; CI, confidence interval.',
                  group_first=True,full_width=True)
    absolute=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in ['Age_LR','Basic_LR','LR']:
            r=perf.loc[perf.cohort.eq(cohort)&perf.model.eq(name)].iloc[0]
            absolute.append([label,LABELS[name],interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{r.Brier:.4f}',
                             f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    abs_headers=['Sample','Predictor set','AUROC (95% CI)','Brier','Accuracy, %','Sensitivity, %','Specificity, %']
    differences=[];paired=pd.read_csv(OUT/'paired_lr.csv')
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for model,ref,title in [('LR','Basic_LR','Full set vs Basic set'),('Basic_LR','Age_LR','Basic set vs Age-only set'),('LR','Age_LR','Full set vs Age-only set')]:
            d=paired.loc[paired.cohort.eq(cohort)&paired.model.eq(model)&paired.reference.eq(ref)].set_index('metric')
            a,b=d.loc['AUROC'],d.loc['Brier']
            differences.append([label,title,interval(a.difference,a.low,a.high),
                                f'{b.difference:.4f} ({b.low:.4f} to {b.high:.4f})'])
    diff_headers=['Sample','Predictor-set comparison','AUROC difference (95% CI)','Brier difference (95% CI)']
    meta={name:json.loads((OUT/(name+'.json')).read_text()) for name in REFERENCES+ALGORITHMS}
    write_software_versions(meta)
    notes=('Age-only set used age alone; Basic set used age, sex, height, weight and BMI; Full set added 16 blood predictors to Basic set (21 predictors). Corresponding models were LR - Age-only set, LR - Basic set and LR - Full set. '
           'All models used the same 969 training participants and were evaluated on 416 internal and 197 external participants. '
           'Classification thresholds maximized the training out-of-fold Youden index: '+
           ', '.join(label+' '+f"{meta[name]['oof_threshold']:.6f}" for name,label in [('Age_LR',MODEL_LABELS['Age_LR']),('Basic_LR',MODEL_LABELS['Basic_LR']),('LR',MODEL_LABELS['LR'])])+'. '
           'The same threshold was used in both evaluation samples. Differences are the first model minus the second; higher AUROC and lower Brier are favorable. '
           'Paired differences were calculated before rounding. Absolute Brier and Wilson classification confidence intervals are in Supplementary Table S4. '
           'AUROC and Brier intervals are conditional on fitted models and use 2,000 outcome-stratified participant bootstrap samples. '
           'The principal blood-panel comparison is Full set versus Basic set (LR - Full set versus LR - Basic set). No equivalence margin or multiplicity-adjusted model-performance test was specified. \\textbf{Abbreviations:} LR, logistic regression; BMI, body mass index; AUROC, area under the receiver operating characteristic curve; CI, confidence interval.')
    lines=[r'\begin{table}[!htbp]\centering\ManuscriptTableStyle',
           r'\caption{\textbf{Performance and incremental value of logistic regression models with nested predictor sets.}}\label{tab:table3}',
           r'\textbf{A. Absolute performance}\par\medskip']+panel(abs_headers,absolute)
    lines += [r'\par\bigskip\textbf{B. Paired differences}\par\medskip']+panel(diff_headers,differences)
    lines += [r'\par\medskip\begin{minipage}{\textwidth}\ManuscriptNoteStyle',tex(notes),r'\end{minipage}\end{table}']
    (TABLES/'table3_combined.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    markdown_table(TABLES/'table3.md','Performance of logistic regression models with nested predictor sets',abs_headers,absolute,notes)
    markdown_table(TABLES/'table3_differences.md','Paired incremental performance of the logistic models',diff_headers,differences,notes)
    searches=search_descriptions()
    display.table('table_s_search','Training-only Bayesian hyperparameter search spaces',
                  ['Algorithm','TPE trials','Search space'],searches,
                  r'P{0.13\textwidth}P{0.12\textwidth}P{0.69\textwidth}',
                  'Ranges are inclusive; log denotes a logarithmic scale, linear a linear scale, and square brackets categorical candidates. '
                  'LR - Age-only set and LR - Basic set used the same C range and 100-trial budget as LR. RF max_depth=None denotes unlimited depth; SVM gamma applies only to RBF. '
                  'Stopping splits used training data only; caps and patience are described above. \\textbf{Abbreviations:} LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network; TPE, tree-structured Parzen estimator; RBF, radial basis function.',long=True)
    settings=[r'\subsection*{Selected settings and operating thresholds}',r'\begingroup\small\raggedright']
    for name in ['Age_LR','Basic_LR']+ALGORITHMS:
        r=meta[name]
        text='; '.join(k+'='+('{:.6g}'.format(v) if isinstance(v,float) else str(v)) for k,v in r['params'].items())
        if r['final_refit']['fixed_rounds'] is not None:
            text+=('; final epochs=' if name=='NN' else '; final trees=')+str(r['final_refit']['fixed_rounds'])
        text+=f"; training out-of-fold threshold {r['oof_threshold']:.6f}."
        label=MODEL_LABELS[name]
        settings += [r'\Needspace{3\baselineskip}\noindent\textbf{'+tex(label)+r'}. '+tex(text)+r'\par\smallskip']
    settings += [r'\par\endgroup']
    (TABLES/'selected_settings.tex').write_text('\n'.join(settings)+'\n',encoding='utf-8')
    uncertainty=json.loads((OUT/'uncertainty_validation.json').read_text(encoding='utf-8'))
    assert uncertainty['status']=='passed' and uncertainty['failed_bootstrap_fits']==0
    cal=pd.read_csv(OUT/'calibration.csv').merge(pd.read_csv(OUT/'calibration_intervals.csv')[
        ['cohort','model','offset_low','offset_high','slope_low','slope_high']],on=['cohort','model'],validate='one_to_one')
    cal_rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for name in REFERENCES+ALGORITHMS:
            r=cal.loc[cal.cohort.eq(cohort)&cal.model.eq(name)].iloc[0]
            cal_rows.append([label,MODEL_LABELS[name],f'{r.mean_probability:.3f}',f'{r.observed_fraction:.3f}',
                             interval(r.calibration_intercept_offset,r.offset_low,r.offset_high),
                             interval(r.calibration_slope,r.slope_low,r.slope_high)])
    cal_notes=('Offset intercept estimates calibration-in-the-large with slope fixed at 1; slope comes from a separate model with freely estimated intercept and slope. '
               'Calibration intervals use 2,000 ordinary participant bootstrap samples per evaluation cohort, shared across strategies; all resamples were estimable. '
               'These evaluation-sample fits did not update predictions. '+
               ('All fits converged. \\textbf{Abbreviations:} LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network.' if cal.calibration_fit_success.all() else 'NA indicates an unestimable calibration fit.'))
    display.table('table_s_calibration','Calibration of reference models and six algorithms using Full set',
                  ['Sample','Model','Mean prediction','Observed fraction','Offset intercept (95% CI)','Slope (95% CI)'],cal_rows,
                  r'P{0.10\textwidth}P{0.15\textwidth}P{0.12\textwidth}P{0.12\textwidth}P{0.255\textwidth}P{0.255\textwidth}',cal_notes,long=False,group_first=True)
    fixed=pd.read_csv(OUT/'classification_intervals.csv').merge(
        pd.read_csv(OUT/'metrics.csv')[['cohort','model','operating_point','Brier','Brier_low','Brier_high']],
        on=['cohort','model','operating_point'],validate='one_to_one')
    fixed_rows=[]
    for cohort,label in [('PPMI_internal','Internal'),('FAHHAUST_external','External')]:
        for operating,threshold_label in [('OOF_Youden','OOF'),('fixed_0.5','0.5')]:
            for name in REFERENCES+ALGORITHMS:
                r=fixed.loc[fixed.cohort.eq(cohort)&fixed.model.eq(name)&fixed.operating_point.eq(operating)].iloc[0]
                fixed_rows.append([label+'; '+threshold_label,MODEL_LABELS[name],interval(r.Brier,r.Brier_low,r.Brier_high,digits=4),
                    *[interval(r[key]*100,r[key+'_low']*100,r[key+'_high']*100,digits=1) for key in ['accuracy','sensitivity','specificity']]])
    notes=('OOF denotes each model-specific Youden threshold selected from training out-of-fold predictions, fixed for both evaluation cohorts; 0.5 denotes the reference threshold. '
           'Accuracy, sensitivity, and specificity are percentages with Wilson 95% confidence intervals; their denominators are all participants, COG-positive participants, and COG-negative participants, respectively. '
           'AUROC and Brier are threshold-independent; repeated Brier values and intervals apply to both operating points. '
           'Brier intervals use 2,000 outcome-stratified participant bootstrap samples conditional on fitted models. '
           '\\textbf{Abbreviations:} LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network; AUROC, area under the receiver operating characteristic curve; CI, confidence interval; COG, composite cognitive-screening outcome.')
    display.table('table_s_fixed','Brier scores and classification performance at training-derived and fixed thresholds',
                  ['Sample; cut-off','Model','Brier (95% CI)','Accuracy, % (95% CI)','Sensitivity, % (95% CI)','Specificity, % (95% CI)'],fixed_rows,
                  r'P{0.10\textwidth}P{0.16\textwidth}P{0.26\textwidth}P{0.16\textwidth}P{0.16\textwidth}P{0.16\textwidth}',
                  notes,long=True,group_first=True)
    # Reserve the complete note as the long table's final footer.
    path=TABLES/'table_s_fixed.tex'
    content=path.read_text(encoding='utf-8')
    footer=(r'\bottomrule'+'\n'+r'\multicolumn{6}{@{}p{\textwidth}@{}}{\ManuscriptNoteStyle '+tex(notes)+r'}\\'+'\n'+r'\endlastfoot')
    content=content.replace(r'\bottomrule\endfoot',r'\bottomrule\endfoot'+'\n'+footer)
    content=content.replace(r'\noindent\ManuscriptNoteStyle '+tex(notes)+'\n','')
    path.write_text(content,encoding='utf-8')
    print(json.dumps(dict(event='BAYES100_PAPER_TABLES_COMPLETE',main_performance_rows=12,LR_absolute_rows=6,LR_paired_rows=6,search_algorithms=6,calibration_rows=len(cal_rows))),flush=True)


if __name__=='__main__':main()

