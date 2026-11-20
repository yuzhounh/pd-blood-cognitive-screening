"""Current six-algorithm paper tables; original reconstruction is supplementary only."""
from pathlib import Path
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS, display_text
import json
import pandas as pd
from common_preprocessing import write_json,sha
from manuscript_statistics import tex,interval,fit_paragraph_columns

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/six_model'
TABLES=ROOT/'outputs/tables'
NAMES=['LR','SVM','RF','XGBoost','CatBoost','NN']
RECONSTRUCTION_SPEC=r'P{0.2100475421\textwidth}P{0.1943274514\textwidth}P{0.4000222183\textwidth}P{0.1956027882\textwidth}'


def header_tex(label,break_mode=None,header_lines=None):
    """Keep units/CI together on a second header line; leave data untouched."""
    parts=(header_lines or {}).get(label)
    if parts is None and break_mode:
        suffix={'ci':'(95% CI)','percent_ci':'% (95% CI)'}[break_mode]
        if label.endswith(suffix):
            prefix=label[:-len(suffix)].rstrip(' ,')
            parts=(prefix,suffix)
    value=(r'\newline'.join(r'\mbox{'+tex(part)+'}' for part in parts)
           if parts else tex(label))
    return r'\ManuscriptHeader{'+value+'}'


def table(name,title,headers,rows,spec,notes,long=False,group_first=False,full_width=False,
          font_size=None,tabcolsep=None,nowrap_body=False,width_fraction=None,
          header_break=None,header_lines=None):
    title,notes=display_text(title),display_text(notes)
    headers=[display_text(value) for value in headers]
    rows=[[display_text(value) for value in row] for row in rows]
    spec=fit_paragraph_columns(spec)
    pd.DataFrame(rows,columns=headers).to_csv(OUT/(name+'_formatted.csv'),index=False)
    display_rows=[list(row) for row in rows]
    if group_first:
        for i in range(1,len(rows)):
            if rows[i][0]==rows[i-1][0]: display_rows[i][0]=''
    md=['### '+title,'','| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    md += ['| '+' | '.join(map(str,row))+' |' for row in display_rows]
    md += ['',notes,'']
    (TABLES/(name+'.md')).write_text('\n'.join(md),encoding='utf-8')
    header=' & '.join(header_tex(x,header_break,header_lines) for x in headers)+r' \\'
    if long:
        lines=[r'\par\begingroup\ManuscriptTableStyle',r'\begin{longtable}{'+spec+'}',
               r'\caption{\textbf{'+tex(title).rstrip('.')+r'.}}\label{tab:'+name+r'}\\',r'\toprule',header,
               r'\midrule\endfirsthead',r'\multicolumn{'+str(len(headers))+r'}{l}{\tablename\ \thetable\ continued}\\',
               r'\toprule',header,r'\midrule\endhead',r'\bottomrule\endfoot']
        local_style=[]
        if font_size is not None:
            local_style.append(r'\ManuscriptTableSize{'+str(font_size[0])+'}{'+str(font_size[1])+'}')
        if tabcolsep is not None:
            local_style.append(r'\setlength{\tabcolsep}{'+str(tabcolsep)+'pt}')
        lines[1:1]=local_style
    else:
        tabular='tabular*' if full_width else 'tabular'
        opening=r'\begin{'+tabular+'}'+(r'{\textwidth}' if full_width else '')+'{'+spec+'}'
        lines=[r'\begin{table}[!htbp]\centering\ManuscriptTableStyle',
               r'\caption{\textbf{'+tex(title).rstrip('.')+r'.}}\label{tab:'+name+'}',opening,r'\toprule',header,r'\midrule']
    if width_fraction is not None:
        assert not long and 0<width_fraction<=1
        lines[1:1]=[r'\begin{minipage}{'+str(width_fraction)+r'\textwidth}\centering']
    for i,row in enumerate(display_rows):
        if group_first and i and rows[i][0]!=rows[i-1][0]: lines.append(r'\addlinespace[0.6em]')
        cells=[tex(x) for x in row]
        if nowrap_body:
            cells=[r'\mbox{'+cell+'}' for cell in cells]
        lines.append(' & '.join(cells)+r' \\')
    if long: lines += [r'\end{longtable}',r'\noindent\ManuscriptNoteStyle '+tex(notes),r'\par\endgroup']
    else: lines += [r'\bottomrule\end{'+tabular+'}',r'\par\medskip\begin{minipage}{\textwidth}\ManuscriptNoteStyle '+tex(notes),r'\end{minipage}\end{table}']
    if width_fraction is not None:
        lines[-1]=lines[-1].replace(r'\end{minipage}\end{table}',r'\end{minipage}\par\end{minipage}\end{table}')
    (TABLES/(name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_reconstruction_table():
    """Render S8 from retained ancillary metrics without historical main outputs."""
    repro=pd.read_csv(ROOT/'results/paper_reproduction/metrics.csv').query("version=='tuned'")
    rows=[]
    for name in ['RF','XGBoost','CatBoost','NN']:
        a=repro.loc[repro.model.eq(name)&repro.flow.eq('split_first')].iloc[0]
        b=repro.loc[repro.model.eq(name)&repro.flow.eq('smote_first')].iloc[0]
        rows.append([name,f'{a.paper_AUROC:.2f}',interval(a.AUROC,a.AUROC_low,a.AUROC_high),f'{b.AUROC:.3f}'])
    table('table_s_reconstruction','Ancillary reconstruction using the original published PPMI table',
          ['Algorithm','Reported AUROC','Workflow A AUROC (95% CI)','Workflow B AUROC'],rows,
          RECONSTRUCTION_SPEC,
          'This ancillary analysis is separate from primary model development. Reported AUROC values preserve the two decimal places reported in the source publication; reconstructed AUROC estimates are shown to three decimal places. A: split observed participants before training-only SMOTE, development n=1,023, observed test n=256. '
          'B: full-table SMOTE before splitting, development n=1,260, test n=316 including 63 synthetic records. '
          'B is a non-independent diagnostic scenario. Its score is not assigned an independent-test confidence interval; the workflows do not use identical evaluation participants. \\textbf{Abbreviations:} RF, random forest; NN, neural network; AUROC, area under the receiver operating characteristic curve; CI, confidence interval; SMOTE, synthetic minority oversampling technique.',False,header_lines={'Workflow A AUROC (95% CI)':('Workflow A','AUROC (95% CI)')})


def main():
    perf=pd.read_csv(OUT/'all_model_performance.csv')
    rows=[]
    for cohort in ['PPMI_internal','FAHHAUST_external']:
        for name in NAMES:
            r=perf.loc[perf.cohort.eq(cohort)&perf.model.eq(name)].iloc[0]
            rows.append(['Internal' if cohort=='PPMI_internal' else 'External',name,
                         interval(r.AUROC,r.AUROC_low,r.AUROC_high),
                         f'{r.Brier:.4f}',f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    table('table2','Internal and external performance of six algorithms using 21 predictors',
          ['Sample','Algorithm','AUROC (95% CI)','Brier','Accuracy, %','Sensitivity, %','Specificity, %'],rows,
          r'@{\extracolsep{\fill}}llccccc@{}',
          'Internal PPMI n=416; external LY-PD n=197. Each model used the same 969 training participants and 21 predictors. '
          'Classification measures use the model-specific training out-of-fold Youden threshold, fixed for both evaluation samples. '
          'Lower Brier scores indicate better probability accuracy. '
          'NN denotes an ordinary fully connected network with three hidden layers. Intervals use 2,000 outcome-stratified participant bootstrap samples conditional on fitted models. \\textbf{Abbreviations:} PPMI, Parkinson\'s Progression Markers Initiative; LY-PD, Luoyang PD cohort; LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network; AUROC, area under the receiver operating characteristic curve; CI, confidence interval.',
          group_first=True,full_width=True)
    launch=json.loads((ROOT/'results/benchmark21_auc/launch.json').read_text())
    nn=json.loads((OUT/'NN.json').read_text())
    grids=[];selected=[]
    for name in NAMES:
        candidates=nn['candidates'] if name=='NN' else launch['candidates'][name]
        record=nn if name=='NN' else json.loads((ROOT/f'results/benchmark21_auc/{name}.json').read_text())
        items=[]
        for key in candidates[0]:
            values=list(dict.fromkeys(str(c[key]) for c in candidates))
            items.append(key+'='+('/'.join(values)))
        grids.append([name,str(len(candidates)),'; '.join(items)])
        chosen='; '.join(f'{k}={v}' for k,v in record['params'].items())
        if record['final_refit'].get('fixed_rounds') is not None:
            chosen+='; final trees/epochs='+str(record['final_refit']['fixed_rounds'])
        selected.append(dict(model=name,configuration=chosen,threshold=record['oof_threshold']))
    table('table_s_search','Training-only candidate settings for the six algorithms',
          ['Algorithm','Configurations','Candidate settings'],grids,
          r'P{0.13\textwidth}P{0.14\textwidth}P{0.67\textwidth}',
          'Configurations maximize mean five-fold training AUROC. For NN, width lists the three hidden layers; training epochs are selected by the same cross-validation criterion. '
          'NN used ReLU, dropout 0.1, Adam, and unweighted binary cross-entropy. No algorithm used class balancing or resampling.',True)
    lines=[r'\subsection*{Selected settings and operating thresholds}',r'\begingroup\small\raggedright']
    for r in selected:
        lines.append(r'\noindent\textbf{'+r['model']+'}. '+tex(r['configuration'])+f"; training out-of-fold threshold {r['threshold']:.6f}."+r'\par\smallskip')
    lines += [r'\noindent LR - Age-only set: $C=0.01$, threshold 0.502105; LR - Basic set: $C=0.1$, threshold 0.499146.',r'\par\endgroup']
    (TABLES/'selected_settings.tex').write_text(display_text('\n'.join(lines))+'\n',encoding='utf-8')
    pd.DataFrame(selected).to_csv(OUT/'selected_settings.csv',index=False)
    cal=pd.read_csv(OUT/'calibration.csv');assert len(cal)==14 and cal.calibration_fit_success.all()
    rows=[]
    for cohort in ['PPMI_internal','FAHHAUST_external']:
        for name in ['Basic_LR']+NAMES:
            r=cal.loc[cal.cohort.eq(cohort)&cal.model.eq(name)].iloc[0]
            rows.append(['Internal' if cohort=='PPMI_internal' else 'External',MODEL_LABELS[name],
                         f'{r.mean_probability:.3f}',f'{r.observed_fraction:.3f}',f'{r.calibration_intercept_offset:.3f}',f'{r.calibration_slope:.3f}'])
    table('table_s_calibration','Calibration of the basic reference and six full-panel algorithms',
          ['Sample','Model','Mean prediction','Observed fraction','Offset intercept','Slope'],rows,
          r'P{0.10\textwidth}P{0.15\textwidth}P{0.16\textwidth}P{0.16\textwidth}P{0.17\textwidth}P{0.11\textwidth}',
          'Offset intercept estimates calibration-in-the-large with slope fixed at 1; the reported slope comes from a separate model with freely estimated intercept and slope. '
          'These descriptive evaluation-sample fits did not update predictions. All fits converged. \\textbf{Abbreviations:} LR, logistic regression; SVM, support vector machine; RF, random forest; NN, neural network.',True)
    case_mix=pd.read_csv(ROOT/'results/manuscript_statistics/table_s_case_mix_formatted.csv',
                         dtype=str,keep_default_na=False)
    table('table4','Age and cognitive-component distributions by outcome',
          case_mix.columns.tolist(),case_mix.values.tolist(),
          r'P{0.12\textwidth}P{0.07\textwidth}P{0.07\textwidth}P{0.23\textwidth}P{0.18\textwidth}P{0.23\textwidth}',
          'Both positive means MoCA <26 and cognitive-item score >=1. Percentages in this column use the number of COG-positive participants as the denominator. '
          'Age summaries use available values before imputation or capping; MoCA was complete. \\textbf{Abbreviations:} COG, composite cognitive-screening outcome; MoCA, Montreal Cognitive Assessment; SD, standard deviation.',False)
    fixed=pd.read_csv(OUT/'metrics.csv').query("operating_point=='fixed_0.5'")
    rows=[]
    for cohort in ['PPMI_internal','FAHHAUST_external']:
        for name in ['Age_LR','Basic_LR']+NAMES:
            r=fixed.loc[fixed.cohort.eq(cohort)&fixed.model.eq(name)].iloc[0]
            rows.append(['Internal' if cohort=='PPMI_internal' else 'External',
                         MODEL_LABELS[name],
                         f'{r.accuracy*100:.1f}',f'{r.sensitivity*100:.1f}',f'{r.specificity*100:.1f}'])
    table('table_s_fixed','Classification performance at the fixed threshold of 0.5',
          ['Sample','Model','Accuracy, %','Sensitivity, %','Specificity, %'],rows,
          r'P{0.12\textwidth}P{0.25\textwidth}P{0.18\textwidth}P{0.18\textwidth}P{0.18\textwidth}',
          'The fitted models and probabilities are identical to the primary evaluation; only the operating threshold changes. AUROC and Brier score are threshold-independent.',True)
    # Replace the old generated full-performance helper as well, so active TeX contains no retired algorithms.
    rows=[]
    for _,r in perf.iterrows():
        rows.append([r.cohort,r.model,interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{r.Brier:.4f}'])
    table('table_s_performance','Complete performance of the eight prediction strategies',
          ['Sample','Strategy','AUROC (95% CI)','Brier'],rows,
          r'P{0.28\textwidth}P{0.16\textwidth}P{0.32\textwidth}P{0.18\textwidth}',
          'The six full-panel algorithms and two logistic references use the same training and evaluation participants.',True)
    write_reconstruction_table()
    combined=TABLES/'table3_combined.tex'
    combined.write_text(combined.read_text(encoding='utf-8').replace(r'\begin{table}[p]',r'\begin{table}[!htbp]'),encoding='utf-8')
    for name in ['table1','table_s_missing']:
        path=TABLES/(name+'.tex')
        content=path.read_text(encoding='utf-8')
        if not content.startswith(r'\par'):
            content=r'\par'+content
        content=content.replace('\n'+r'\endgroup','\n'+r'\par\endgroup')
        path.write_text(content,encoding='utf-8')
    versions={k:v for k,v in launch['packages'].items() if k in ['numpy','pandas','scikit-learn','xgboost','catboost','torch']}
    versions['optuna']='5.0.0'
    (TABLES/'software_versions.tex').write_text('\n'.join(r'\noindent '+tex(k)+' '+tex(v)+r'\\' for k,v in versions.items())+'\n',encoding='utf-8')
    write_json(OUT/'table_manifest.json',dict(main_algorithms=NAMES,main_performance_rows=12,
               prediction_strategies=8,calibration_rows=14,fixed_threshold_rows=16,
               original_PPMI_used_for_primary_development=False,
               main_inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'all_model_performance.csv',OUT/'NN.json']}))
    print({'event':'SIX_MODEL_TABLES_COMPLETE','algorithms':NAMES,'main_rows':12})


if __name__=='__main__': main()
