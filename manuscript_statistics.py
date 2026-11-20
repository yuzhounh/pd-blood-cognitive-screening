"""Complete prediction-level intervals and generate manuscript tables; no model fits."""
from pathlib import Path
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS, display_text
import importlib.metadata
import json
import platform
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from common_preprocessing import LABS, clean_sheet, sha, write_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/manuscript_statistics'
TABLES = ROOT / 'outputs/tables'
SEED, N_BOOT = 20261014, 2000
FEATURES = ['age', 'sex', 'height', 'weight', 'BMI'] + LABS
SOURCES = {
    'Age_LR': ROOT / 'data/derived/age_reference_auc/age_LR.npz',
    'Basic_LR': ROOT / 'data/derived/project_assessment_auc/basic_LR.npz',
    'LR': ROOT / 'data/derived/benchmark21_auc/LR.npz',
}
COHORTS = ['PPMI_internal', 'FAHHAUST_external']
LABELS = PREDICTOR_SET_LABELS


def tex(s):
    return (display_text(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
            .replace('>=', r'$\geq$').replace('10^12/L', r'$10^{12}$/L')
            .replace('10^9/L', r'$10^9$/L').replace('kg/m2', r'kg/m$^2$')
            .replace('umol/L', r'$\mu$mol/L'))


def interval(v, lo, hi, digits=3):
    return f'{v:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})'


def fit_paragraph_columns(spec):
    widths = re.findall(r'P\{([0-9.]+)\\textwidth\}', spec)
    if not widths:
        return spec
    total = sum(map(float, widths))
    return re.sub(r'P\{([0-9.]+)\\textwidth\}',
                  lambda m: r'P{\dimexpr' + f'{float(m[1])/total:.6f}' +
                  r'\textwidth-2\tabcolsep\relax}', spec)


def save_table(name, title, headers, rows, spec, notes, long=False):
    title,notes=display_text(title),display_text(notes)
    rows=[[display_text(value) for value in row] for row in rows]
    spec = spec.replace('p{', 'P{')
    spec = fit_paragraph_columns(spec)
    frame = pd.DataFrame(rows, columns=headers)
    frame.to_csv(OUT / f'{name}_formatted.csv', index=False)
    markdown = [f'### {title}', '', '| ' + ' | '.join(headers) + ' |',
                '|' + '|'.join(['---'] * len(headers)) + '|']
    markdown += ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]
    markdown += ['', notes, '']
    (TABLES / f'{name}.md').write_text('\n'.join(markdown), encoding='utf-8')
    heading = ' & '.join(r'\ManuscriptHeader{'+tex(h)+'}' for h in headers) + r' \\'
    if long:
        lines = [r'\par\begingroup\ManuscriptTableStyle',
                 r'\begin{longtable}{' + spec + '}',
                 r'\caption{\textbf{' + tex(title).rstrip('.') + r'.}}\label{tab:' + name + r'}\\',
                 r'\toprule', heading, r'\midrule\endfirsthead',
                 r'\multicolumn{' + str(len(headers)) + r'}{l}{\tablename\ \thetable\ continued}\\',
                 r'\toprule', heading, r'\midrule\endhead',
                 r'\bottomrule\endfoot']
    else:
        lines = [r'\begin{table}[!htbp]\centering\ManuscriptTableStyle',
                 r'\caption{\textbf{' + tex(title).rstrip('.') + r'.}}\label{tab:' + name + '}',
                 r'\begin{tabular}{' + spec + '}', r'\toprule', heading, r'\midrule']
    lines += [' & '.join(tex(x) for x in row) + r' \\' for row in rows]
    note_text = tex(notes).replace('Abbreviations:', r'\textbf{Abbreviations:}')
    if long:
        lines += [r'\end{longtable}', r'\noindent\ManuscriptNoteStyle ' + note_text, r'\par\endgroup']
    else:
        lines += [r'\bottomrule\end{tabular}', r'\par\medskip',
                  r'\begin{minipage}{\textwidth}\ManuscriptNoteStyle ' + note_text,
                  r'\end{minipage}\end{table}']
    (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    launch_path = ROOT / 'results/benchmark21_auc/launch.json'
    launch = json.loads(launch_path.read_text(encoding='utf-8'))
    for rel, expected in launch['hashes'].items():
        assert sha(ROOT / rel) == expected, rel
    inputs = [ROOT / 'data/derived/cog_pm3/ppmi_cog.csv',
              ROOT / 'data/derived/cog_v1/fahhaust.csv',
              ROOT / 'data/derived/cog_pm3/frozen_split.npz', launch_path,
              ROOT / 'results/age_reference_auc/comparison.csv',
              ROOT / 'results/benchmark21_auc/metrics.csv',
              ROOT / 'results/project_assessment_auc/paired_vs_basic.csv',
              ROOT / 'results/paper_reproduction/metrics.csv',
              ROOT / 'results/paper_reproduction/data_manifest.json',
              ROOT / 'results/external_auc_gap/diagnostic.json',
              *SOURCES.values()]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    full, _ = clean_sheet(pd.read_csv(inputs[0]))
    ext, _ = clean_sheet(pd.read_csv(inputs[1]))
    split = np.load(inputs[2])
    by_id = full.set_index('PATNO', drop=False)
    cohorts = {'PPMI_train': by_id.loc[split['train_id']],
               'PPMI_internal': by_id.loc[split['internal_id']],
               'FAHHAUST_external': ext}
    assert [(len(d), int(d.COG.sum())) for d in cohorts.values()] == [(969,475),(416,204),(197,122)]
    assert not set(split['train_id']).intersection(split['internal_id'])
    comparison = pd.read_csv(inputs[4])
    previous = pd.read_csv(inputs[5])
    old_pairs = pd.read_csv(inputs[6])
    saved = {name: dict(np.load(path)) for name, path in SOURCES.items()}
    absolute, paired = [], []
    rng = np.random.default_rng(SEED)
    tables_only = '--tables-only' in sys.argv
    for cohort in ([] if tables_only else COHORTS):
        d = cohorts[cohort]
        yk, pk, idk = ('yi','pi','internal_id') if cohort == COHORTS[0] else ('ye','p','external_id')
        y = d.COG.to_numpy(int)
        for name, a in saved.items():
            assert np.array_equal(a[idk], d.PATNO.to_numpy()), (cohort, name, 'ID')
            assert np.array_equal(a[yk], y), (cohort, name, 'label')
            assert np.isfinite(a[pk]).all() and ((a[pk] >= 0) & (a[pk] <= 1)).all()
            assert np.array_equal(a['train_id'], split['train_id'])
        predictions = {name: a[pk] for name, a in saved.items()}
        points = {name: np.array([roc_auc_score(y,p),brier_score_loss(y,p)])
                  for name,p in predictions.items()}
        draws = {name: np.empty((N_BOOT,2)) for name in predictions}
        cls = [np.flatnonzero(y == k) for k in [0,1]]
        for b in range(N_BOOT):
            ix = np.r_[rng.choice(cls[0],len(cls[0])),rng.choice(cls[1],len(cls[1]))]
            for name,p in predictions.items():
                draws[name][b] = [roc_auc_score(y[ix],p[ix]),np.mean((y[ix]-p[ix])**2)]
        for name in predictions:
            metric_row = comparison.loc[(comparison.cohort == cohort) & (comparison.model == name)
                                        & (comparison.operating_point == 'OOF_Youden')].iloc[0]
            for k,metric in enumerate(['AUROC','Brier']):
                assert abs(points[name][k] - metric_row[metric]) < 1e-12
                lo,hi = np.quantile(draws[name][:,k],[.025,.975])
                source = 'saved_prediction_bootstrap'
                if name == 'LR':
                    existing = previous.loc[(previous.cohort == cohort) & (previous.model == name)
                                            & (previous.operating_point == 'OOF_Youden')].iloc[0]
                    lo,hi = existing[metric+'_low'],existing[metric+'_high']
                    source = 'existing_benchmark_interval'
                absolute.append(dict(cohort=cohort,model=name,metric=metric,value=points[name][k],
                                     low=float(lo),high=float(hi),source=source))
        for model,reference in [('Basic_LR','Age_LR'),('LR','Age_LR'),('LR','Basic_LR')]:
            for k,metric in enumerate(['AUROC','Brier']):
                delta = points[model][k]-points[reference][k]
                lo,hi = np.quantile(draws[model][:,k]-draws[reference][:,k],[.025,.975])
                source = 'saved_prediction_bootstrap'
                if reference == 'Basic_LR':
                    existing = old_pairs.loc[(old_pairs.cohort == cohort) & (old_pairs.model == model)
                                             & (old_pairs.metric == metric)].iloc[0]
                    assert np.allclose([delta,lo,hi],existing[['difference','low','high']].to_numpy(float),atol=1e-12,rtol=0)
                    delta,lo,hi = existing[['difference','low','high']].to_numpy(float)
                    source = 'existing_paired_interval'
                paired.append(dict(cohort=cohort,model=model,reference=reference,metric=metric,
                                   difference=float(delta),low=float(lo),high=float(hi),source=source))
    if tables_only:
        abs_df = pd.read_csv(OUT/'absolute_intervals.csv')
        pair_df = pd.read_csv(OUT/'paired_lr.csv')
        assert (len(abs_df),len(pair_df)) == (12,12)
    else:
        abs_df, pair_df = pd.DataFrame(absolute), pd.DataFrame(paired)
        abs_df.to_csv(OUT/'absolute_intervals.csv',index=False)
        pair_df.to_csv(OUT/'paired_lr.csv',index=False)

    write_descriptive_tables(cohorts)
    repro = pd.read_csv(inputs[7]).query("version == 'tuned'")
    repro_rows=[]
    for name in ['RF','XGBoost','CatBoost','NN']:
        a=repro.loc[(repro.model==name)&(repro.flow=='split_first')].iloc[0]
        b=repro.loc[(repro.model==name)&(repro.flow=='smote_first')].iloc[0]
        repro_rows.append([{'NN':'Neural network'}.get(name,name),f'{a.paper_AUROC:.2f}',
                           interval(a.AUROC,a.AUROC_low,a.AUROC_high),f'{b.AUROC:.3f}'])
    save_table('table2','Discrimination under two reconstruction workflows using the published PPMI table',
               ['Algorithm','Reported AUC','Workflow A AUC (95% CI)','Workflow B AUC'],repro_rows,
               r'p{0.17\textwidth}p{0.14\textwidth}p{0.38\textwidth}p{0.20\textwidth}',
               'Workflow A split participants before training-only SMOTE: 1,023 development and 256 observed test records (98 COG-positive). '
               'Workflow B applied SMOTE before splitting: 1,260 development and 316 test records (158 positive), including 63 synthetic test records. '
               'Workflow B is a non-independent diagnostic scenario; no valid independent-test CI is assigned. '
               'Reported AUCs are from Wang and Yan (2025). Hyperparameters were tuned separately within each workflow; the workflows are not paired tests on identical participants.')
    main_rows=[]
    for cohort in COHORTS:
        for name in LABELS:
            r=comparison.loc[(comparison.cohort==cohort)&(comparison.model==name)&(comparison.operating_point=='OOF_Youden')].iloc[0]
            ci=abs_df.loc[(abs_df.cohort==cohort)&(abs_df.model==name)&(abs_df.metric=='AUROC')].iloc[0]
            main_rows.append(['Internal' if cohort==COHORTS[0] else 'External',
                             {'Age_LR':'Age','Basic_LR':'Basic (5)','LR':'Basic + blood (21)'}[name],
                             interval(r.AUROC,ci.low,ci.high),f'{100*r.accuracy:.1f}',
                             f'{100*r.sensitivity:.1f}',f'{100*r.specificity:.1f}'])
    save_table('table3','Performance of logistic regression models with nested predictor sets',
               ['Sample','Predictors','AUC (95% CI)','Accuracy, %','Sensitivity, %','Specificity, %'],main_rows,
               r'p{0.09\textwidth}p{0.21\textwidth}p{0.29\textwidth}p{0.09\textwidth}p{0.10\textwidth}p{0.10\textwidth}',
               'All three L2-penalized logistic models were trained on the same 969 participants. Basic predictors were age, sex, height, weight and BMI; the 21-predictor model added 16 blood predictors. '
               'Each model used its training out-of-fold Youden threshold for both evaluation samples: age 0.502105, basic 0.499146, and basic + blood 0.432646. '
               'AUC intervals condition on the fitted models and use 2,000 outcome-stratified participant bootstrap samples.')
    pair_rows=[]
    for cohort in COHORTS:
        for model,reference in [('LR','Basic_LR'),('Basic_LR','Age_LR'),('LR','Age_LR')]:
            vals=[]
            for metric in ['AUROC','Brier']:
                r=pair_df.loc[(pair_df.cohort==cohort)&(pair_df.model==model)&(pair_df.reference==reference)&(pair_df.metric==metric)].iloc[0]
                vals.append(interval(r.difference,r.low,r.high,4 if metric=='Brier' else 3))
            pair_rows.append(['Internal' if cohort==COHORTS[0] else 'External',
                             {'LR':'21','Basic_LR':'5'}[model]+' vs '+{'Basic_LR':'5','Age_LR':'1'}[reference],*vals])
    save_table('table3_differences','Paired incremental performance of the logistic models',
               ['Sample','Predictor comparison','AUC difference (95% CI)','Brier difference (95% CI)'],pair_rows,
               r'p{0.10\textwidth}p{0.18\textwidth}p{0.34\textwidth}p{0.32\textwidth}',
               'Differences are the first model minus the second model on the same participants. A positive AUC difference and a negative Brier difference favor the first model. '
               'The 21-versus-5 comparison is the principal blood-panel increment; its intervals are reused unchanged from the completed analysis. '
               'Age comparisons were calculated from saved predictions without model refitting. No equivalence margin or multiplicity-adjusted hypothesis test was specified.')
    panel_tables=[]
    for name in ['table3','table3_differences']:
        content=(TABLES/f'{name}.tex').read_text(encoding='utf-8')
        panel_tables.append(content[content.index(r'\begin{tabular}'):content.index(r'\end{tabular}')+len(r'\end{tabular}')])
    combined=[r'\begin{table}[p]\centering\ManuscriptTableStyle',
              r'\caption{\textbf{Performance and incremental value of logistic regression models with nested predictor sets.}}\label{tab:table3}',
              r'\textbf{A. Absolute performance}\par\medskip',panel_tables[0],
              r'\par\bigskip\textbf{B. Paired differences}\par\medskip',panel_tables[1],
              r'\par\medskip\begin{minipage}{\textwidth}\ManuscriptNoteStyle',
              'Basic predictors were age, sex, height, weight and BMI; the 21-predictor model added 16 blood predictors. '
              'All models used the same 969 training participants and were evaluated on 416 internal and 197 external participants. '
              'Classification thresholds maximized the training out-of-fold Youden index: age 0.502105, basic 0.499146, and basic + blood 0.432646. '
              'The same threshold was used in both evaluation samples. Differences are the first model minus the second; higher AUC and lower Brier are favorable. '
              'Intervals are conditional on the fitted models, based on 2,000 outcome-stratified participant bootstrap samples. '
              'The principal blood-panel comparison is 21 versus 5 predictors. No equivalence margin or multiplicity-adjusted hypothesis test was specified.',
              r'\end{minipage}\end{table}']
    (TABLES/'table3_combined.tex').write_text('\n'.join(combined)+'\n',encoding='utf-8')
    all_models = comparison.query("operating_point == 'OOF_Youden'").copy()
    for i,r in all_models.iterrows():
        if r.model in LABELS:
            ci=abs_df.loc[(abs_df.cohort==r.cohort)&(abs_df.model==r.model)&(abs_df.metric=='AUROC')].iloc[0]
            all_models.loc[i,['AUROC_low','AUROC_high']]=ci[['low','high']].to_numpy(float)
        else:
            ci=previous.loc[(previous.cohort==r.cohort)&(previous.model==r.model)&(previous.operating_point=='OOF_Youden')].iloc[0]
            all_models.loc[i,['AUROC_low','AUROC_high']]=ci[['AUROC_low','AUROC_high']].to_numpy(float)
    all_models.to_csv(OUT/'all_model_performance.csv',index=False)
    full_rows=[]
    for _,r in all_models.iterrows():
        full_rows.append(['Internal' if r.cohort==COHORTS[0] else 'External',
                          MODEL_LABELS[r.model],
                          interval(r.AUROC,r.AUROC_low,r.AUROC_high),f'{100*r.accuracy:.1f}',
                          f'{100*r.sensitivity:.1f}',f'{100*r.specificity:.1f}',f'{r.Brier:.4f}'])
    save_table('table_s_performance','Internal and external performance of all evaluated models',
               ['Sample','Model','AUC (95% CI)','Acc., %','Sens., %','Spec., %','Brier'],full_rows,
               r'p{0.09\textwidth}p{0.15\textwidth}p{0.28\textwidth}p{0.09\textwidth}p{0.09\textwidth}p{0.09\textwidth}p{0.09\textwidth}',
               'Internal n=416; external n=197. All models use the same training sample (n=969). '
               'Except the one- and five-predictor logistic references, all models use 21 predictors. '
               'Classification measures use the training out-of-fold Youden threshold. Acc., accuracy; Sens., sensitivity; Spec., specificity.',True)
    assert all(sha(ROOT / rel)==expected for rel,expected in before.items())
    write_json(OUT/'validation.json',dict(status='passed',models_refitted=0,
               bootstrap_replicates=N_BOOT,seed=SEED,absolute_interval_rows=len(abs_df),
               paired_interval_rows=len(pair_df),performance_rows=len(all_models),
               saved_IDs_labels_and_training_split_match=True,existing_LR_vs_basic_intervals_unchanged=True,
               source_hashes_unchanged=True))
    write_json(OUT/'manifest.json',dict(code_sha256=sha(__file__),input_sha256=before,
               mode='reuse_saved_intervals' if tables_only else 'bootstrap_and_tables',
               training_anthropometry=dict(height_below_100cm=int(cohorts['PPMI_train'].height.lt(100).sum()),
                   minimum_recorded_height_cm=float(cohorts['PPMI_train'].height.min()),
                   maximum_raw_BMI=float(cohorts['PPMI_train'].BMI.max())),
               python=sys.executable,python_version=platform.python_version(),
               versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn']},
               outputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(OUT.glob('*.csv'))}))
    print(pair_df.loc[pair_df.metric.eq('AUROC')].to_string(index=False),flush=True)
    print(json.dumps(dict(event='MANUSCRIPT_STATISTICS_COMPLETE',models_refitted=0,
                         interval_rows=len(abs_df)+len(pair_df),tables=len(list(TABLES.glob('*.tex'))))),flush=True)



def descriptive_display_rows(rows):
    """Group outcome components; keep statistic definitions in the table note."""
    components = ['MoCA <26 only', 'Cognitive item >=1 only', 'Both positive components']
    suffixes = [', n (%)', ', mean (SD)', ', median [Q1, Q3]']
    normalized = []
    for row in rows:
        label = row[0]
        for suffix in suffixes:
            if label.endswith(suffix):
                label = label[:-len(suffix)]
                break
        normalized.append([label, *row[1:]])
    lookup = {row[0]: row for row in normalized}
    ordered = []
    for row in normalized:
        if row[0] in components:
            continue
        ordered.append(row)
        if row[0] == 'COG-positive':
            ordered.extend(lookup[label] for label in components)
    return ordered


def write_descriptive_tables(cohorts):
    missing = []
    for feature in FEATURES:
        row = {'predictor':feature}
        for name,d in cohorts.items():
            row[name+'_missing'] = int(d[feature].isna().sum())
        missing.append(row)
    pd.DataFrame(missing).to_csv(OUT/'predictor_missingness.csv',index=False)
    rows = [['Participants',*[str(len(d)) for d in cohorts.values()]]]
    def count(mask,d):
        return f'{int(mask.sum())} ({100*mask.sum()/len(d):.1f}%)'
    rows += [['COG-positive',*[count(d.COG.eq(1),d) for d in cohorts.values()]],
             ['Male sex',*[count(d.sex.eq(1),d) for d in cohorts.values()]]]
    labels = {'age':'Age, years','height':'Height, cm','weight':'Weight, kg',
              'BMI':'BMI, kg/m2','MCATOT':'MoCA score'}
    units = {x:'10^9/L' for x in ['Basophils','Eosinophils','Lymphocytes','Monocytes','Neutrophils','Platelets','WBC']}
    units.update({'RBC':'10^12/L','Hemoglobin':'g/L','Total Protein':'g/L','Serum Uric Acid':'umol/L',
                  'Serum Chloride':'mmol/L','Serum Glucose':'mmol/L','Serum Potassium':'mmol/L','Serum Sodium':'mmol/L'})
    for feature in ['age','height','weight','BMI','MCATOT'] + LABS:
        values = []
        quantiles = []
        for d in cohorts.values():
            x = d[feature].dropna()
            if feature in ['age','MCATOT']:
                values.append(f'{x.mean():.2f} ({x.std(ddof=1):.2f})')
            else:
                q = x.quantile([.25,.5,.75]).to_numpy()
                quantiles.append(q)
                values.append(f'{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]')
        if quantiles and all(np.isfinite(q).all() and
                             np.isclose(q, np.rint(q), atol=1e-9, rtol=0).all()
                             for q in quantiles):
            values = [f'{q[1]:.0f} [{q[0]:.0f}, {q[2]:.0f}]' for q in quantiles]
        label = labels.get(feature,feature+(', '+units[feature] if feature in units else ''))
        rows.append([label,*values])
    for label,func in [
        ('MoCA <26 only',lambda d:d.MCATOT.lt(26)&d.NP1COG.eq(0)),
        ('Cognitive item >=1 only',lambda d:d.MCATOT.ge(26)&d.NP1COG.ge(1)),
        ('Both positive components',lambda d:d.MCATOT.lt(26)&d.NP1COG.ge(1)),
        ('Missing age',lambda d:d.age.isna()),
        ('Missing height/weight/BMI',lambda d:d.BMI.isna()),
        ('Any missing blood predictor',lambda d:d[LABS].isna().any(axis=1))]:
        rows.append([label,*[count(func(d),d) for d in cohorts.values()]])
    save_table('table1','Characteristics of the model-development and evaluation samples',
               ['Characteristic','PPMI training','PPMI internal','LY-PD'],descriptive_display_rows(rows),
               # Widths follow padded, unwrapped 9 pt cell maxima; preserve full page width.
               r'p{0.292945\textwidth}p{0.235685\textwidth}p{0.235685\textwidth}p{0.235685\textwidth}',
               'Values are n or n (%), mean (SD) for age and MoCA, and median [Q1, Q3] for anthropometry and blood predictors. '
               'Summaries use observed values before imputation or capping; percentages use the full sample denominator. '
               'The three positive-component rows partition COG-positive participants. Abbreviations: BMI, body mass index; '
               'MoCA, Montreal Cognitive Assessment; NLR, neutrophil-to-lymphocyte ratio; RBC, red blood cell count; WBC, white blood cell count; SD, standard deviation; Q1, first quartile; Q3, third quartile; PPMI, Parkinson\'s Progression Markers Initiative; LY-PD, Luoyang PD cohort; COG, composite cognitive-screening outcome.',True)
    missing_labels = {'age':'Age', 'sex':'Sex', 'height':'Height', 'weight':'Weight'}
    missing_rows=[[missing_labels.get(row['predictor'],row['predictor']),*[str(row[k+'_missing']) for k in cohorts]] for row in missing]
    save_table('table_s_missing','Missing predictor values before training-derived imputation',
               ['Predictor','PPMI training','PPMI internal','LY-PD'],missing_rows,
               r'p{0.37\textwidth}p{0.19\textwidth}p{0.19\textwidth}p{0.19\textwidth}',
               'Counts include missing values and invalid observations set to missing by uniform validity rules. '
               'No outcome component was imputed. \\textbf{Abbreviations:} BMI, body mass index; NLR, neutrophil-to-lymphocyte ratio.',False)


def refresh_descriptive_only():
    """Refresh current descriptive tables without any historical model outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    source=ROOT/'data/derived/cog_pm3/ppmi_cog.csv'
    ext_source=ROOT/'data/derived/cog_v1/fahhaust.csv'
    full,_=clean_sheet(pd.read_csv(source))
    ext,_=clean_sheet(pd.read_csv(ext_source))
    split=np.load(ROOT/'data/derived/cog_pm3/frozen_split.npz')
    indexed=full.set_index('PATNO',drop=False)
    cohorts={'PPMI_train':indexed.loc[split['train_id']], 'PPMI_internal':indexed.loc[split['internal_id']], 'FAHHAUST_external':ext}
    for name,d in zip(['training','internal','external'],cohorts.values()):
        z=np.load(ROOT/'data/derived/bayes100_input'/f'{name}.npz')
        assert np.array_equal(d.PATNO.to_numpy(),z['id'])
        assert np.array_equal(d.COG.to_numpy(int),z['y'])
        assert np.allclose(d[FEATURES].to_numpy(float),z['X'],equal_nan=True,rtol=1e-13,atol=1e-13)
    write_descriptive_tables(cohorts)
    rows=[]
    for label,d in zip(['Training','Internal','External'],cohorts.values()):
        for cog in [0,1]:
            g=d.loc[d.COG.eq(cog)];both=int((g.MCATOT.lt(26)&g.NP1COG.ge(1)).sum())
            rows.append([label,cog,len(g),f'{g.age.mean():.2f} ({g.age.std(ddof=1):.2f})',f'{g.MCATOT.mean():.2f}',f'{both} ({100*both/len(g):.1f}%)' if both else '0'])
    pd.DataFrame(rows,columns=['Sample','COG','n','Age, mean (SD)','Mean MoCA','Both positive']).to_csv(OUT/'table_s_case_mix_formatted.csv',index=False)
    write_json(OUT/'descriptive_manifest.json',dict(status='passed',input_manifest_sha256=sha(ROOT/'data/derived/bayes100_input/input_manifest.json'),participants_and_labels_match_frozen_inputs=True,models_refitted=0,outputs=['table1','table_s_missing','table_s_case_mix_formatted']))

if __name__ == '__main__':
    if '--descriptive-only' in sys.argv: refresh_descriptive_only()
    else: main()
