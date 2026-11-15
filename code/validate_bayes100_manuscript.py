"""Check final tables, source narratives, figures and compiled v3 deliverables."""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from common_preprocessing import sha,write_json
from display_labels import MODEL_LABELS, PREDICTOR_SET_LABELS

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manuscript'
OUT=M/'results/bayes100'
DISPLAY=M/'results/manuscript_statistics'


def main():
    numerical=json.loads((OUT/'validation.json').read_text())
    assert numerical['status']=='passed'
    absolute=pd.read_csv(OUT/'absolute_intervals.csv')
    paired=pd.read_csv(OUT/'paired_lr.csv')
    perf=pd.read_csv(OUT/'all_model_performance.csv')
    table2=(M/'tables/table2.tex').read_text()
    table3=(M/'tables/table3_combined.tex').read_text()
    results=(M/'sections/results.tex').read_text()
    for frame,point,precision in [(absolute,'value',None),(paired,'difference',None)]:
        for _,r in frame.iterrows():
            digits=3 if r.metric=='AUROC' else 4
            keys=[point] if point=='value' and r.metric=='Brier' else [point,'low','high']
            for k in keys:
                assert f'{r[k]:.{digits}f}' in table3,(r.to_dict(),k)
    for _,r in perf.loc[~perf.model.isin(['Age_LR','Basic_LR'])].iterrows():
        for k,digits in [('AUROC',3),('AUROC_low',3),('AUROC_high',3),('Brier',4)]:
            assert f'{r[k]:.{digits}f}' in table2,(r.model,r.cohort,k)
        for k in ['accuracy','sensitivity','specificity']:
            assert f'{100*r[k]:.1f}' in table2
    for table in [table2,table3]:
        assert r'\begin{tabular*}{\textwidth}' in table
        assert 'Accuracy, \\%' in table and 'Sensitivity, \\%' in table and 'Specificity, \\%' in table
    assert table2.count('Internal &')==table2.count('External &')==1
    assert table3.count('Internal &')==table3.count('External &')==2
    assert [MODEL_LABELS[key] for key in ['Age_LR','Basic_LR','LR']]==['LR - Set 1','LR - Set 2','LR']
    assert [PREDICTOR_SET_LABELS[key] for key in ['Age_LR','Basic_LR','LR']]==['Set 1','Set 2','Full set']
    assert 'lP{0.15\\textwidth}' not in table3 and 'Predictor set' in table3
    assert all(label in table3 for label in ['Set 1','Set 2','Full set'])
    methods=(M/'sections/methods.tex').read_text()
    assert 'First Affiliated Hospital of Henan University of Science and Technology' in methods
    assert 'Luoyang, China' in methods and 'labelled this dataset FAHHAUST-PD' in methods
    introduction=(M/'sections/introduction.tex').read_text()
    assert 'For brevity, we refer to this published cohort as the Luoyang PD cohort (LY-PD)' in introduction
    assert introduction.index('Luoyang PD cohort (LY-PD)') < introduction.index('We evaluated the incremental')
    assert "Early Parkinson's Disease Longitudinal Singapore (PALS)" in introduction
    assert 'The other five algorithms used Full set.' in methods
    assert "cognitive screening in Parkinson's disease: a cross-cohort evaluation" in (M/'manuscript.tex').read_text()
    assert '``FAHHAUST-PD' in (M/'sections/declarations.tex').read_text()
    assert all(s in methods for s in ['100 Optuna','mean of the five','all 969','outcome-based resampling'])
    assert not re.search(r'email|contacted',methods+'\n'+results,re.I)
    assert 'For an ancillary comparison' in methods
    assert 'without a paired effect test' in methods
    assert '63 synthetic test records' in results
    search=(M/'tables/table_s_search.tex').read_text()
    assert 'Standard TPE configuration' not in search
    assert all(term in search for term in ['kernel','min\\_samples\\_leaf','colsample\\_bytree','l2\\_leaf\\_reg','batch\\_size','log integer'])
    fixed_table=(M/'tables/table_s_fixed.tex').read_text()
    for _,r in perf.iterrows():
        assert f'{r.Brier:.4f} ({r.Brier_low:.4f} to {r.Brier_high:.4f})' in fixed_table
    uncertainty=json.loads((OUT/'uncertainty_validation.json').read_text())
    assert uncertainty['status']=='passed' and uncertainty['failed_bootstrap_fits']==0
    assert uncertainty['prediction_models_refitted']==uncertainty['threshold_updates']==uncertainty['input_updates']==0
    cal_intervals=pd.read_csv(OUT/'calibration_intervals.csv')
    classification=pd.read_csv(OUT/'classification_intervals.csv')
    assert len(cal_intervals)==16 and len(classification)==32
    calibration_table=(M/'tables/table_s_calibration.tex').read_text()
    for _,r in cal_intervals.iterrows():
        assert r.bootstrap_attempts==r.bootstrap_successes==2000
        assert f'{r.calibration_intercept_offset:.3f} ({r.offset_low:.3f} to {r.offset_high:.3f})' in calibration_table
        assert f'{r.calibration_slope:.3f} ({r.slope_low:.3f} to {r.slope_high:.3f})' in calibration_table
    for _,r in classification.iterrows():
        for key in ['accuracy','sensitivity','specificity']:
            assert 0<=r[key+'_low']<=r[key]<=r[key+'_high']<=1
            assert f'{r[key]*100:.1f} ({r[key+"_low"]*100:.1f} to {r[key+"_high"]*100:.1f})' in fixed_table
    assert 'Wilson' in methods and 'ordinary participant bootstrap' in methods
    assert 'Paired differences were calculated before rounding.' in table3
    for section in ['abstract','introduction','methods','results','discussion','conclusion']:
        text=(M/f'sections/{section}.tex').read_text()
        assert len(text)>200 and not re.search(r'TODO|TBD|INSERT HERE|PLACEHOLDER',text)
        assert not re.search(r'(?:age-only|basic|full-panel) LR|Basic LR \(5\)',text,re.I),section
    workflow=(M/'figures/figure1_study_design.svg').read_text()
    assert all(term in workflow for term in ['Set 1: 1 predictor','Set 2: 5 predictors','Full set: 21 predictors','LR - Set 1','LR - Set 2'])
    assert all(term not in workflow for term in ['Updated raw PPMI','hospital-icon','evaluation only','Accuracy and sensitivity'])
    svg=ET.fromstring(workflow);ns={'svg':'http://www.w3.org/2000/svg'}
    for x in ['75','508','941','1374']:
        rows=[t for t in svg.findall('svg:text',ns) if t.get('x')==x and t.get('y') in ['944','986','1028','1070']]
        assert len(rows)==4,(x,len(rows))
        for t in rows:
            assert t.get('font-size')==('30' if t.get('y')=='944' else '29')
            if t.get('y') in ['986','1028']:assert t.get('fill')=='#263744'
            if t.get('y')=='1070':assert t.get('fill')=='#596775'
    bottoms=[]
    for x in ['650','1250']:
        bottoms.append(max(float(r.get('y'))+float(r.get('height')) for r in svg.findall('svg:rect',ns)
                           if r.get('x')==x and r.get('width')=='500'))
    assert bottoms==[737,737],bottoms
    legends=(M/'figures/figure_legends.tex').read_text()
    assert legends.count(r'\textbf{Abbreviations:}')==4
    declarations=(M/'sections/declarations.tex').read_text()
    assert 'ChatGPT (OpenAI)' in declarations
    assert 'Codex' not in declarations and 'DeepSeek Harness' not in declarations
    assert 'finite search grids' not in (M/'sections/discussion.tex').read_text()
    for rel in ['figure_manifest.json']:
        record=json.loads((DISPLAY/rel).read_text())
        assert record['font_family']=='Arial' and record['figure2_points']==16
        for key in ['input_sha256','output_sha256']:
            for path,expected in record[key].items():assert sha(ROOT/path)==expected,path
    calfig=json.loads((OUT/'calibration_figure_manifest.json').read_text())
    assert calfig['model_or_threshold_updates']==0
    for key in ['input_sha256','output_sha256']:
        for path,expected in calfig[key].items():assert sha(ROOT/path)==expected,path
    presentation=json.loads((OUT/'presentation_figure_manifest.json').read_text())
    assert presentation['empirical_ROC_curves']==16
    assert presentation['model_refits']==presentation['threshold_updates']==0
    assert len(presentation['AUROC_checks'])==16 and all(r['agrees_with_retained_result'] for r in presentation['AUROC_checks'])
    for key in ['input_sha256','output_sha256']:
        for path,expected in presentation[key].items():assert sha(ROOT/path)==expected,path
    bins=pd.read_csv(OUT/'calibration_bins.csv')
    assert len(bins.groupby(['cohort','model']))==6
    assert all(n in [416,197] for n in bins.groupby(['cohort','model']).n.sum())
    control=ROOT/'manuscript/results/B128 Retained Measurement'
    checked=json.loads((control/'comparison_validation.json').read_text())
    assert checked['status']=='passed' and checked['primary_predictions_receipts_and_inputs_unchanged']
    assert checked['same_training_packages'] and checked['same_predictive_and_preprocessing_code']
    sensitivity=(M/'tables/table_s_anthropometry.tex').read_text()
    comparison=pd.read_csv(control/'clean_retained_historical_comparison.csv')
    assert len(comparison)==16
    for _,r in comparison.iterrows():
        for k,digits in [('AUROC_clean',3),('AUROC_retained',3),('Brier_clean',4),('Brier_retained',4)]:
            assert f'{r[k]:.{digits}f}' in sensitivity,(r.model,r.cohort,k)
    supp=(M/'supplementary_materials.tex').read_text()
    assert '100--250' in methods and '100--250' in supp
    assert 'outcome-stratified participant bootstrap' in methods
    assert 'Anthropometric-handling sensitivity analysis' in methods
    assert 'negligible' not in supp and 'artifact of our unified' not in supp
    assert r'\subsection*{Software versions}' in supp
    build=json.loads((DISPLAY/'build_validation.json').read_text())
    for rel,expected in build['source_sha256'].items():assert sha(ROOT/rel)==expected,rel
    for name,record in build['documents'].items():
        assert sha(ROOT/record['pdf'])==record['sha256']
        assert record['compiled_without_errors_or_undefined_references']
        assert not any('Overfull' in w for w in record['layout_warnings'])
        text=(ROOT/f'tmp/latex/{name}/extracted.txt').read_text(encoding='utf-8')
        aux=(ROOT/f'tmp/latex/{name}/{name}.aux').read_text(encoding='utf-8')
        labels=set(re.findall(r'\\newlabel\{tab:[^}]+\}\{\{(S?\d+)\}',aux))
        assert labels==({'1','2','3','4'} if name=='manuscript' else {f'S{i}' for i in range(1,8)})
        assert not re.search(r'TabPFN|TabICL|TabFM|TabM|RealMLP|LightGBM',text)
        assert not re.search(r'\bdoi\s*:|doi\.org',text,re.I)
        if name=='manuscript':
            assert set(re.findall(r'^\s*Figure (\d+)\.',text,re.M))=={'1','2','3','4'}
            authors=['Jing Wang','Yanfang Chen','Kunpeng Xia']
            normalized=' '.join(text.split())
            assert [normalized.index(author) for author in authors]==sorted(normalized.index(author) for author in authors)
            assert 'wangjing@xynu.edu.cn' in text
            assert all(term in text for term in ['Funding','Conflict of interest',
                       'Ethics approval and consent to participate','31900710','2024-496',
                       'Data availability','Code availability','pd-blood-cognitive-screening'])
            assert '31600862' not in text
            assert 'LY-PD' in text and 'Luoyang PD cohort' in text
        else:
            assert not re.search(r'Figure S\d+\.',text)
            assert 'Figure 4 of the main manuscript' in text
    write_json(OUT/'manuscript_consistency.json',dict(status='passed',
        results_source='completed 100-trial Bayesian searches',prediction_strategies=8,
        main_algorithms=['LR','SVM','RF','XGBoost','CatBoost','NN'],main_tables=4,
        supplementary_tables=7,main_figures=4,supplementary_figures=0,
        original_PPMI_used_for_primary_development=False,
        detailed_original_workflow_confined_to_supplement=True,ancillary_summary_reported_in_main=True,source_results_and_PDF_consistent=True,
        final_PDFs_in_manuscript=True,author_information='Three-author order confirmed by user on 2026-10-23; existing affiliations and correspondence retained',
        declarations='P22 funding with NSFC 31900710; original-source ethics and consent for PPMI and LY-PD (original FAHHAUST-PD); author contributions; public code; AI assistance',
        remaining_submission_details='Final author approval and journal-specific submission declarations'))
    print((OUT/'manuscript_consistency.json').read_text(),flush=True)


if __name__=='__main__':main()
