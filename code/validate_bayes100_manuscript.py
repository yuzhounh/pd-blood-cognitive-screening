"""Check final tables, source narratives, figures and compiled v3 deliverables."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from common_preprocessing import sha,write_json

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
    methods=(M/'sections/methods.tex').read_text()
    assert all(s in methods for s in ['100 Optuna','mean of the five','all 969','outcome-based resampling'])
    assert not re.search(r'Workflow [AB]|1,279|1279|reconstruction|email|contacted',methods+'\n'+results,re.I)
    for section in ['abstract','introduction','methods','results','discussion','conclusion']:
        text=(M/f'sections/{section}.tex').read_text()
        assert len(text)>200 and not re.search(r'TODO|TBD|INSERT HERE|PLACEHOLDER',text)
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
    bins=pd.read_csv(OUT/'calibration_bins.csv')
    assert len(bins.groupby(['cohort','model']))==6
    assert all(n in [416,197] for n in bins.groupby(['cohort','model']).n.sum())
    build=json.loads((DISPLAY/'build_validation.json').read_text())
    for rel,expected in build['source_sha256'].items():assert sha(ROOT/rel)==expected,rel
    for name,record in build['documents'].items():
        assert sha(ROOT/record['pdf'])==record['sha256']
        assert record['compiled_without_errors_or_undefined_references']
        assert not any('Overfull' in w for w in record['layout_warnings'])
        text=(ROOT/f'tmp/latex/{name}/extracted.txt').read_text(encoding='utf-8')
        aux=(ROOT/f'tmp/latex/{name}/{name}.aux').read_text(encoding='utf-8')
        labels=set(re.findall(r'\\newlabel\{tab:[^}]+\}\{\{(S?\d+)\}',aux))
        assert labels==({'1','2','3'} if name=='manuscript' else {f'S{i}' for i in range(1,8)})
        assert not re.search(r'TabPFN|TabICL|TabFM|TabM|RealMLP|LightGBM',text)
        assert not re.search(r'\bdoi\s*:|doi\.org',text,re.I)
        if name=='manuscript':
            assert set(re.findall(r'^\s*Figure (\d+)\.',text,re.M))=={'1','2'}
            authors=['Jing Wang','Yanfang Chen','Kunpeng Xia']
            normalized=' '.join(text.split())
            assert [normalized.index(author) for author in authors]==sorted(normalized.index(author) for author in authors)
            assert 'wangjing@xynu.edu.cn' in text
            assert all(term in text for term in ['Funding','Conflict of interest',
                       'Ethics approval and consent to participate','31900710','31600862','2024-496'])
        else:assert 'Figure S1.' in text
    write_json(OUT/'manuscript_consistency.json',dict(status='passed',
        results_source='completed 100-trial Bayesian searches',prediction_strategies=8,
        main_algorithms=['LR','SVM','RF','XGBoost','CatBoost','NN'],main_tables=3,
        supplementary_tables=7,main_figures=2,supplementary_figures=1,
        original_PPMI_used_for_primary_development=False,
        original_workflow_confined_to_supplement=True,source_results_and_PDF_consistent=True,
        final_PDFs_in_manuscript=True,author_information='Three-author order confirmed by user on 2026-10-23; existing affiliations and correspondence retained',
        declarations='E01 funding and conflict statements; source ethics and consent for PPMI and FAHHAUST; de-identified secondary analysis',
        remaining_submission_details='Author contributions, public code location and journal-specific submission metadata'))
    print((OUT/'manuscript_consistency.json').read_text(),flush=True)


if __name__=='__main__':main()
