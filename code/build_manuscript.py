"""Compile the multi-file paper sections and supplement into manuscript/ PDFs."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
from common_preprocessing import sha, write_json

ROOT=Path(__file__).resolve().parents[1]
MANUSCRIPT=ROOT/'manuscript'
FACTORIAL='--factorial' in sys.argv or r'\newcommand{\IncrementFigure}' in (MANUSCRIPT/'figures/figure_legends.tex').read_text(encoding='utf-8')


def markdown_exports():
    bib='\n'.join(p.read_text(encoding='utf-8') for p in sorted(MANUSCRIPT.glob('*.bib')))
    references={}
    for key,body in re.findall(r'@\w+\{([^,]+),(.*?)\n\}',bib,re.S):
        doi=re.search(r'(?i)doi\s*=\s*\{([^}]+)\}',body)
        link='https://doi.org/'+doi[1] if doi else re.search(r'https?://[^}\s]+',body)[0]
        references[key]=f'[{key}]({link})'
    table_text={name:(MANUSCRIPT/f'tables/{name}.md').read_text(encoding='utf-8')
                for name in (['table1','table2','table3','table4'] if FACTORIAL else ['table1','table2','table3','table3_differences','table4'])}
    inline_tables={name:table_text[name] for name in table_text}
    if not FACTORIAL:inline_tables['table3_combined']=table_text['table3']+'\n'+table_text['table3_differences']
    figure_sources={r'\StudyDesignFigure':('Figure 1. Study design','figure1_factorial_design' if FACTORIAL else 'figure1_study_design'),r'\PerformanceFigure':('Figure 2. Discrimination','figure2_predictor_set_auroc' if FACTORIAL else 'figure2_performance_case_mix'),r'\ROCFigure':('Figure 3. ROC curves','figure3_roc'),r'\IncrementFigure':('Figure 3. Blood-panel increments','figure3_blood_panel_increment'),r'\CalibrationFigure':('Figure 4. LR calibration','figure4_calibration')}
    inline_figures={key:'!['+label+']('+(MANUSCRIPT/'figures'/(stem+'.png')).as_posix()+')' for key,(label,stem) in figure_sources.items()}
    sections={}
    for name in ['abstract','introduction','methods','results','discussion','conclusion','declarations']:
        text=(MANUSCRIPT/f'sections/{name}.tex').read_text(encoding='utf-8')
        text=re.sub(r'\\section\{([^}]+)\}',r'## \1',text)
        text=re.sub(r'\\section\*\{([^}]+)\}',r'## \1',text)
        text=re.sub(r'\\subsection\{([^}]+)\}',r'### \1',text)
        text=re.sub(r'\\cite\{([^}]+)\}',lambda m:', '.join(references[k] for k in m[1].split(',')),text)
        text=re.sub(r'\\emph\{([^}]+)\}',r'*\1*',text)
        text=re.sub(r'\\textbf\{([^}]+)\}',r'**\1**',text)
        text=re.sub(r'\\ref\{tab:([^}]+)\}',lambda m:{'table1':'1','table2':'2','table3':'3','table4':'4'}[m[1]],text)
        text=re.sub(r'\\ref\{fig:([^}]+)\}',lambda m:{'design':'1','performance':'2','roc':'3','increment':'3','calibration':'4'}[m[1]],text)
        text=re.sub(r'\\url\{([^}]+)\}',lambda m:'['+m[1]+']('+m[1]+')',text)
        text=re.sub(r'\\href\{([^}]+)\}\{([^}]+)\}',lambda m:'['+m[2]+']('+m[1].replace(r'\#','#')+')',text)
        for key,value in inline_tables.items():
            text=text.replace(r'\input{tables/'+key+'}',value)
        for key,value in inline_figures.items():
            text=text.replace(key,value)
        text=re.sub(r'\$([^$]+)\$',lambda m:r'\('+m[1]+r'\)',text)
        text=text.replace(r'\begin{abstract}','## Abstract').replace(r'\end{abstract}','').replace(r'\noindent','')
        text=text.replace('\\%','%').replace(r'\_','_').replace('~',' ').replace('--','–')
        text=text.replace('``','“').replace("''",'”')
        sections[name]=text
    (MANUSCRIPT/'methods_results.md').write_text(sections['methods']+'\n'+sections['results'],encoding='utf-8')
    tables=[]
    table_order=[('table1','Table 1'),('table2','Table 2'),('table3','Table 3'),('table4','Table 4')] if FACTORIAL else [('table1','Table 1'),('table2','Table 2'),('table3','Table 3, panel A'),('table3_differences','Table 3, panel B'),('table4','Table 4')]
    for name,label in table_order:
        text=table_text[name]
        tables.append('## '+label+'\n\n'+text)
    (MANUSCRIPT/'main_tables.md').write_text('\n'.join(tables),encoding='utf-8')
    main_source=(MANUSCRIPT/'manuscript.tex').read_text()
    title=re.search(r'\\title\{\\ManuscriptTitle\{([^}]+)\}\}',main_source)[1]
    authors=re.findall(r'([A-Za-z]+ [A-Za-z]+)\\textsuperscript\{([^}]+)\}',main_source)
    author_header=[]
    if authors:
        author_header=['**Authors:** '+'; '.join(f'{name} ({markers})' for name,markers in authors),'']
        affiliations=re.findall(r'\\textsuperscript\{(\d+)\}\\textit\{([^}]+)\}',main_source)
        author_header += [f'{number}. {text}' for number,text in affiliations]+['']
        email=re.search(r'mailto:([^}]+)',main_source)[1]
        corresponding=', '.join(name for name,markers in authors if '*' in markers)
        author_header += [f'**Corresponding author:** {corresponding}; [{email}](mailto:{email})','']
    full=['# '+title,'']+author_header+[sections[name] for name in sections]
    (MANUSCRIPT/'manuscript.md').write_text('\n'.join(full),encoding='utf-8')


def main():
    markdown_exports()
    compiler=shutil.which('latexmk')
    extractor=shutil.which('pdftotext')
    if compiler is None or extractor is None:
        raise RuntimeError('latexmk and pdftotext are required')
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in MANUSCRIPT.rglob('*')
             if p.is_file() and (p.suffix in ['.tex','.bib','.bst'] or (p.parent.name=='figures' and p.suffix in {'.pdf','.svg'}))}
    records={}
    for name in ['manuscript','supplementary_materials']:
        build=ROOT/'tmp/latex'/name
        build.mkdir(parents=True,exist_ok=True)
        cmd=[compiler,'-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error',
             f'-outdir={build.as_posix()}',name+'.tex']
        result=subprocess.run(cmd,cwd=MANUSCRIPT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (build/'build_stdout.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
        log=(build/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        pdf=build/(name+'.pdf')
        if result.returncode or not pdf.exists() or re.search(r'^!|(?:Reference|Citation).*undefined|There were undefined',log,re.M):
            raise RuntimeError(f'{name} compilation failed; see {build}')
        if 'Overfull \\hbox' in log or 'Overfull \\vbox' in log:
            raise RuntimeError(f'{name} has an overfull box; see {build}')
        text_path=build/'extracted.txt'
        subprocess.run([extractor,'-layout',str(pdf),str(text_path)],check=True,capture_output=True)
        text=text_path.read_text(encoding='utf-8')
        if name=='manuscript':
            required=['Abstract','Introduction','Methods','Results','Discussion','Conclusion','0.724','0.717','0.653',
                      '969','416','197','0.124','0.004','0.0163','0.0279','100','Figure 1.','Figure 2.','Figure 3.','Figure 4.','Table 4.']
        else:
            required=['Supplementary','S1','S2','S3','S4','S5','S6','S7','0.626','0.779','Figure 4 of the main manuscript']
        if FACTORIAL:
            required=['18','100','969','416','197','Figure 1.','Figure 2.','Figure 3.','Figure 4.','Table 4.','0.607','0.687'] if name=='manuscript' else ['S1','S2A','S2B','S3A','S3B','S4','S5A','S5B','S6','S7','S8','S9','Figure S1.','Figure S2.','Figure S3.','Figure S4.','Figure S5.']
        assert all(term in text for term in required),(name,'missing PDF content')
        assert all(sha(ROOT/rel)==expected for rel,expected in sources.items()),'Source changed during build'
        destination=MANUSCRIPT/(name+'.pdf')
        staged=destination.with_suffix('.pdf.new')
        shutil.copy2(pdf,staged)
        os.replace(staged,destination)
        records[name]=dict(pdf=destination.relative_to(ROOT).as_posix(),sha256=sha(destination),
                           pages=int(re.findall(r'Output written on .*?\((\d+) pages?',log,re.S)[-1]),
                           compiled_without_errors_or_undefined_references=True,
                           required_source_content_in_PDF=True,
                           layout_warnings=re.findall(r'(?:Overfull|Underfull) \\hbox[^\n]*',log))
        print(json.dumps({name:records[name]}),flush=True)
    validation_path=MANUSCRIPT/('results/B129 Full Factorial/build_validation.json' if FACTORIAL else 'results/manuscript_statistics/build_validation.json')
    write_json(validation_path,
               dict(status='passed',source_sha256=sources,documents=records))


if __name__=='__main__':
    main()
