"""Reproducible input export without running historical predictive models."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common_preprocessing import LABS,clean_sheet,sha,write_json
from body_window import nearest_body

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['age','sex','height','weight','BMI']+LABS


def main(check_only=False):
    current=ROOT/'data/derived/cog_pm3'
    source=current/'ppmi_cog.csv';split_path=current/'frozen_split.npz'
    if not source.exists():
        if check_only:raise FileNotFoundError(source)
        original=pd.read_csv(ROOT/'data/derived/cog_v1/ppmi_cog.csv')
        link=pd.read_csv(ROOT/'data/derived/cog_v1/linkage_private.csv')
        vital=pd.read_csv(ROOT/'data/raw/ppmi_2026-10-14/Vital_Signs_13Oct2026.csv',low_memory=False)
        body,_=nearest_body(link,vital)
        updated=original.drop(columns=['height','weight']).merge(body[['PATNO','height','weight','BMI']],
                 on='PATNO',how='left',validate='one_to_one',sort=False)
        keep=[c for c in original if c not in ['height','weight']]
        pd.testing.assert_frame_equal(original[keep],updated[keep],check_exact=True)
        current.mkdir(parents=True,exist_ok=True);updated.to_csv(source,index=False)
        body.to_csv(current/'body_linkage_private.csv',index=False)
    ppmi,_=clean_sheet(pd.read_csv(source))
    ppmi=ppmi.loc[ppmi.COHORT_DEFINITION.eq("Parkinson's Disease")].reset_index(drop=True)
    assert (len(ppmi),int(ppmi.COG.sum()),ppmi.PATNO.nunique())==(1385,679,1385)
    tr,te=train_test_split(np.arange(len(ppmi)),test_size=.3,random_state=20261014,stratify=ppmi.COG)
    split=dict(train_id=ppmi.PATNO.to_numpy()[tr],internal_id=ppmi.PATNO.to_numpy()[te],
               y=ppmi.COG.to_numpy(int)[tr],yi=ppmi.COG.to_numpy(int)[te])
    if split_path.exists():
        frozen=np.load(split_path)
        assert all(np.array_equal(value,frozen[key]) for key,value in split.items()),'Existing split differs; do not overwrite it.'
    else:
        if check_only:raise FileNotFoundError(split_path)
        np.savez_compressed(split_path,**split)
    external_source=ROOT/'data/derived/cog_v1/fahhaust.csv'
    external,_=clean_sheet(pd.read_csv(external_source))
    assert (len(external),int(external.COG.sum()),external.PATNO.nunique())==(197,122,197)
    frames={'training':ppmi.iloc[tr],'internal':ppmi.iloc[te],'external':external}
    output=ROOT/'data/derived/bayes100_input'
    if not check_only:output.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():
        path=output/(name+'.npz')
        expected=dict(X=frame[FEATURES].to_numpy(float),y=frame.COG.to_numpy(int),id=frame.PATNO.to_numpy())
        if path.exists():
            existing=np.load(path)
            for key,value in expected.items():
                assert np.array_equal(value,existing[key],equal_nan=True),(name,key)
        else:
            if check_only:raise FileNotFoundError(path)
            np.savez_compressed(path,**expected)
    sources=[source,external_source,split_path]
    manifest=dict(features=FEATURES,COG_rule='MoCA<26 OR cognitive-item>=1; both components observed',seed=20261014,
                  counts={k:dict(n=len(v),positive=int(v.COG.sum())) for k,v in frames.items()},
                  source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                  files_sha256={p.name:sha(p) for p in sorted(output.glob('*.npz'))})
    manifest_path=output/'input_manifest.json'
    if manifest_path.exists():
        assert manifest==__import__('json').loads(manifest_path.read_text()),'Existing input manifest differs; no overwrite permitted.'
    else:
        if check_only:raise FileNotFoundError(manifest_path)
        write_json(manifest_path,manifest)
    print({'event':'BAYES100_INPUTS_CHECKED','source':'updated raw PPMI and released FAHHAUST',
           'counts':manifest['counts'],'existing_inputs_and_split_preserved':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    main(parser.parse_args().check)
