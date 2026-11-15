"""Reproduce the current ordered, COG-stratified split without fitting a model."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common_preprocessing import clean_sheet

ROOT=Path(__file__).resolve().parents[1]

def main(check=False):
    directory=ROOT/'data/derived/cog_pm3'
    frame,_=clean_sheet(pd.read_csv(directory/'ppmi_cog.csv'))
    frame=frame.loc[frame.COHORT_DEFINITION.eq("Parkinson's Disease")].reset_index(drop=True)
    assert len(frame)==1385 and frame.PATNO.nunique()==1385 and int(frame.COG.sum())==679
    train,test=train_test_split(np.arange(len(frame)),test_size=.3,
                                stratify=frame.COG.to_numpy(int),random_state=20261014)
    arrays=dict(train_id=frame.PATNO.to_numpy()[train],internal_id=frame.PATNO.to_numpy()[test],
                y=frame.COG.to_numpy(int)[train],yi=frame.COG.to_numpy(int)[test])
    assert (len(train),len(test),int(arrays['y'].sum()),int(arrays['yi'].sum()))==(969,416,475,204)
    destination=directory/'frozen_split.npz'
    if destination.exists():
        with np.load(destination) as old:
            assert all(np.array_equal(old[k],v) for k,v in arrays.items()),'Existing split differs; refusing to overwrite'
        print('Existing frozen split matches all four ordered arrays; no files changed.')
    elif check:
        raise FileNotFoundError(destination)
    else:
        # Exclusive creation protects an existing split, including a concurrent writer.
        with destination.open('xb') as output:np.savez_compressed(output,**arrays)
        print('Created frozen split for the authorized reproduction cohort; no models fitted.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    main(parser.parse_args().check)
