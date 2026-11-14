"""Current raw PPMI, complete ascertainment of the published COG OR endpoint."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pilot import ROOT, LABS, clean_sheet, write_json
from prepare_current_ppmi import RAW, UNITS, load, month, collapse, screen_sex

OUT = ROOT / 'data/derived/cog_v1'


def cog_target(moca, cognitive_item):
    m = np.asarray(moca, dtype=float)
    c = np.asarray(cognitive_item, dtype=float)
    valid = np.isfinite(m) & (m >= 0) & (m <= 30) & np.isin(c, [0,1,2,3,4])
    return np.where(valid, (m < 26) | (c >= 1), np.nan)


def complete_episodes(mo, nup):
    joined = mo.merge(nup, on='PATNO', suffixes=('_moca','_item'))
    joined['component_gap'] = (joined.moca_month - joined.item_month).abs()
    joined = joined[joined.component_gap.le(2)].copy()
    joined['index_month'] = joined[['moca_month','item_month']].max(axis=1)
    joined['EVENT_ID'] = np.where(joined.item_month.ge(joined.moca_month),
                                  joined.EVENT_ID_item, joined.EVENT_ID_moca)
    joined['component_mismatch'] = joined.EVENT_ID_moca.ne(joined.EVENT_ID_item)
    joined = joined.sort_values(['PATNO','index_month','EVENT_ID','component_mismatch',
                                 'component_gap','moca_month','item_month'],
                                ascending=[True,True,True,True,True,False,False])
    return joined.drop_duplicates(['PATNO','index_month','EVENT_ID'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = load('Participant_Status*.csv')
    status = status[status.COHORT.isin([1,2,3,4])].copy()
    if status.PATNO.duplicated().any():
        raise ValueError('Current status must be unique per participant')
    ids = set(status.PATNO)
    pieces = []
    for rank, pattern in [(0,'Blood_Chemistry___Hematology_*.csv'),(1,'Blood_Chemistry___Hematology-Archived_*.csv')]:
        b = load(pattern)
        b = b[b.PATNO.isin(ids) & b.LTSTNAME.isin(UNITS)].copy()
        b['blood_month'] = month(b.LCOLLDT)
        b['value'] = pd.to_numeric(b.LSIRES, errors='coerce')
        b.loc[b.LSIUNIT.ne(b.LTSTNAME.map(UNITS)), 'value'] = np.nan
        b['rank'] = rank
        pieces.append(b[['PATNO','EVENT_ID','blood_month','LTSTNAME','value','rank']])
    b = pd.concat(pieces, ignore_index=True).dropna(subset=['blood_month'])
    keys = ['PATNO','EVENT_ID','blood_month','LTSTNAME']
    b = b[b['rank'].eq(b.groupby(keys)['rank'].transform('min'))]
    b = collapse(b, keys, 'value').pivot(index=['PATNO','EVENT_ID','blood_month'],
                                        columns='LTSTNAME', values='value').reset_index()
    b['n_labs'] = b[list(UNITS)].notna().sum(axis=1)
    b = b[b.n_labs.gt(0)].rename(columns={'EVENT_ID':'blood_event'})
    mo = load('Montreal*.csv')
    mo = mo[mo.PATNO.isin(ids)].copy()
    mo['moca_month'] = month(mo.INFODT)
    mo['MCATOT'] = pd.to_numeric(mo.MCATOT, errors='coerce')
    mo.loc[~mo.MCATOT.between(0,30), 'MCATOT'] = np.nan
    mo = collapse(mo.dropna(subset=['moca_month']), ['PATNO','EVENT_ID','moca_month'], 'MCATOT').dropna(subset=['MCATOT'])
    nu = load('MDS-UPDRS_Part_I_*.csv')
    nu = nu[nu.PATNO.isin(ids)].copy()
    nu['item_month'] = month(nu.INFODT)
    nu['NP1COG'] = pd.to_numeric(nu.NP1COG, errors='coerce')
    nu.loc[~nu.NP1COG.isin([0,1,2,3,4]), 'NP1COG'] = np.nan
    nu = collapse(nu.dropna(subset=['item_month']), ['PATNO','EVENT_ID','item_month'], 'NP1COG').dropna(subset=['NP1COG'])
    episodes = complete_episodes(mo, nu)
    candidates = episodes.merge(b, on='PATNO')
    candidates['blood_gap'] = candidates.index_month - candidates.blood_month
    candidates = candidates[candidates.blood_gap.between(0,2) & candidates.blood_month.le(candidates.moca_month)].copy()
    candidates['blood_event_mismatch'] = candidates.blood_event.ne(candidates.EVENT_ID)
    candidates = candidates.sort_values(['PATNO','index_month','component_mismatch','component_gap',
                          'blood_gap','blood_event_mismatch','n_labs'],
                          ascending=[True,True,True,True,True,True,False], kind='stable')
    chosen = candidates.drop_duplicates('PATNO').copy()
    chosen = chosen.merge(status[['PATNO','COHORT','COHORT_DEFINITION']], on='PATNO', validate='one_to_one')
    age = collapse(load('Age_at_visit*.csv'), ['PATNO','EVENT_ID'], 'AGE_AT_VISIT')
    chosen = chosen.merge(age, on=['PATNO','EVENT_ID'], how='left', validate='many_to_one')
    chosen['age'] = pd.to_numeric(chosen.AGE_AT_VISIT, errors='coerce')
    demo = load('Demographics*.csv')
    demo['sex'] = screen_sex(demo.SEX)
    chosen = chosen.merge(collapse(demo, ['PATNO'], 'sex'), on='PATNO', how='left', validate='one_to_one')
    v = load('Vital_Signs*.csv')
    v['vital_month'] = month(v.INFODT)
    for c in ['WGTKG','HTCM']:
        v[c] = pd.to_numeric(v[c], errors='coerce')
    v = v[v.WGTKG.gt(0) & v.HTCM.gt(0)]
    near = chosen[['PATNO','EVENT_ID','index_month']].merge(v[['PATNO','EVENT_ID','vital_month','WGTKG','HTCM']],
                                                     on='PATNO', suffixes=('','_vital'))
    near['vital_gap'] = near.index_month - near.vital_month
    near['vital_mismatch'] = near.EVENT_ID.ne(near.EVENT_ID_vital)
    near = near[near.vital_gap.between(0,3)]
    gkeys = ['PATNO','vital_month','vital_gap','vital_mismatch']
    near = collapse(near, gkeys, 'WGTKG').merge(collapse(near, gkeys, 'HTCM'), on=gkeys)
    near = near.sort_values(['PATNO','vital_mismatch','vital_gap','vital_month']).drop_duplicates('PATNO')
    chosen = chosen.merge(near[['PATNO','WGTKG','HTCM','vital_month']], on='PATNO', how='left', validate='one_to_one')
    chosen['height'], chosen['weight'] = chosen.HTCM, chosen.WGTKG
    chosen['NLR'] = chosen.Neutrophils / chosen.Lymphocytes.where(chosen.Lymphocytes.gt(0))
    chosen['COG'] = cog_target(chosen.MCATOT, chosen.NP1COG)
    if chosen.COG.isna().any():
        raise ValueError('Unascertained outcome reached complete COG cohort')
    cols = ['PATNO','COHORT','COHORT_DEFINITION','MCATOT','NP1COG','COG'] + LABS + ['age','sex','height','weight']
    chosen[cols].to_csv(OUT/'ppmi_cog.csv', index=False)
    chosen[['PATNO','EVENT_ID','index_month','moca_month','item_month','blood_month','vital_month']].to_csv(OUT/'linkage_private.csv', index=False)
    e = pd.read_excel(ROOT/'data/raw/published_cohorts.xlsx',sheet_name='FAHHAUST-PD').iloc[:,:26]
    expected = cog_target(e.MCATOT,e.NP1COG)
    if not np.array_equal(e.COG.to_numpy(),expected):
        raise ValueError('Published external COG no longer matches frozen OR definition')
    e.to_csv(OUT/'fahhaust.csv',index=False)
    _, qc = clean_sheet(chosen[cols])
    pd_chosen = chosen[chosen.COHORT.eq(1)]
    summary = {'endpoint':'COG = (MCATOT < 26) OR (NP1COG >= 1), both components observed',
     'data_export_date':'2026-10-13','current_status_n':len(status),
     'current_PD_n':int(status.COHORT.eq(1).sum()),'all_complete_paired_n':len(chosen),
     'all_positive_n':int(chosen.COG.sum()),'PD_complete_paired_n':len(pd_chosen),
     'PD_positive_n':int(pd_chosen.COG.sum()),'external_n':len(e),'external_positive_n':int(e.COG.sum()),
     'cohorts':chosen.groupby('COHORT_DEFINITION').agg(n=('PATNO','size'),positive=('COG','sum')).astype(int).to_dict('index'),
     'component_month_gap':chosen.component_gap.value_counts().sort_index().astype(int).to_dict(),
     'blood_month_gap':chosen.blood_gap.value_counts().sort_index().astype(int).to_dict(),
     'missing_or_invalid_after_QC':qc['missing_after'],
     'current_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob('*.csv'))},
     'build_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(ROOT/'manuscript/results/cog_feasibility.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['current_source_hashes','missing_or_invalid_after_QC']},ensure_ascii=False))


if __name__ == '__main__':
    main()
