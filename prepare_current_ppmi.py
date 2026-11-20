"""Build one earliest eligible paired observation per current PPMI PD participant."""
import os
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from pilot import ROOT, LABS, clean_sheet, write_json

def resolve_ppmi_raw():
    env_dir = os.environ.get('PPMI_DATA_DIR')
    if env_dir:
        p = Path(env_dir).expanduser()
        for cand in (p, p / 'blood_clinical_20261014', p / 'ppmi'):
            if cand.is_dir() and any(cand.glob('Participant_Status*.csv')):
                return cand
        return p
    for cand in (ROOT / 'data/raw/ppmi', ROOT / 'data/raw'):
        if cand.is_dir() and any(cand.glob('Participant_Status*.csv')):
            return cand
    return ROOT / 'data/raw/ppmi'

RAW = resolve_ppmi_raw()
OUT = ROOT / 'data/derived/current_ppmi'
UNITS = {x: 'GI/L' for x in ['Basophils','Eosinophils','Lymphocytes','Monocytes','Neutrophils','Platelets','WBC']}
UNITS.update({'RBC':'TI/L','Hemoglobin':'g/L','Total Protein':'g/L',
 'Serum Chloride':'mmol/L','Serum Glucose':'mmol/L','Serum Potassium':'mmol/L',
 'Serum Sodium':'mmol/L','Serum Uric Acid':'umol/L'})


def load(pattern, **kwargs):
    matches = list(RAW.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Ambiguous source pattern {pattern}')
    return pd.read_csv(matches[0], low_memory=False, **kwargs)


def month(series):
    d = pd.to_datetime(series, format='%m/%Y', errors='coerce')
    return d.dt.year * 12 + d.dt.month


def collapse(df, keys, value):
    """Retain a unique known value; conflicting legal values become unknown."""
    g = df.groupby(keys, dropna=False)[value]
    out = g.first().to_frame()
    out.loc[g.nunique().gt(1), value] = np.nan
    return out.reset_index()


def screen_sex(series):
    values = pd.to_numeric(series, errors='coerce')
    return values.where(values.isin([0, 1]))


def choose_pairs(cog, blood):
    pairs = cog.merge(blood, on='PATNO', suffixes=('_cog','_blood'))
    pairs['gap'] = pairs.cog_month - pairs.blood_month
    pairs = pairs[pairs.gap.between(0, 2)].copy()
    pairs['event_mismatch'] = pairs.EVENT_ID_cog.ne(pairs.EVENT_ID_blood)
    pairs = pairs.sort_values(['PATNO','cog_month','gap','event_mismatch','valid_lab_count'],
                              ascending=[True,True,True,True,False], kind='stable')
    return pairs.drop_duplicates('PATNO', keep='first').copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = load('Participant_Status*.csv')
    pd_status = status[pd.to_numeric(status.COHORT, errors='coerce').eq(1)].copy()
    if pd_status.PATNO.duplicated().any():
        raise ValueError('Nonunique current PD status')
    pd_ids = set(pd_status.PATNO)
    sources = []
    for rank, pattern in [(0,'Blood_Chemistry___Hematology_*.csv'),(1,'Blood_Chemistry___Hematology-Archived_*.csv')]:
        b = load(pattern)
        b = b[b.PATNO.isin(pd_ids) & b.LTSTNAME.isin(UNITS)].copy()
        b['blood_month'] = month(b.LCOLLDT)
        b['value'] = pd.to_numeric(b.LSIRES, errors='coerce')
        expected = b.LTSTNAME.map(UNITS)
        b.loc[b.LSIUNIT.ne(expected), 'value'] = np.nan
        b['source_rank'] = rank
        sources.append(b[['PATNO','EVENT_ID','blood_month','LTSTNAME','value','source_rank']])
    blood = pd.concat(sources, ignore_index=True).dropna(subset=['blood_month'])
    keys = ['PATNO','EVENT_ID','blood_month','LTSTNAME']
    ranks = blood.groupby(keys).source_rank.transform('min')
    blood = blood[blood.source_rank.eq(ranks)]
    conflicting_blood = int(blood.groupby(keys).value.nunique().gt(1).sum())
    blood = collapse(blood, keys, 'value')
    blood = blood.pivot(index=['PATNO','EVENT_ID','blood_month'], columns='LTSTNAME', values='value').reset_index()
    blood['valid_lab_count'] = blood[list(UNITS)].notna().sum(axis=1)
    blood = blood[blood.valid_lab_count.gt(0)]
    mo = load('Montreal*.csv')
    mo = mo[mo.PATNO.isin(pd_ids)].copy()
    mo['cog_month'] = month(mo.INFODT)
    mo['MCATOT'] = pd.to_numeric(mo.MCATOT, errors='coerce')
    mo.loc[~mo.MCATOT.between(0,30), 'MCATOT'] = np.nan
    mo = collapse(mo.dropna(subset=['cog_month']), ['PATNO','EVENT_ID','cog_month'], 'MCATOT').dropna(subset=['MCATOT'])
    chosen = choose_pairs(mo, blood)
    chosen = chosen.rename(columns={'EVENT_ID_cog':'EVENT_ID'})
    nup = load('MDS-UPDRS_Part_I_*.csv')
    nup = nup[nup.PATNO.isin(pd_ids)].copy()
    nup['item_month'] = month(nup.INFODT)
    nup['NP1COG'] = pd.to_numeric(nup.NP1COG, errors='coerce')
    nup.loc[~nup.NP1COG.isin([0,1,2,3,4]), 'NP1COG'] = np.nan
    nup = collapse(nup.dropna(subset=['item_month']), ['PATNO','EVENT_ID','item_month'], 'NP1COG')
    near = chosen[['PATNO','EVENT_ID','cog_month']].merge(nup, on='PATNO', suffixes=('','_item'))
    near['item_gap'] = (near.item_month - near.cog_month).abs()
    near['item_event_mismatch'] = near.EVENT_ID.ne(near.EVENT_ID_item)
    near = near[near.item_gap.le(1) & near.NP1COG.notna()].sort_values(
        ['PATNO','item_event_mismatch','item_gap','item_month']).drop_duplicates('PATNO')
    chosen = chosen.merge(near[['PATNO','NP1COG']], on='PATNO', how='left', validate='one_to_one')
    age = collapse(load('Age_at_visit*.csv'), ['PATNO','EVENT_ID'], 'AGE_AT_VISIT')
    chosen = chosen.merge(age, on=['PATNO','EVENT_ID'], how='left', validate='many_to_one')
    chosen['age'] = pd.to_numeric(chosen.AGE_AT_VISIT, errors='coerce')
    demo = load('Demographics*.csv')
    # Code_List SCREEN/SEX is 0=female, 1=male (not PARTICIPANT_PROFILE's 1/2).
    demo['sex'] = screen_sex(demo.SEX)
    demo = collapse(demo, ['PATNO'], 'sex')
    chosen = chosen.merge(demo, on='PATNO', how='left', validate='one_to_one')
    vital = load('Vital_Signs*.csv')
    vital['vital_month'] = month(vital.INFODT)
    for c in ['WGTKG','HTCM']:
        vital[c] = pd.to_numeric(vital[c], errors='coerce')
    vital = vital[vital.WGTKG.gt(0) & vital.HTCM.gt(0)]
    near = chosen[['PATNO','EVENT_ID','cog_month']].merge(vital[['PATNO','EVENT_ID','vital_month','WGTKG','HTCM']], on='PATNO', suffixes=('','_vital'))
    near['vital_gap'] = (near.cog_month - near.vital_month).abs()
    near['vital_event_mismatch'] = near.EVENT_ID.ne(near.EVENT_ID_vital)
    near = near[near.vital_gap.le(2)].sort_values(['PATNO','vital_event_mismatch','vital_gap','vital_month'])
    # Conflicting same-occasion measurements remain missing before choosing a visit.
    near = collapse(near, ['PATNO','vital_month','vital_gap','vital_event_mismatch'], 'WGTKG').merge(
        collapse(vital.rename(columns={'INFODT':'unused'}), ['PATNO','vital_month'], 'HTCM'),
        on=['PATNO','vital_month'], how='left')
    near = near.sort_values(['PATNO','vital_event_mismatch','vital_gap','vital_month']).drop_duplicates('PATNO')
    chosen = chosen.merge(near[['PATNO','WGTKG','HTCM','vital_month']], on='PATNO', how='left', validate='one_to_one')
    chosen['height'], chosen['weight'] = chosen.HTCM, chosen.WGTKG
    chosen['COHORT'], chosen['COHORT_DEFINITION'] = 1, "Parkinson's Disease"
    chosen['COG'] = np.where(chosen.MCATOT.lt(26), 1., np.where(chosen.NP1COG.notna(), chosen.NP1COG.ge(1).astype(float), np.nan))
    chosen['NLR'] = chosen.Neutrophils / chosen.Lymphocytes.where(chosen.Lymphocytes.gt(0))
    cols = ['PATNO','COHORT','COHORT_DEFINITION','MCATOT','NP1COG','COG'] + LABS + ['age','sex','height','weight']
    chosen[cols].to_csv(OUT/'ppmi_current.csv',index=False)
    chosen[['PATNO','EVENT_ID','EVENT_ID_blood','cog_month','blood_month','vital_month','gap','valid_lab_count']].to_csv(OUT/'linkage_private.csv',index=False)
    external = pd.read_excel(ROOT/'data/raw/published_cohorts.xlsx',sheet_name='FAHHAUST-PD').iloc[:,:26]
    external.to_csv(OUT/'fahhaust.csv',index=False)
    _, qc = clean_sheet(chosen[cols])
    summary = {'data_export_date':'2026-10-13','download_date':'2026-10-14',
      'current_PD_participants':len(pd_status),'PD_with_recorded_MoCA':mo.PATNO.nunique(),
      'PD_with_blood_sample':blood.PATNO.nunique(),'paired_participants':len(chosen),
      'MoCA_lt26':int(chosen.MCATOT.lt(26).sum()),'COG_positive':int(chosen.COG.eq(1).sum()),
      'COG_unknown':int(chosen.COG.isna().sum()),'blood_conflicting_test_groups':conflicting_blood,
      'vitals_after_cognition_month':int(chosen.vital_month.gt(chosen.cog_month).sum()),
      'lag_months':chosen.gap.value_counts().sort_index().astype(int).to_dict(),
      'valid_direct_lab_counts':chosen.valid_lab_count.value_counts().sort_index().astype(int).to_dict(),
      'QC':qc,'rules':'One earliest eligible observation per current PD participant; blood in same/prior 2 calendar months; time precision is month.',
      'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RAW.glob('*.csv'))}}
    write_json(ROOT/'results/current_ppmi_feasibility.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['QC','inputs']},ensure_ascii=False))


if __name__ == '__main__':
    main()
