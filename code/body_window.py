"""Nearest complete anthropometric observation within a symmetric month window."""
import numpy as np
import pandas as pd


def nearest_body(index, vital, window=3):
    v = vital.copy()
    dt = pd.to_datetime(v.INFODT, format='%m/%Y', errors='coerce')
    v['vital_month'] = dt.dt.year * 12 + dt.dt.month
    for c in ['HTCM', 'WGTKG']:
        v[c] = pd.to_numeric(v[c], errors='coerce')
    v = v.loc[v.HTCM.gt(0) & v.WGTKG.gt(0) & v.vital_month.notna()].copy()
    keys = ['PATNO', 'EVENT_ID', 'vital_month']
    grouped = v.groupby(keys, dropna=False)
    values = grouped[['HTCM', 'WGTKG']].first()
    conflict = grouped[['HTCM', 'WGTKG']].nunique().gt(1).any(axis=1)
    values = values.loc[~conflict].reset_index()
    near = index[['PATNO', 'EVENT_ID', 'index_month']].merge(
        values, on='PATNO', suffixes=('', '_vital'))
    near['vital_gap'] = near.index_month - near.vital_month
    near['distance'] = near.vital_gap.abs()
    near['event_mismatch'] = near.EVENT_ID.ne(near.EVENT_ID_vital)
    near['future'] = near.vital_gap.lt(0)
    near = near.loc[near.distance.le(window)].sort_values(
        ['PATNO', 'distance', 'event_mismatch', 'future', 'vital_month', 'EVENT_ID_vital'],
        kind='stable').drop_duplicates('PATNO')
    chosen = index[['PATNO', 'EVENT_ID', 'index_month']].merge(
        near[['PATNO', 'HTCM', 'WGTKG', 'vital_month', 'vital_gap', 'EVENT_ID_vital']],
        on='PATNO', how='left', validate='one_to_one')
    chosen = chosen.rename(columns={'HTCM':'height', 'WGTKG':'weight'})
    chosen['BMI'] = chosen.weight / (chosen.height / 100) ** 2
    assert chosen.vital_gap.dropna().abs().le(window).all()
    assert chosen.PATNO.nunique() == len(index)
    return chosen, {'conflicting_visit_months_excluded':int(conflict.sum())}
