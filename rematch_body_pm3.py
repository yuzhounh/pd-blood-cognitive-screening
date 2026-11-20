"""Update body measurements only; preserve cohort, outcome and exact split order."""
from pathlib import Path
import numpy as np
import pandas as pd
from body_window import nearest_body
from common_preprocessing import LABS, clean_sheet, sha, write_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'data/derived/cog_pm3'
RESULTS = ROOT/'results/benchmark21_pm3'
FEATURES = ['age','sex','height','weight','BMI'] + LABS


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = ROOT/'data/derived/cog_v1/ppmi_cog.csv'
    link_source = ROOT/'data/derived/cog_v1/linkage_private.csv'
    from prepare_current_ppmi import RAW as PPMI_RAW
    vital_source = next(PPMI_RAW.glob('Vital_Signs_*.csv'))
    split_source = ROOT/'data/derived/benchmark21/LR.npz'
    original = pd.read_csv(source)
    link = pd.read_csv(link_source)
    vital = pd.read_csv(vital_source, low_memory=False)
    body, qc = nearest_body(link, vital)
    updated = original.drop(columns=['height','weight']).merge(
        body[['PATNO','height','weight','BMI']], on='PATNO', how='left',
        validate='one_to_one', sort=False)
    unchanged = [c for c in original.columns if c not in ['height','weight']]
    pd.testing.assert_frame_equal(original[unchanged], updated[unchanged], check_exact=True)
    assert len(updated) == 3957 and updated.PATNO.nunique() == 3957
    updated.to_csv(OUT/'ppmi_cog.csv', index=False)
    body.to_csv(OUT/'body_linkage_private.csv', index=False)
    prior = np.load(split_source)
    np.savez_compressed(OUT/'frozen_split.npz', train_id=prior['train_id'],
        internal_id=prior['internal_id'], y=prior['y'], yi=prior['yi'])
    old,_ = clean_sheet(original)
    new,_ = clean_sheet(updated)
    external,_ = clean_sheet(pd.read_csv(ROOT/'data/derived/cog_v1/fahhaust.csv'))
    rows, summary = [], {}
    for label, key in [('PPMI_train','train_id'),('PPMI_internal','internal_id')]:
        a = old.set_index('PATNO').loc[prior[key]]
        b = new.set_index('PATNO').loc[prior[key]]
        assert np.array_equal(b.COG.to_numpy(int),prior['y' if key=='train_id' else 'yi'])
        summary[label] = dict(n=len(b), body_missing_before=int(a.BMI.isna().sum()),
            body_missing_after=int(b.BMI.isna().sum()),
            newly_available=int((a.BMI.isna()&b.BMI.notna()).sum()),
            existing_record_changed=int((a.BMI.notna()&b.BMI.notna()&
                ~np.isclose(a.BMI,b.BMI,rtol=0,atol=1e-10,equal_nan=True)).sum()))
        for feature in FEATURES:
            rows.append(dict(cohort=label,feature=feature,n=len(b),
                missing_before=int(a[feature].isna().sum()),
                missing_after=int(b[feature].isna().sum())))
    for feature in FEATURES:
        rows.append(dict(cohort='FAHHAUST_external',feature=feature,n=len(external),
            missing_before=int(external[feature].isna().sum()),
            missing_after=int(external[feature].isna().sum())))
    pd.DataFrame(rows).to_csv(RESULTS/'missingness_before_after.csv',index=False)
    manifest = dict(rule='nearest complete same-record height/weight within index +/-3 calendar months; distance first, same event then prior month as ties',
        sources={str(p.relative_to(ROOT)):sha(p) for p in
            [source,link_source,vital_source,split_source,Path(__file__),ROOT/'body_window.py']},
        input='data/derived/cog_pm3/ppmi_cog.csv', input_sha256=sha(OUT/'ppmi_cog.csv'),
        external='data/derived/cog_v1/fahhaust.csv',
        external_sha256=sha(ROOT/'data/derived/cog_v1/fahhaust.csv'),
        frozen_split='data/derived/cog_pm3/frozen_split.npz',
        frozen_split_sha256=sha(OUT/'frozen_split.npz'),
        unchanged_nonbody_fields=True, exact_original_split=True,
        cohort_summary=summary, vital_gap_counts=body.vital_gap.value_counts().sort_index().astype(int).to_dict(), **qc)
    write_json(RESULTS/'input_manifest.json',manifest)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
