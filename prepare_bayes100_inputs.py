"""Versioned input reconstruction; preserve the original participant split."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
from common_preprocessing import LABS, clean_sheet, sha, write_json
from body_window import nearest_body
from body_rules import BODY_RULE_VERSION, HEIGHT_MIN_CM, HEIGHT_MAX_CM

ROOT = Path(__file__).resolve().parent
FEATURES = ['age', 'sex', 'height', 'weight', 'BMI'] + LABS


def main(check_only=False, rebuild=False):
    if check_only and rebuild:
        raise ValueError('Choose --check or --rebuild-body, not both')
    current = ROOT / 'data/derived/cog_pm3'
    source = current / 'ppmi_cog.csv'
    split_path = current / 'frozen_split.npz'
    output = ROOT / 'data/derived/bayes100_input'
    with np.load(split_path) as z:
        split = {k: z[k].copy() for k in z.files}
    split_hash = sha(split_path)
    work = ROOT / 'tmp/input_rebuild'
    if rebuild:
        work.mkdir(parents=True, exist_ok=True)
        original = pd.read_csv(ROOT / 'data/derived/cog_v1/ppmi_cog.csv')
        link = pd.read_csv(ROOT / 'data/derived/cog_v1/linkage_private.csv')
        from prepare_current_ppmi import RAW as PPMI_RAW
        vital = pd.read_csv(next(PPMI_RAW.glob('Vital_Signs_*.csv')), low_memory=False)
        body, body_info = nearest_body(link, vital)
        keep = [c for c in original if c not in ['height', 'weight', 'BMI']]
        updated = original[keep].merge(body[['PATNO', 'height', 'weight', 'BMI']],
                                      on='PATNO', how='left', validate='one_to_one', sort=False)
        pd.testing.assert_frame_equal(original[keep], updated[keep], check_exact=True)
        updated.to_csv(work / 'ppmi_cog.csv', index=False)
        body.to_csv(work / 'body_linkage_private.csv', index=False)
        data_source = work / 'ppmi_cog.csv'
    else:
        data_source = source
    ppmi, info = clean_sheet(pd.read_csv(data_source))
    ppmi = ppmi.loc[ppmi.COHORT_DEFINITION.eq("Parkinson's Disease")].reset_index(drop=True)
    assert (len(ppmi), int(ppmi.COG.sum()), ppmi.PATNO.nunique()) == (1385, 679, 1385)
    ids = ppmi.set_index('PATNO', drop=False)
    assert set(split['train_id']).isdisjoint(split['internal_id'])
    assert set(split['train_id']) | set(split['internal_id']) == set(ids.index)
    external_source = ROOT / 'data/derived/cog_v1/fahhaust.csv'
    external, _ = clean_sheet(pd.read_csv(external_source))
    assert (len(external), int(external.COG.sum()), external.PATNO.nunique()) == (197, 122, 197)
    frames = {'training': ids.loc[split['train_id']], 'internal': ids.loc[split['internal_id']], 'external': external}
    assert np.array_equal(frames['training'].COG.to_numpy(int), split['y'])
    assert np.array_equal(frames['internal'].COG.to_numpy(int), split['yi'])
    arrays = {}
    for name, frame in frames.items():
        assert frame.height.dropna().between(HEIGHT_MIN_CM, HEIGHT_MAX_CM).all()
        expected_bmi = frame.weight / (frame.height / 100) ** 2
        assert np.allclose(frame.BMI, expected_bmi, equal_nan=True, rtol=1e-14)
        values = dict(X=frame[FEATURES].to_numpy(float), y=frame.COG.to_numpy(int), id=frame.PATNO.to_numpy())
        arrays[name] = values
        if rebuild:
            np.savez_compressed(work / (name + '.npz'), **values)
        else:
            with np.load(output / (name + '.npz')) as existing:
                for key, value in values.items():
                    assert np.array_equal(value, existing[key], equal_nan=True), (name, key)
    report = None
    if rebuild:
        changes = {}
        for name, values in arrays.items():
            with np.load(output / (name + '.npz')) as old:
                assert np.array_equal(old['id'], values['id']) and np.array_equal(old['y'], values['y'])
                eq = (old['X'] == values['X']) | (np.isnan(old['X']) & np.isnan(values['X']))
                # CSV reconstruction can introduce sub-ulp serialization differences.
                eq |= np.isclose(old['X'], values['X'], rtol=1e-13, atol=1e-13, equal_nan=True)
                assert eq[:, [0, 1] + list(range(5, 21))].all(), 'Unexpected non-body change'
                changes[name] = dict(changed_body_cells={FEATURES[k]: int((~eq[:, k]).sum()) for k in [2,3,4]},
                                     missing={FEATURES[k]: int(np.isnan(values['X'][:, k]).sum()) for k in [2,3,4]},
                                     participants_preserved=True, labels_preserved=True)
        report = dict(rule=BODY_RULE_VERSION, body_matching=body_info, cohorts=changes, split_sha256=split_hash)
        # All comparisons pass before replacing any active input. Retain prior arrays for this repair comparison.
        for name in arrays:
            prior = work / (name + '_before.npz')
            if not prior.exists():
                prior.write_bytes((output / (name + '.npz')).read_bytes())
        old_manifest = output / 'input_manifest.json'
        if not (work / 'input_manifest_before.json').exists():
            (work / 'input_manifest_before.json').write_bytes(old_manifest.read_bytes())
        for filename in ['ppmi_cog.csv', 'body_linkage_private.csv']:
            os.replace(work / filename, current / filename)
        for name in arrays:
            os.replace(work / (name + '.npz'), output / (name + '.npz'))
    manifest = dict(features=FEATURES, COG_rule='MoCA<26 OR cognitive-item>=1; both components observed', seed=20261014,
                    cleaning_rule=dict(version=BODY_RULE_VERSION, height_cm=[HEIGHT_MIN_CM, HEIGHT_MAX_CM],
                                       invalid_values='missing; preserve participants and frozen split', window_months=3),
                    counts={k: dict(n=len(v), positive=int(v.COG.sum())) for k,v in frames.items()},
                    source_sha256={q.relative_to(ROOT).as_posix(): sha(q) for q in [source, external_source, split_path]},
                    code_sha256={n:sha(ROOT / n) for n in ['body_rules.py','body_window.py','common_preprocessing.py','prepare_bayes100_inputs.py']},
                    files_sha256={q.name: sha(q) for q in sorted(output.glob('*.npz'))})
    assert sha(split_path) == split_hash, 'Frozen split was modified'
    if rebuild:
        write_json(output / 'input_manifest.json', manifest)
        write_json(work / 'input_change_report.json', report)
        print(json.dumps(report), flush=True)
    else:
        assert manifest == json.loads((output / 'input_manifest.json').read_text()), 'Manifest mismatch'
    print(json.dumps(dict(event='BAYES100_INPUTS_CHECKED', counts=manifest['counts'], split_preserved=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--rebuild-body', action='store_true')
    args = parser.parse_args()
    main(args.check, args.rebuild_body)
