"""Small checks for consequential endpoint/preprocessing behavior."""
import unittest
import numpy as np
import pandas as pd
from pilot import CappedImputer, LABS, clean_sheet, moca_target
from prepare_current_ppmi import choose_pairs, collapse, screen_sex
from prepare_cog_ppmi import cog_target, complete_episodes


class ScientificBehavior(unittest.TestCase):
    def test_endpoint_and_unknown(self):
        got = moca_target([25, 26, 30, np.nan])
        self.assertEqual(got[:3].tolist(), [1, 0, 0])
        self.assertTrue(np.isnan(got[3]))

    def test_caps_and_imputation_use_training(self):
        x = np.array([[20, 0, 20], [40, 1, 30], [60, 1, np.nan]], float)
        prep = CappedImputer().fit(x)
        fill = prep.fill_.copy()
        a = prep.transform([[10000, np.nan, np.nan]])
        np.testing.assert_array_equal(fill, prep.fill_)
        self.assertTrue(np.isfinite(a).all())
        self.assertLess(a[0, 0], 3)

    def test_cbc_error_propagates_to_ratio_without_deleting_patient(self):
        row = {x: 1.0 for x in LABS}
        row.update(PATNO=1, age=60, sex=1, height=170, weight=70, MCATOT=26, COG=0,
                   WBC=5, Neutrophils=10, Lymphocytes=1)
        clean, qc = clean_sheet(pd.DataFrame([row]))
        self.assertEqual(len(clean), 1)
        self.assertEqual(qc['invalid_cbc_people'], 1)
        self.assertTrue(pd.isna(clean.loc[0, 'NLR']))
        self.assertTrue(pd.isna(clean.loc[0, 'Neutrophils']))
        self.assertAlmostEqual(clean.loc[0, 'BMI'], 70/1.7**2)

    def test_pairing_excludes_future_and_selects_earliest_eligible_visit(self):
        cog = pd.DataFrame({'PATNO':[1,1], 'EVENT_ID':['BL','V04'],
                            'cog_month':[100,112], 'MCATOT':[25,28]})
        blood = pd.DataFrame({'PATNO':[1,1,1], 'EVENT_ID':['SC','BL','V04'],
                              'blood_month':[99,101,112], 'valid_lab_count':[15,15,15]})
        pairs = choose_pairs(cog,blood)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs.iloc[0].cog_month,100)
        self.assertEqual(pairs.iloc[0].blood_month,99)

    def test_conflicting_values_are_unknown(self):
        got = collapse(pd.DataFrame({'id':[1,1,2,2], 'x':[1,2,3,3]}), ['id'], 'x')
        self.assertTrue(np.isnan(got.loc[got.id.eq(1),'x'].iloc[0]))
        self.assertEqual(got.loc[got.id.eq(2),'x'].iloc[0],3)

    def test_screen_sex_preserves_female_zero(self):
        got = screen_sex(pd.Series([0, 1, 2, np.nan]))
        self.assertEqual(got.iloc[:2].tolist(), [0, 1])
        self.assertTrue(got.iloc[2:].isna().all())

    def test_published_cog_or_and_complete_ascertainment(self):
        got = cog_target([25,26,29,30,25,np.nan], [0,0,1,0,np.nan,1])
        self.assertEqual(got[:4].tolist(), [1,0,1,0])
        self.assertTrue(np.isnan(got[4:]).all())

    def test_composite_index_is_latest_component_month(self):
        mo = pd.DataFrame({'PATNO':[1,1], 'EVENT_ID':['SC','V04'], 'moca_month':[100,112], 'MCATOT':[26,24]})
        nu = pd.DataFrame({'PATNO':[1,1], 'EVENT_ID':['BL','V04'], 'item_month':[102,115], 'NP1COG':[1,0]})
        got = complete_episodes(mo,nu)
        self.assertEqual(len(got),1)
        self.assertEqual(got.iloc[0].index_month,102)
        self.assertEqual(got.iloc[0].EVENT_ID,'BL')


if __name__ == '__main__':
    unittest.main()
