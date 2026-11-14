"""Essential behavior of the new month window and nearest-record rule."""
import unittest
import numpy as np
import pandas as pd
from body_window import nearest_body


class BodyWindowTests(unittest.TestCase):
    def test_distance_precedes_event_and_ties_prefer_prior(self):
        index=pd.DataFrame({'PATNO':[1,2,3],'EVENT_ID':['BL']*3,'index_month':[2026*12+6]*3})
        vital=pd.DataFrame({'PATNO':[1,1,2,2,3], 'EVENT_ID':['BL','V01','BL','BL','BL'],
            'INFODT':['03/2026','07/2026','05/2026','07/2026','10/2026'],
            'HTCM':[170]*5,'WGTKG':[60,70,80,90,100]})
        out,_=nearest_body(index,vital)
        self.assertEqual(out.weight.iloc[:2].tolist(),[70,80])
        self.assertEqual(out.vital_gap.iloc[:2].tolist(),[-1,1])
        self.assertTrue(np.isnan(out.BMI.iloc[2]))
        self.assertAlmostEqual(out.BMI.iloc[0],70/1.7**2)

    def test_conflicting_and_incomplete_records_are_not_combined(self):
        index=pd.DataFrame({'PATNO':[1,2],'EVENT_ID':['BL']*2,'index_month':[2026*12+6]*2})
        vital=pd.DataFrame({'PATNO':[1,1,2,2],'EVENT_ID':['BL']*4,
            'INFODT':['06/2026']*4,'HTCM':[170,170,170,np.nan],
            'WGTKG':[60,61,np.nan,70]})
        out,qc=nearest_body(index,vital)
        self.assertEqual(qc['conflicting_visit_months_excluded'],1)
        self.assertTrue(out.BMI.isna().all())


if __name__ == '__main__':
    unittest.main()
