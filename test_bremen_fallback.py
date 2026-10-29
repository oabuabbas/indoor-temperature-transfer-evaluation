import unittest
import numpy as np
import pandas as pd
from bremen_fallback_v1 import route
from audit_bremen import aggregate

class BremenPolicyTests(unittest.TestCase):
    def test_routing_depends_only_on_available_predictions(self):
        sel=np.array([21.,np.nan,np.nan,np.nan]);now=np.array([20.,20.,np.nan,np.nan]);daily=np.array([19.,19.,19.,np.nan])
        pred,branch=route(sel,now,daily)
        np.testing.assert_allclose(pred[:3],[21.,20.,19.])
        self.assertTrue(np.isnan(pred[3]))
        self.assertEqual(branch.tolist(),['selector','persistence','daily','abstain'])

    def test_duplicate_minutes_do_not_inflate_coverage(self):
        timestamps=pd.date_range('2023-03-01',periods=60,freq='min',tz='UTC')
        df=pd.DataFrame({'Time':timestamps.strftime('%Y-%m-%dT%H:%M:%SZ'),'Temperature':20.})
        df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
        h,q=aggregate(df)
        self.assertEqual(q['duplicate_time_rows'],2)
        self.assertEqual(len(h),1)
        self.assertEqual(h.temp_c.iloc[0],20.)

if __name__=='__main__':unittest.main()
