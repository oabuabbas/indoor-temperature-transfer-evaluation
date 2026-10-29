import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import bolzano_external_v1 as b

class BolzanoTests(unittest.TestCase):
    def test_duplicate_conflicts_and_hourly_threshold(self):
        times=pd.date_range('2024-01-01',periods=8,freq='5min')
        d=pd.DataFrame({'timestamp':times.astype(str),'Temperature':[20.]*8})
        d=pd.concat([d,pd.DataFrame({'timestamp':[str(times[0])],'Temperature':[22.]})],ignore_index=True)
        mean,meta=b.aggregate(d,'nominal_mean');drop,meta2=b.aggregate(d,'nominal_drop_conflict')
        self.assertAlmostEqual(mean.iloc[0],20.125)
        self.assertTrue(pd.isna(drop.iloc[0]));self.assertEqual(meta2['conflict_removed_rows'],2)
    def test_rome_ambiguous_and_missing_are_removed(self):
        d=pd.DataFrame({'timestamp':['2024-03-31 02:20:00','2024-10-27 02:20:00','2024-01-01 03:00:00'],'Temperature':[20,21,22]})
        _,meta=b.aggregate(d,'hypothetical_rome')
        self.assertEqual(meta['clock_removed_rows'],2)
    def test_future_target_not_used_for_current_forecast(self):
        idx=pd.date_range('2023-12-01','2026-01-01',freq='h',inclusive='left');s=pd.Series(20.+np.sin(np.arange(len(idx))/20),index=idx)
        changed=s.copy();t=pd.Timestamp('2024-01-01');changed.loc[t+pd.Timedelta(hours=6)]+=8
        def fakepred(frame,model):
            return pd.DataFrame({'persistence':frame.temp_now,'daily_seasonal':frame.seasonal_yesterday,'source_daily_ridge':frame.temp_now},index=frame.index)
        with patch.object(b,'predictions',side_effect=fakepred),patch.object(b,'fit',return_value=None),patch.object(b,'predict',side_effect=lambda model,x:np.zeros(len(x))):
            first,_=b.evaluate(s,None);second,_=b.evaluate(changed,None)
        a=first.loc[first.issue_hour_start.eq(t)].iloc[0];c=second.loc[second.issue_hour_start.eq(t)].iloc[0]
        self.assertAlmostEqual(a.fallback,c.fallback);self.assertNotEqual(a.target,c.target)
if __name__=='__main__':unittest.main()
