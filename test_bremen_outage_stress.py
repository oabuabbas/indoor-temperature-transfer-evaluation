import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import bremen_outage_stress_v1 as stress

class OutageTest(unittest.TestCase):
    def test_masks_reproducible_and_block_length(self):
        a=stress.make_mask(1440,'block72h_30d',(72,720),77)
        self.assertTrue(np.array_equal(a,stress.make_mask(1440,'block72h_30d',(72,720),77)))
        self.assertEqual(a.sum(),144)
        self.assertFalse(stress.make_mask(200,'none',0,1).any())
    def test_oracle_target_hidden_from_routing(self):
        idx=pd.date_range(stress.MASK_START,stress.END,freq='h',inclusive='left')
        s=pd.Series(20+np.sin(np.arange(len(idx))/24),index=idx)
        t=stress.START
        mask=np.zeros(len(s),dtype=bool)
        mask[idx.get_loc(t+pd.Timedelta(hours=6))]=True
        def dummy_predictions(frame,model):
            return pd.DataFrame({'persistence':frame.temp_now,'daily_seasonal':frame.seasonal_yesterday,'source_daily_ridge':frame.temp_now},index=frame.index)
        with patch.object(stress,'make_mask',return_value=mask), patch.object(stress,'predictions',side_effect=dummy_predictions), patch.object(stress,'fit',return_value=None), patch.object(stress,'predict',side_effect=lambda model,x:np.zeros(len(x))):
            _,_,_,points,_=stress.evaluate_room(s,None,'iid05',1,.05,0,0)
        row=points[points.idx==t].iloc[0]
        self.assertAlmostEqual(row.target,s.loc[t+pd.Timedelta(hours=6)])
        self.assertTrue(np.isfinite(row.fallback))
        # The unavailable future hour cannot become a current input later.
        later=points[points.idx==t+pd.Timedelta(hours=6)].iloc[0]
        self.assertTrue(np.isnan(later.persistence))

    def test_abstention_is_not_scored(self):
        idx=pd.date_range(stress.MASK_START,stress.END,freq='h',inclusive='left')
        s=pd.Series(21.,index=idx)
        with patch.object(stress,'make_mask',return_value=np.ones(len(s),dtype=bool)):
            _,branches,_,points,_=stress.evaluate_room(s,None,'iid20',2,.2,0,0)
        self.assertTrue(points.fallback.isna().all())
        self.assertTrue(all(row['scored']==0 for row in branches))
        self.assertGreater(sum(row['opportunities'] for row in branches if row['branch']=='abstain'),0)

if __name__=='__main__':unittest.main()
