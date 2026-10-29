import unittest
import numpy as np
import pandas as pd
from ideal_nonlinear_tuning_v1 import GRID,model_for
from ideal_development_v1 import frame_from_series,windows,H
class TestTuningPlan(unittest.TestCase):
 def test_fixed_candidates_and_no_early_stopping(self):
  self.assertEqual(list(GRID),['shallow','moderate','richer'])
  for k in GRID:
   model=model_for(k);self.assertFalse(model.early_stopping);self.assertEqual(model.random_state,20261029);self.assertEqual(model.max_iter,GRID[k]['max_iter'])
 def test_history_targets_precede_test_start(self):
  idx=pd.date_range('2018-01-01','2018-03-01',freq='h',tz='UTC',inclusive='left');s=pd.Series(np.sin(np.arange(len(idx)))+20,index=idx)
  start=pd.Timestamp('2018-02-01',tz='UTC');end=pd.Timestamp('2018-03-01',tz='UTC')
  hist,test=windows(frame_from_series(s),start,end)
  self.assertTrue((hist.index+pd.Timedelta(hours=H)<start).all());self.assertTrue((test.index>=start).all());self.assertTrue((test.index+pd.Timedelta(hours=H)<end).all())
if __name__=='__main__':unittest.main()
