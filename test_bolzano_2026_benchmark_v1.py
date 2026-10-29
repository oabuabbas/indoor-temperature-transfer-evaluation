import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import bolzano_2026_benchmark_v1 as b
class TestBolzanoBenchmark(unittest.TestCase):
 def test_score_uses_intersection_of_all_models(self):
  d=pd.DataFrame({'target':[1.,2.,3.], 'month':['a']*3,'branch':['selector']*3})
  for model in b.METHODS:d[model]=[1.,2.,3.]
  d.loc[1,'local_hist_gradient_boosting']=np.nan;d.loc[2,'daily_seasonal']=np.nan
  z,_=b.score(d);self.assertEqual(z['common_n'],1);self.assertTrue(all(x['n']==1 for x in z['common_metrics']))
 def test_boosting_history_excludes_future_target(self):
  idx=pd.date_range('2025-12-01','2026-02-02',freq='h',inclusive='left');s=pd.Series(20+np.sin(np.arange(len(idx))/10),index=idx)
  start=pd.Timestamp('2026-01-01');hist=b.frame_from_series(s);hist=hist[(hist.index>=start-pd.Timedelta(days=30))&(hist.index+pd.Timedelta(hours=b.H)<start)]
  d=pd.DataFrame({'month':['2026-01-01T00:00:00'],'temp_now':[21.],'temp_lag1':[21.],'seasonal_yesterday':[20.],'target':[30.]})
  choices=pd.DataFrame([{'month':'2026-01-01T00:00:00','history_n':len(hist)}]);seen=[]
  class Spy:
   def fit(self,x,y):seen.append(y.copy());return self
   def predict(self,x):return np.zeros(len(x))
  with patch.object(b,'make_boosting',return_value=Spy()):
   a=b.add_boosting(d,s,choices);s2=s.copy();s2.loc[start+pd.Timedelta(hours=6)]=1000
   c=b.add_boosting(d,s2,choices)
  self.assertTrue(np.array_equal(seen[0],seen[1]));self.assertTrue(np.array_equal(a.local_hist_gradient_boosting,c.local_hist_gradient_boosting))
if __name__=='__main__':unittest.main()
