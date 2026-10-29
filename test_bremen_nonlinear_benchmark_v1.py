import unittest
import numpy as np
import pandas as pd
from bremen_nonlinear_benchmark_v1 import compute,verify_old_results
from bolzano_2026_benchmark_v1 import METHODS
class BremenBenchmarkTests(unittest.TestCase):
 def test_only_intersection_contributes_and_groupwise_rmse(self):
  d=pd.DataFrame({'sensor':['a','a','b','b'],'month':['m']*4,'target':[0.,0.,0.,0.]})
  for model in METHODS:d[model]=[1.,2.,3.,4.]
  d.loc[1,'local_hist_gradient_boosting']=np.nan
  rooms,months=compute(d);r=rooms[rooms.model.eq('rolling_selector')].set_index('sensor')
  self.assertEqual(r.loc['a','n'],1);self.assertEqual(r.loc['b','n'],2)
  self.assertAlmostEqual(r.loc['b','rmse_c'],np.sqrt((9+16)/2))
  self.assertEqual(len(months),14)
 def test_old_result_disagreement_stops_new_benchmark(self):
  d=pd.DataFrame({'month':['m'],'target':[1.]})
  for model in METHODS[:-1]:d[model]=[1.]
  old=pd.DataFrame([{'sensor':'a','month':'m','model':model,'opportunities':1,'ready':1,'scored':1,'abs_error_sum':0.} for model in METHODS[:-1]])
  choices=pd.DataFrame([{'month':'m','history_n':24,'choice':'persistence'}]);saved=choices.assign(sensor='a')
  verify_old_results('a',d,choices,old,saved)
  old.loc[0,'abs_error_sum']=1.
  with self.assertRaises(ValueError):verify_old_results('a',d,choices,old,saved)
if __name__=='__main__':unittest.main()
