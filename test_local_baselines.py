import unittest
import numpy as np
import pandas as pd
from local_baselines import bootstrap,choose_models
from daily_residual import FEATURES

class BaselineChecks(unittest.TestCase):
    def test_joint_bootstrap_constant_paired_difference(self):
        rows=[]
        for sensor in ['a','b']:
            for week in ['2021-03-01','2021-03-08','2021-03-15']:
                for model,val in [('daily_seasonal',2.),('persistence',3.),('source_residual',3.),('local_ridge',3.),('local_histgb',3.)]:
                    rows.append({'sensor':sensor,'week':week,'model':model,'abs_sum':val,'n':10})
        d=bootstrap(pd.DataFrame(rows))
        np.testing.assert_allclose(d.observed_diff_c,.1,atol=1e-12)
        np.testing.assert_allclose(d.q025_c,.1,atol=1e-12)
        np.testing.assert_allclose(d.q975_c,.1,atol=1e-12)
        self.assertTrue((d.min_sensors_per_draw==2).all())

    def test_models_fit_constant_residual_and_use_development(self):
        x=np.random.default_rng(42).normal(size=(120,len(FEATURES)))
        d=pd.DataFrame(x,columns=FEATURES);d['target']=d.seasonal_yesterday+1.
        lin,tree,alpha,leaves=choose_models(d.iloc[:40],d.iloc[40:80],d.iloc[:80])
        np.testing.assert_allclose(tree.predict(d.iloc[80:][FEATURES].to_numpy()),1.,atol=1e-8)
        self.assertIn(alpha,[1.,100.,1000.]);self.assertIn(leaves,[7,15])

if __name__=='__main__':unittest.main()
