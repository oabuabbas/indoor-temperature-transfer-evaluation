import unittest
import numpy as np
import pandas as pd
from adaptation_stress import fit
from selector_cold_start import evaluate

class ColdStartTests(unittest.TestCase):
    def test_uses_no_features_before_origin_and_same_test_counts(self):
        # Include negative issue hours to test that pre-origin rows are ignored.
        index=np.arange(-50, 61*24)
        df=pd.DataFrame({'temp_now':20+np.sin(index/20),'temp_lag1':20+np.sin((index-1)/20),
                         'seasonal_yesterday':20+np.sin((index-18)/20),
                         'target':20+np.sin((index+6)/20)},index=index)
        model=fit(df.iloc[100:500][['temp_now','temp_lag1','seasonal_yesterday']].to_numpy(),
                  (df.iloc[100:500].target-df.iloc[100:500].seasonal_yesterday).to_numpy())
        output=pd.DataFrame(evaluate('test','house','room',df,0,24,6,model))
        week=output[(output.phase=='days_7_14') & (output.method=='select_7d')].iloc[0]
        # Issues 18..161 inclusive: earliest daily lag at hour 0, and target < hour 168.
        self.assertEqual(week.validation_n,144)
        self.assertEqual(week.test_n,162)
        for _,group in output.groupby('phase'):self.assertEqual(group.test_n.nunique(),1)
        self.assertEqual(set(output[output.phase=='days_7_14'].method),
                         {'select_7d','persistence','daily_seasonal','source_daily_ridge'})

if __name__=='__main__':unittest.main()
