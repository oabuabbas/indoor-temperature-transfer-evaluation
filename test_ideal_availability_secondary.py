import unittest
import numpy as np
import pandas as pd
from ideal_holdout_availability_secondary import build_counts

class AvailabilityTests(unittest.TestCase):
    def test_features_use_only_past_and_target_is_separate(self):
        idx=pd.date_range('2017-01-01',periods=50,freq='h',tz='UTC')
        s=pd.Series(np.arange(50,dtype=float),index=idx)
        s.loc[idx[36]]=np.nan
        result=build_counts(s,pd.DatetimeIndex([idx[30]]),True)
        self.assertTrue(result['source_ready'][0])
        self.assertTrue(result['selector_ready'][0])
        self.assertFalse(result['selector_evaluable'][0])
        no_history=build_counts(s,pd.DatetimeIndex([idx[30]]),False)
        self.assertTrue(no_history['source_ready'][0])
        self.assertFalse(no_history['selector_ready'][0])

    def test_daily_reference_can_exist_after_record_end(self):
        idx=pd.date_range('2017-01-01',periods=30,freq='h',tz='UTC')
        s=pd.Series(np.ones(30),index=idx)
        future=pd.DatetimeIndex([idx[-1]+pd.Timedelta(hours=1)])
        flags=build_counts(s,future,True)
        self.assertTrue(flags['daily_ready'][0])
        self.assertFalse(flags['persistence_ready'][0])
        self.assertFalse(flags['source_ready'][0])
        self.assertTrue(flags['fallback_ready'][0])

if __name__=='__main__':unittest.main()
