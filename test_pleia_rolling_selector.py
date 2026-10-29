import unittest
import numpy as np
import pandas as pd
from pleia_rolling_selector import H, make_frame, partition

class PleiaSelectorTests(unittest.TestCase):
    def test_boundaries_include_only_known_past_and_future_in_block(self):
        index=pd.date_range('2021-01-01',periods=1500,freq='h',tz='UTC')
        frame=pd.DataFrame({'value':np.arange(len(index))},index=index)
        start=index[720]; end=index[1440]
        valid,test=partition(frame,start,end)
        self.assertEqual(valid.index.max()+pd.Timedelta(hours=H),start-pd.Timedelta(hours=1))
        self.assertEqual(test.index.min(),start)
        self.assertEqual(test.index.max()+pd.Timedelta(hours=H),end-pd.Timedelta(hours=1))

    def test_no_gap_compression(self):
        t=pd.date_range('2021-01-01',periods=50,freq='h',tz='UTC')
        g=pd.DataFrame({'hour':t.delete(20),'temp':np.delete(np.arange(50.),20)})
        frame=make_frame(g)
        self.assertNotIn(t[20],frame.index)
        self.assertNotIn(t[14],frame.index)
        self.assertEqual(frame.loc[t[30],'seasonal_yesterday'],12.)
        self.assertEqual(frame.loc[t[30],'target'],36.)

if __name__=='__main__':unittest.main()
