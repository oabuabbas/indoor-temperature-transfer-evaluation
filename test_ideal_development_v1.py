import unittest
import numpy as np
import pandas as pd
from ideal_development_v1 import frame_from_series,windows,H
from ideal_development_prepare import split_source

class DevelopmentTests(unittest.TestCase):
    def test_rolling_history_and_forecast_boundary(self):
        t=pd.date_range('2017-08-01',periods=2500,freq='h',tz='UTC')
        s=pd.Series(np.arange(len(t),dtype=float),index=t)
        frame=frame_from_series(s)
        start=pd.Timestamp('2017-10-01',tz='UTC');end=pd.Timestamp('2017-11-01',tz='UTC')
        history,test=windows(frame,start,end)
        self.assertTrue(((history.index+pd.Timedelta(hours=H))<start).all())
        self.assertTrue((test.index>=start).all())
        self.assertTrue(((test.index+pd.Timedelta(hours=H))<end).all())
        issue=start+pd.Timedelta(hours=2)
        self.assertEqual(frame.loc[issue,'target'],s.loc[issue+pd.Timedelta(hours=6)])
        self.assertEqual(frame.loc[issue,'seasonal_yesterday'],s.loc[issue-pd.Timedelta(hours=18)])

    def test_source_split_is_stable(self):
        self.assertEqual(split_source(5),split_source('5'))
        self.assertEqual({split_source(i) for i in range(1,100)},{'source','validation'})

if __name__=='__main__':unittest.main()
