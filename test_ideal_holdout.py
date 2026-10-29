import unittest
from ideal_development_prepare import select
from ideal_development_v1 import windows,H
import pandas as pd

class HoldoutTests(unittest.TestCase):
    def test_house_split_has_no_cross_group_leakage(self):
        dev=select('development');hold=select('holdout')
        self.assertEqual(len(dev),177)
        self.assertEqual(len(hold),73)
        self.assertFalse(set(dev.home)&set(hold.home))
        self.assertTrue(dev.home.is_unique and hold.home.is_unique)

    def test_history_target_hour_finishes_before_test(self):
        index=pd.date_range('2017-09-01',periods=2200,freq='h',tz='UTC')
        frame=pd.DataFrame({'value':range(len(index))},index=index)
        start=pd.Timestamp('2017-10-01',tz='UTC');end=pd.Timestamp('2017-11-01',tz='UTC')
        history,test=windows(frame,start,end)
        self.assertLess((history.index+pd.Timedelta(hours=H)).max(),start)
        self.assertGreaterEqual(test.index.min(),start)
        self.assertLess((test.index+pd.Timedelta(hours=H)).max(),end)

if __name__=='__main__':unittest.main()
