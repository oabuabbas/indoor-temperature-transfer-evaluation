import unittest
import pandas as pd
from selector_history_sensitivity import subset

class HistoryWindowsTests(unittest.TestCase):
    def test_different_history_lengths_share_test_and_no_future_labels(self):
        f=pd.DataFrame({'value':range(2000)},index=range(2000))
        windows=[subset(f,720,1440,days*24,6) for days in (7,14,30)]
        self.assertTrue(all(test.index.equals(windows[0][1].index) for _,test in windows))
        for valid,test in windows:
            self.assertLess(valid.index.max()+6,720)
            self.assertGreaterEqual(test.index.min(),720)
            self.assertLess(test.index.max()+6,1440)
        self.assertEqual(windows[0][0].index.min(),720-7*24)
        self.assertEqual(windows[2][0].index.min(),0)

if __name__=='__main__':unittest.main()
