import unittest
import pandas as pd
from thermbuild_gating import windows

class GatingWindowsTests(unittest.TestCase):
    def test_labels_are_available_before_selection_and_evaluation_stays_in_block(self):
        frame=pd.DataFrame({'target':range(1500)},index=range(1500))
        history, test=windows(frame,720,1440)
        self.assertEqual(history.index.max(),713)
        self.assertEqual(test.index.min(),720)
        self.assertEqual(test.index.max(),1433)
        self.assertTrue(((history.index+6)<720).all())
        self.assertTrue(((test.index+6)<1440).all())

if __name__ == '__main__': unittest.main()
