import unittest
import numpy as np
import pandas as pd
from bremen_joint_missing_v1 import create_masks
class JointMasks(unittest.TestCase):
    def test_shared_masks_and_reproducibility(self):
        s={f'room{i}':pd.Series(np.repeat(20.,1440)) for i in range(4)}
        for name in ['common_iid10','common_block24_14d']:
            a=create_masks(name,0,s);b=create_masks(name,0,s)
            for k in s:self.assertTrue(np.array_equal(a[k],b[k]) and np.array_equal(a[k],a['room0']))
        self.assertEqual(create_masks('common_block24_14d',0,s)['room0'][:1344].sum(), 24*4)
    def test_state_mask_uses_previous_hour_and_not_future(self):
        n=20000;s={'room0':pd.Series(np.full(n,26.)),'room1':pd.Series(np.full(n,20.))}
        a=create_masks('temperature_dependent',0,s)
        self.assertGreater(a['room0'].mean(),.45);self.assertLess(a['room1'].mean(),.06)
        changed={k:v.copy() for k,v in s.items()};changed['room0'].iloc[1000:]=20.
        b=create_masks('temperature_dependent',0,changed)
        self.assertTrue(np.array_equal(a['room0'][:1001],b['room0'][:1001]))
        self.assertLess(b['room0'][1001:].mean(),.06)
if __name__=='__main__':unittest.main()
