import unittest
from compare_clean_replay import compare_json

class ReplayComparisonTests(unittest.TestCase):
    def test_numeric_tolerance_does_not_hide_material_or_structural_changes(self):
        compare_json({'count':2,'mean':[0.5,1.0]},{'count':2,'mean':[0.500000001,1.0]})
        with self.assertRaises(AssertionError):compare_json({'count':2},{'count':3})
        with self.assertRaises(AssertionError):compare_json({'mean':0.5},{'mean':0.5001})
        with self.assertRaises(AssertionError):compare_json({'x':1},{'y':1})

if __name__=='__main__':unittest.main()
