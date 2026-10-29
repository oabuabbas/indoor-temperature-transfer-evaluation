import unittest
from create_ideal_house_split import assignment

class SplitTests(unittest.TestCase):
    def test_stable_and_two_groups_present(self):
        ids=range(1,500)
        first=[assignment(i) for i in ids]
        self.assertEqual(first,[assignment(i) for i in ids])
        self.assertEqual(set(first),{'development','holdout'})
        self.assertEqual(assignment(42),assignment('42'))

if __name__=='__main__':unittest.main()
