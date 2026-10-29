import unittest
import pandas as pd
from ideal_holdout_weekly_secondary import weekly_bootstrap

class WeeklyResampleTests(unittest.TestCase):
    def test_paired_identical_offset_and_reproducible_draws(self):
        data=pd.DataFrame([{'home':home,'week':week,'n':2,'local_sum':4.,'selector_sum':6.}
                           for home in (1,2) for week in ('2017-10-02','2017-10-09')])
        a=weekly_bootstrap(data);b=weekly_bootstrap(data)
        self.assertEqual(a,b)
        self.assertEqual(a['observed_mean_delta_c'],1.)
        self.assertEqual(a['q025_c'],1.)
        self.assertEqual(a['q975_c'],1.)
        self.assertEqual(a['min_homes_per_draw'],2)
        self.assertEqual(a['draws_with_missing_homes'],0)

    def test_missing_home_is_counted(self):
        data=pd.DataFrame([{'home':1,'week':'2017-10-02','n':2,'local_sum':4.,'selector_sum':6.},
                           {'home':2,'week':'2017-10-09','n':2,'local_sum':4.,'selector_sum':6.}])
        result=weekly_bootstrap(data)
        self.assertLess(result['min_homes_per_draw'],2)
        self.assertGreater(result['draws_with_missing_homes'],0)
        self.assertEqual(result['observed_mean_delta_c'],1.)

if __name__=='__main__':unittest.main()
