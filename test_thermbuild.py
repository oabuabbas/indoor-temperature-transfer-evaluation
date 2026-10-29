import unittest
import numpy as np
import pandas as pd
from thermbuild_external import ROOMS, hourly, evaluation_frame

class ThermBuildTests(unittest.TestCase):
    def test_aggregate_and_missing(self):
        days = np.repeat(np.arange(2,444),96)
        raw = pd.DataFrame({'TIME':np.arange(len(days)), 'consecutive_days':days})
        for room in ROOMS: raw[room] = 20.0
        raw.loc[:1,ROOMS[0]] = np.nan
        raw.loc[4,ROOMS[0]] = np.nan
        actual = hourly(raw)
        self.assertTrue(np.isnan(actual.loc[0,ROOMS[0]]))
        self.assertEqual(actual.loc[1,ROOMS[0]],20.0)
        self.assertEqual(len(actual), 442*24)
        with self.assertRaises(ValueError): hourly(raw.drop(index=9))

    def test_filter_changes_hourly_means_and_coverage(self):
        days = np.repeat(np.arange(2,444),96)
        raw = pd.DataFrame({'TIME':np.arange(len(days)), 'consecutive_days':days})
        for room in ROOMS: raw[room] = 20.0
        raw.loc[0,ROOMS[0]] = 12.0
        raw.loc[4:5,ROOMS[0]] = 12.0
        strict, wide = hourly(raw), hourly(raw,10,45)
        self.assertEqual(strict.loc[0,ROOMS[0]], 20.0)
        self.assertEqual(wide.loc[0,ROOMS[0]], 18.0)
        self.assertTrue(np.isnan(strict.loc[1,ROOMS[0]]))
        self.assertEqual(wide.loc[1,ROOMS[0]], 16.0)

    def test_forecast_time_and_previous_day(self):
        series = pd.Series(np.arange(100,dtype=float))
        result = evaluation_frame(series,6)
        self.assertEqual(result.loc[30,'target'],36)
        self.assertEqual(result.loc[30,'seasonal_yesterday'],12)
        result = evaluation_frame(series,24)
        self.assertEqual(result.loc[30,'seasonal_yesterday'],30)
        self.assertEqual(result.loc[30,'target'],54)

if __name__ == '__main__': unittest.main()
