import unittest
import numpy as np
import pandas as pd
from aalborg_external_v1 import ROOMS,METHODS,prepare,metrics
class AalborgTest(unittest.TestCase):
 def test_utc_offsets_and_eight_bin_threshold(self):
  times=pd.date_range('2023-10-29 00:00',periods=8,freq='5min',tz='UTC')
  d=pd.DataFrame({'timestamp':[t.tz_convert('Europe/Copenhagen').strftime('%Y-%m-%dT%H:%M:%S%z') for t in times]})
  for room in ROOMS:d[f'{room}:Sensor__room_temperature']=20.
  d.loc[0,'RoomA:Sensor__room_temperature']=np.nan
  series,q=prepare(d);self.assertEqual(q['missing_or_invalid_timestamps'],0)
  self.assertTrue(pd.isna(series['RoomA'].iloc[0]));self.assertEqual(series['RoomB'].iloc[0],20.)
 def test_duplicate_utc_detected_even_if_offsets_differ(self):
  d=pd.DataFrame({'timestamp':['2023-10-29T02:30:00+0200','2023-10-29T01:30:00+0100']})
  for room in ROOMS:d[f'{room}:Sensor__room_temperature']=20.
  with self.assertRaises(ValueError):prepare(d)
 def test_common_support_requires_all_eight(self):
  d=pd.DataFrame({'room':['A','A'],'month':['m','m'],'target':[20.,20.]})
  for model in METHODS:d[model]=[21.,21.]
  d.loc[1,'hgb_ideal_selected']=np.nan
  rooms,months,own,pair=metrics(d);self.assertTrue((rooms.n==1).all());self.assertEqual(pair.n.iloc[0],2)
if __name__=='__main__':unittest.main()
