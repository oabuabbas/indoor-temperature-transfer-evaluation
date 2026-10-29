import gzip
import tempfile
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import ideal_development_prepare as prepare_module
from audit_nest import OUT

class RangeAggregationTests(unittest.TestCase):
    def test_wide_and_strict_hourly_means_share_issue_but_can_change_target(self):
        start=pd.Timestamp('2017-01-01',tz='UTC')
        rows=[]
        for slot in range(4):
            for reading in range(50):
                t=start+pd.Timedelta(minutes=10*slot,seconds=12*reading)
                rows.append(t.strftime('%Y-%m-%d %H:%M:%S')+','+('120' if reading<25 else '200')+'\n')
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            root=Path(tmp);archive=root/'sensor.zip';wide=root/'wide';strict=root/'strict'
            wide.mkdir();strict.mkdir()
            with zipfile.ZipFile(archive,'w') as z:z.writestr('sensor.csv.gz',gzip.compress(''.join(rows).encode()))
            old=prepare_module.ARCHIVE
            try:
                prepare_module.ARCHIVE=archive
                prepare_module.prepare((1,2,'sensor.csv.gz'),10,45,wide)
                prepare_module.prepare((1,2,'sensor.csv.gz'),15,40,strict)
            finally:prepare_module.ARCHIVE=old
            a=pd.read_csv(wide/'2.csv.gz');b=pd.read_csv(strict/'2.csv.gz')
            self.assertEqual(a.hour_utc.iloc[0],b.hour_utc.iloc[0])
            self.assertAlmostEqual(a.temp_c.iloc[0],16.)
            self.assertAlmostEqual(b.temp_c.iloc[0],20.)
            self.assertEqual(a.valid_slots.iloc[0],4)

if __name__=='__main__':unittest.main()
