import gzip
import tempfile
import unittest
import zipfile
from pathlib import Path
import pandas as pd
import audit_ideal_temperature as module

class IdealAuditTests(unittest.TestCase):
    def test_chunk_boundary_duplicate_and_complete_60_day_window(self):
        # Exactly 25 readings in every 10-minute slot over 60 days.
        base=pd.Timestamp('2017-01-01',tz='UTC')
        rows=[]
        for slot in range(60*24*6):
            for reading in range(25):
                t=base+pd.Timedelta(minutes=10*slot,seconds=12*reading)
                rows.append(t.strftime('%Y-%m-%d %H:%M:%S')+',210\n')
        rows.insert(200000,rows[199999])
        with tempfile.TemporaryDirectory(dir=module.OUT) as temp:
            archive=Path(temp)/'sample.zip'
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('sample.csv.gz',gzip.compress(''.join(rows).encode()))
            previous=module.ARCHIVE
            try:
                module.ARCHIVE=archive
                result=module.audit((1,2,'sample.csv.gz',0))
            finally:module.ARCHIVE=previous
        self.assertEqual(result['status'],'ok',result['error'])
        self.assertEqual(result['duplicate_adjacent'],1)
        self.assertEqual(result['time_reversals'],0)
        self.assertEqual(result['qualified_hours'],1440)
        self.assertEqual(result['max_qualified_hours_60d'],1440)
        self.assertTrue(result['coverage_eligible'])

    def test_audit_can_read_verified_gzip_payload_without_zip(self):
        data=gzip.compress(b'2017-01-01 00:00:00,210\n2017-01-01 00:00:12,220\n')
        result=module.audit((1,2,'sample.csv.gz',0,data))
        self.assertEqual(result['status'],'ok',result['error'])
        self.assertEqual(result['raw_rows'],2)
        self.assertEqual(result['valid_samples'],2)
        self.assertFalse(result['coverage_eligible'])

    def test_wide_range_counts_change_without_changing_original_rule(self):
        payload=gzip.compress(b'2017-01-01 00:00:00,120\n2017-01-01 00:00:12,210\n2017-01-01 00:00:24,460\n')
        strict=module.audit((1,2,'sample.csv.gz',0,payload))
        wide=module.audit((1,2,'sample.csv.gz',0,payload),10,45)
        self.assertEqual(strict['outside_15_40'],2)
        self.assertEqual(wide['outside_10_45'],1)
        self.assertEqual(wide['valid_samples'],2)

if __name__=='__main__':unittest.main()
