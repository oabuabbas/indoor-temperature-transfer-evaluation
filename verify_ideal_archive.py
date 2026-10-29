"""Compare downloaded IDEAL archive against the provider's published MD5."""
import hashlib,json
from audit_nest import DATA, OUT

ARCHIVE=DATA/'ideal'/'room_and_appliance_sensors.zip'
EXPECTED='e879cbea7b03fc1d8e1eb9dfe95df504'

def main():
    digest=hashlib.md5()
    with ARCHIVE.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    stat=ARCHIVE.stat()
    result={'archive_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'md5':digest.hexdigest(),
            'provider_md5':EXPECTED,'matches':digest.hexdigest()==EXPECTED,
            'provider_metadata':'https://datashare.ed.ac.uk/server/api/core/bitstreams/b6e8483c-e5a8-4867-ba5a-1bcf0b4e96e2'}
    (OUT/'ideal_archive_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    if not result['matches'] or stat.st_size!=10005114253: raise ValueError('Archive does not match provider')

if __name__=='__main__':main()
