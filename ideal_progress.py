"""Read-only snapshot; incomplete prefix is not representative of IDEAL homes."""
import json
from pathlib import Path
from audit_nest import DATA,OUT

def main():
    root=DATA/'ideal';final=root/'room_and_appliance_sensors.zip';partial=root/'room_and_appliance_sensors.zip.part'
    path=final if final.exists() else partial
    records=[]
    for f in (OUT/'ideal_sensor_audit_cache').glob('*.json'):
        records.append(json.loads(f.read_text(encoding='utf-8')))
    check=OUT/'ideal_archive_verification.json'
    verified=check.exists() and final.exists() and json.loads(check.read_text(encoding='utf-8')).get('matches') is True
    print(json.dumps({'expected_archive_bytes':10005114253,
        'assembled_bytes':path.stat().st_size if path.exists() else 0,'download_name':path.name,
        'verified_archive':verified,
        'audited_sensor_files':len(records),'total_sensor_files':1958,
        'audit_errors':sum(x['status']!='ok' for x in records),
        'scope':'Incomplete prefix is not a representative sample; no prediction results.'},indent=2))

if __name__=='__main__':main()
