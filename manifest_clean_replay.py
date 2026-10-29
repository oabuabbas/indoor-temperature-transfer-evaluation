"""Confirm copied code/protocol identity and record shared raw archives."""
import hashlib,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT/'replay_clean_20261029'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    copied=[]
    for p in sorted(REPLAY.iterdir()):
        if p.is_file() and (p.suffix=='.py' or p.name.startswith('protocol') and p.suffix=='.md'):
            original=ROOT/p.name
            if not original.exists() or sha(original)!=sha(p):raise ValueError('Copied source mismatch: '+p.name)
            copied.append({'name':p.name,'sha256':sha(p)})
    links=[]
    for name in ['documentation.zip','metadata_and_surveys.zip','room_and_appliance_sensors.zip']:
        a=ROOT/'data'/'ideal'/name;b=REPLAY/'data'/'ideal'/name
        if not os.path.samefile(a,b):raise ValueError('Not the same hardlinked source: '+name)
        links.append({'name':name,'size':a.stat().st_size,'same_file':True})
    result={'copied_sources':copied,'raw_archives':links,
            'note':'Shared raw files must only be read, never written; full checksum is independently recalculated in replay.'}
    (REPLAY/'outputs'/'replay_input_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verified',len(copied),'copied source/protocol files and',len(links),'shared input archives')

if __name__=='__main__':main()
