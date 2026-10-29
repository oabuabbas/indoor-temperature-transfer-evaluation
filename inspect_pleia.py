"""Download and inspect official PLEIAData archive without extracting bulk files."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
from audit_nest import DATA, OUT, digest

URL='https://zenodo.org/api/records/7620136/files/Data_Nature.zip/content'
def main():
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    file=DATA/'PLEIAData_zenodo_7620136.zip'; part=file.with_suffix('.part')
    if not file.exists():
        print('Downloading PLEIAData archive...',flush=True)
        try:
            with urllib.request.urlopen(URL,timeout=120) as s,part.open('wb') as f:
                while block:=s.read(1024*1024): f.write(block)
            part.replace(file)
        except Exception:
            part.unlink(missing_ok=True); raise
    meta=json.load(urllib.request.urlopen('https://zenodo.org/api/records/7620136',timeout=40))
    info=next(x for x in meta['files'] if x['key']=='Data_Nature.zip')
    md5=digest(file,'md5')
    if md5!=info['checksum'].split(':',1)[1] or file.stat().st_size!=info['size']:
        raise ValueError('PLEIAData checksum or file size differs from Zenodo')
    report={'doi':'10.5281/zenodo.7620136','license':meta['metadata']['license']['id'],'size':file.stat().st_size,'md5':md5,'sha256':digest(file,'sha256'),'members':[]}
    with zipfile.ZipFile(file) as z:
        for f in z.infolist():
            record={'name':f.filename,'bytes_uncompressed':f.file_size,'bytes_compressed':f.compress_size}
            if not f.is_dir() and f.filename.lower().endswith(('.csv','.txt')):
                with z.open(f) as stream:
                    record['first_lines']=[stream.readline(8192).decode('utf-8',errors='replace').strip() for _ in range(3)]
            report['members'].append(record)
    (OUT/'pleia_archive_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Entries',len(report['members']),'archive verified',flush=True)
    for x in report['members']:
        print(x['name'],x['bytes_uncompressed'],flush=True)

if __name__=='__main__': main()
