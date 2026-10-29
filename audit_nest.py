"""Download, validate, and audit one year of open NEST data; run persistence baseline."""
from __future__ import annotations
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
ITEMS = {
    'UMAR': (24551992, 43480683, 'umar_2019-2020.csv', ['temp_272', 'temp_273', 'temp_274', 'temp_275', 'temp_276']),
    'SolAce': (24551971, 43480713, 'solace_2019-2020.csv', ['temp_meeting', 'temp_office']),
    'DFAB': (24551989, 43480701, 'dfab_2019-2020.csv', ['temp_371', 'temp_472', 'temp_474', 'temp_476', 'temp_571', 'temp_573', 'temp_574']),
}
COMMON = ['temp_amb', 'irrad']
TEST_START = pd.Timestamp('2020-04-01')
TEST_END = pd.Timestamp('2020-07-01')

def download(file_id: int, destination: Path) -> None:
    if destination.exists():
        return
    tmp = destination.with_suffix('.part')
    try:
        with urllib.request.urlopen(f'https://ndownloader.figshare.com/files/{file_id}', timeout=120) as src, tmp.open('wb') as dst:
            while block := src.read(1024 * 1024):
                dst.write(block)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def digest(path: Path, method: str) -> str:
    h = hashlib.new(method)
    with path.open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    manifest, quality, results, monthly = [], [], [], []
    for unit, (article_id, file_id, name, rooms) in ITEMS.items():
        path = DATA / name
        print(f'Downloading/checking {unit}...', flush=True)
        download(file_id, path)
        meta = json.load(urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{article_id}', timeout=30))
        source_file = next(f for f in meta['files'] if f['id'] == file_id)
        actual_md5 = digest(path, 'md5')
        if actual_md5 != source_file['computed_md5'] or path.stat().st_size != source_file['size']:
            raise ValueError(f'{unit}: file hash/size differs from Figshare metadata')
        manifest.append({'unit': unit, 'figshare_file_id': file_id, 'url': f'https://ndownloader.figshare.com/files/{file_id}', 'bytes': path.stat().st_size, 'sha256': digest(path, 'sha256'), 'md5': actual_md5, 'figshare_md5': source_file['computed_md5']})
        headers = pd.read_csv(path, nrows=0).columns.tolist()
        cols = ['time'] + COMMON + rooms
        missing = sorted(set(cols) - set(headers))
        if missing:
            raise ValueError(f'{unit} missing required columns: {missing}')
        d = pd.read_csv(path, usecols=cols, low_memory=False)
        for col in COMMON + rooms:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        t = pd.to_datetime(d['time'], errors='coerce')
        duplicates = t.duplicated(keep=False) & t.notna()
        unique_t = pd.DatetimeIndex(t.dropna().unique()).sort_values()
        expected_t = pd.date_range(t.min(), t.max(), freq='min')
        summary = {'unit': unit, 'rows': len(d), 'bad_timestamp_rows': int(t.isna().sum()), 'duplicate_timestamp_rows': int(duplicates.sum()), 'missing_grid_minutes': int(len(expected_t.difference(unique_t))), 'off_grid_timestamps': int((t.dropna().dt.floor('min') != t.dropna()).sum()), 'first_local': str(t.min()), 'last_local': str(t.max())}
        for col in COMMON + rooms:
            summary['missing_pct_' + col] = round(float(d[col].isna().mean() * 100), 4)
        quality.append(summary)
        d['time'] = t
        d = d.loc[t.notna() & ~duplicates].set_index('time').sort_index()
        for month, group in d.groupby(d.index.to_period('M')):
            for col in COMMON + rooms:
                monthly.append({'unit': unit, 'month': str(month), 'variable': col, 'observed_rows': len(group), 'missing_pct': round(float(group[col].isna().mean() * 100), 4)})
        hourly_count = d[rooms].resample('1h').count()
        hourly = d[rooms].resample('1h').mean().where(hourly_count >= 45)
        for room in rooms:
            s = hourly[room]
            pred = s.shift(1)
            mask = (s.index >= TEST_START) & (s.index < TEST_END) & s.notna() & pred.notna()
            err = (s[mask] - pred[mask]).to_numpy()
            results.append({'unit': unit, 'room': room, 'test_start_local': str(TEST_START), 'test_end_exclusive_local': str(TEST_END), 'valid_pairs': int(err.size), 'mae_c': round(float(np.mean(np.abs(err))), 4) if err.size else None, 'rmse_c': round(float(np.sqrt(np.mean(err ** 2))), 4) if err.size else None})
        print(f'{unit}: rows={len(t)}, duplicate time rows={duplicates.sum()}', flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(quality).to_csv(OUT / 'quality.csv', index=False)
    pd.DataFrame(monthly).to_csv(OUT / 'monthly_missingness.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / 'persistence_baseline.csv', index=False)
    print('Done:', OUT, flush=True)

if __name__ == '__main__':
    main()
