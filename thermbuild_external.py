"""Relative-time, zero-target-fit external test; see protocol_thermbuild.md."""
import json
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS
from adaptation_stress import fit, predict
from daily_residual import nest_source

FEATURES = ['temp_now', 'temp_lag1', 'seasonal_yesterday']
ROOMS = ['room_1_child1_Tair', 'room_2_child2_Tair', 'room_3_sleep_Tair',
         'room_4_bath_Tair', 'room_5_living_Tair', 'room_6_dining_Tair',
         'room_7_kitchen_Tair']

def hourly(raw, lower=15, upper=40):
    """Preserve full ordinal-day grid, and reject unexpected sequence structure."""
    d = raw.copy()
    if not np.array_equal(d.TIME.to_numpy(), np.arange(len(d))):
        raise ValueError('TIME is not consecutive from zero')
    d = d[d.consecutive_days.between(2, 443)].copy()
    counts = d.groupby('consecutive_days', sort=True).size()
    if list(counts.index) != list(range(2, 444)) or not counts.eq(96).all():
        raise ValueError('Expected 442 complete days of 96 records')
    if not d.consecutive_days.is_monotonic_increasing:
        raise ValueError('Days out of order')
    d['hour'] = (d.consecutive_days.astype(int) - 2) * 24 + d.groupby('consecutive_days').cumcount() // 4
    values = d[ROOMS].apply(pd.to_numeric, errors='coerce')
    values = values.where(values.ge(lower) & values.le(upper))
    values['hour'] = d['hour'].to_numpy()
    group = values.groupby('hour')
    return group.mean().where(group.count() >= 3).reindex(range(442 * 24))

def evaluation_frame(series, h):
    return pd.DataFrame({'temp_now': series, 'temp_lag1': series.shift(1),
                         'seasonal_yesterday': series.shift(24-h),
                         'target': series.shift(-h)}).dropna()

def main():
    houses = {}
    with zipfile.ZipFile(DATA/'ThermBuild_measure_raw.zip') as archive:
        for name in archive.namelist():
            if not name.endswith('_wetReal_raw.csv'): continue
            house = 'Use1001_BSE1' if 'Use1001' in name else 'Use1002_BSE2' if 'Use1002' in name else None
            if house is None or house in houses: raise ValueError('Unexpected measured member')
            with archive.open(name) as stream:
                raw = pd.read_csv(stream, usecols=['TIME', 'consecutive_days'] + ROOMS, low_memory=False)
            houses[house] = hourly(raw)
    if len(houses) != 2: raise ValueError('Expected both measured houses')
    output = []
    for h in [1, 6, 24]:
        source = pd.concat([nest_source(name, item[3], h) for unit, item in ITEMS.items()
                            for name in [item[2], unit.lower()+'_2020-2021.csv']])
        x = source[FEATURES].to_numpy()
        model_daily = fit(x, (source.target-source.seasonal_yesterday).to_numpy())
        model_current = fit(x, (source.target-source.temp_now).to_numpy())
        for house, observations in houses.items():
            for room in ROOMS:
                frame = evaluation_frame(observations[room], h)
                if len(frame) < 24: continue
                x = frame[FEATURES].to_numpy()
                pred = {'persistence':frame.temp_now.to_numpy(),
                        'daily_seasonal':frame.seasonal_yesterday.to_numpy(),
                        'source_daily_ridge':frame.seasonal_yesterday.to_numpy()+predict(model_daily,x),
                        'source_current_ridge':frame.temp_now.to_numpy()+predict(model_current,x)}
                for label, prediction in pred.items():
                    err = prediction-frame.target.to_numpy()
                    output.append({'horizon_h':h, 'house':house, 'room':room, 'model':label,
                                   'n':len(frame), 'source_n':len(source),
                                   'mae_c':float(np.abs(err).mean()), 'rmse_c':float(np.sqrt(np.mean(err**2)))})
        print('Completed horizon', h, 'source rows',len(source),flush=True)
    result = pd.DataFrame(output)
    result.to_csv(OUT/'thermbuild_room_results.csv', index=False, float_format='%.8f')
    house = result.groupby(['horizon_h','house','model'],as_index=False).agg(
        rooms=('room','nunique'), mae_c=('mae_c','mean'), rmse_c=('rmse_c','mean'), observations=('n','sum'))
    house.to_csv(OUT/'thermbuild_house_results.csv',index=False,float_format='%.8f')
    total = house.groupby(['horizon_h','model'],as_index=False).agg(houses=('house','nunique'),
        rooms=('rooms','sum'), mae_c=('mae_c','mean'), rmse_c=('rmse_c','mean'))
    total.to_csv(OUT/'thermbuild_summary.csv',index=False,float_format='%.8f')
    quality = {house:{'hours':len(df), 'valid_hours_by_room':df.notna().sum().to_dict()} for house,df in houses.items()}
    (OUT/'thermbuild_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    print(total.to_string(index=False),flush=True)

if __name__ == '__main__': main()
