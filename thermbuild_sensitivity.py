"""Post-result exploratory sensitivity; see protocol_thermbuild_sensitivity.md."""
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS
from adaptation_stress import fit, predict
from daily_residual import nest_source
from thermbuild_external import FEATURES, ROOMS, hourly, evaluation_frame

HORIZON = 6

def scores(frame, model):
    x = frame[FEATURES].to_numpy()
    y = frame.target.to_numpy()
    return pd.DataFrame({'persistence':np.abs(frame.temp_now.to_numpy()-y),
                         'daily_seasonal':np.abs(frame.seasonal_yesterday.to_numpy()-y),
                         'source_daily_ridge':np.abs(frame.seasonal_yesterday.to_numpy()+predict(model,x)-y)},
                        index=frame.index)

def main():
    source = pd.concat([nest_source(name,item[3],HORIZON) for unit,item in ITEMS.items()
                        for name in [item[2],unit.lower()+'_2020-2021.csv']])
    model = fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    room_results, block_results, quality = [], [], []
    with zipfile.ZipFile(DATA/'ThermBuild_measure_raw.zip') as archive:
        names = [name for name in archive.namelist() if name.endswith('_wetReal_raw.csv')]
        if len(names) != 2: raise ValueError('Expected two measured houses')
        for name in names:
            house = 'Use1001_BSE1' if 'Use1001' in name else 'Use1002_BSE2' if 'Use1002' in name else None
            if house is None: raise ValueError('Unexpected house')
            with archive.open(name) as f:
                raw = pd.read_csv(f,usecols=['TIME','consecutive_days']+ROOMS,low_memory=False)
            filtered = {'15_40':hourly(raw,15,40),'10_45':hourly(raw,10,45)}
            rows = raw[raw.consecutive_days.between(2,443)]
            for room in ROOMS:
                val = pd.to_numeric(rows[room],errors='coerce').replace([np.inf,-np.inf],np.nan)
                quality.append({'house':house,'room':room,'total_raw':len(val),'nonmissing':int(val.notna().sum()),
                                'outside_15_40':int((val.notna() & ~val.between(15,40)).sum()),
                                'outside_10_45':int((val.notna() & ~val.between(10,45)).sum())})
                frames = {key:evaluation_frame(df[room],HORIZON) for key,df in filtered.items()}
                common = frames['15_40'].index.intersection(frames['10_45'].index)
                for key,frame in frames.items():
                    for support,subset in [('own',frame),('common',frame.loc[common])]:
                        errors = scores(subset,model)
                        for label in errors:
                            room_results.append({'house':house,'room':room,'filter':key,'support':support,
                                                 'model':label,'n':len(errors),'mae_c':errors[label].mean()})
                    if key == '15_40':
                        errors = scores(frame,model)
                        blocks = (errors.index.to_numpy()+HORIZON)//(60*24)
                        for block,part in errors.groupby(blocks):
                            for label in part:
                                block_results.append({'house':house,'room':room,'block':int(block)+1,
                                                      'target_hour_start':int(block)*1440,
                                                      'target_hour_end_exclusive':min((int(block)+1)*1440,442*24),
                                                      'model':label,'n':len(part),'mae_c':part[label].mean()})
    room = pd.DataFrame(room_results)
    room.to_csv(OUT/'thermbuild_sensitivity_rooms.csv',index=False,float_format='%.8f')
    pd.DataFrame(quality).to_csv(OUT/'thermbuild_sensitivity_quality.csv',index=False)
    blocks = pd.DataFrame(block_results)
    blocks.to_csv(OUT/'thermbuild_temporal_rooms.csv',index=False,float_format='%.8f')
    allkeys=['filter','support','model']
    house = room.groupby(allkeys+['house'],as_index=False).agg(rooms=('room','nunique'),mae_c=('mae_c','mean'))
    summary = house.groupby(allkeys,as_index=False).agg(houses=('house','nunique'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'thermbuild_sensitivity_summary.csv',index=False,float_format='%.8f')
    block_house = blocks.groupby(['house','block','target_hour_start','target_hour_end_exclusive','model'],as_index=False).agg(
        rooms=('room','nunique'),min_n=('n','min'),mae_c=('mae_c','mean'))
    block_house = block_house[(block_house.rooms == len(ROOMS)) & (block_house.min_n >= 24)]
    block_house.to_csv(OUT/'thermbuild_temporal_house.csv',index=False,float_format='%.8f')
    print(summary.to_string(index=False),flush=True)
    print('\nHouse blocks\n',block_house.to_string(index=False),flush=True)

if __name__ == '__main__': main()
