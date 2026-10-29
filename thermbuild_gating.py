"""Rolling, target-past-only model selector; protocol_thermbuild_gating.md."""
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS
from adaptation_stress import fit, predict
from daily_residual import nest_source
from thermbuild_external import FEATURES, ROOMS, hourly, evaluation_frame

HORIZON = 6
BLOCK = 30 * 24
LENGTH = 442 * 24
CANDIDATES = ['persistence','daily_seasonal','source_daily_ridge']

def windows(frame, start, end, horizon=HORIZON):
    """Issue and target timestamps relative to first full day of the archive."""
    issue = frame.index.to_numpy()
    target = issue + horizon
    valid = (issue >= start-BLOCK) & (target < start)
    test = (issue >= start) & (target < end)
    return frame.loc[valid], frame.loc[test]

def predictions(frame, model):
    return pd.DataFrame({'persistence':frame.temp_now.to_numpy(),
                         'daily_seasonal':frame.seasonal_yesterday.to_numpy(),
                         'source_daily_ridge':frame.seasonal_yesterday.to_numpy()+predict(model,frame[FEATURES].to_numpy())},index=frame.index)

def main():
    source=pd.concat([nest_source(name,item[3],HORIZON) for unit,item in ITEMS.items()
                      for name in [item[2],unit.lower()+'_2020-2021.csv']])
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    roomblocks=[]
    with zipfile.ZipFile(DATA/'ThermBuild_measure_raw.zip') as archive:
        names=[x for x in archive.namelist() if x.endswith('_wetReal_raw.csv')]
        if len(names)!=2: raise ValueError('Expected two houses')
        houses=set()
        for name in names:
            house='Use1001_BSE1' if 'Use1001' in name else 'Use1002_BSE2' if 'Use1002' in name else None
            if house is None or house in houses: raise ValueError('Unexpected house')
            houses.add(house)
            with archive.open(name) as f:
                raw=pd.read_csv(f,usecols=['TIME','consecutive_days']+ROOMS,low_memory=False)
            data=hourly(raw)
            for room in ROOMS:
                frame=evaluation_frame(data[room],HORIZON)
                for start in range(BLOCK,LENGTH,BLOCK):
                    end=min(start+BLOCK,LENGTH)
                    validation,test=windows(frame,start,end)
                    if min(len(validation),len(test))<24: continue
                    pv=predictions(validation,model)
                    validation_mae=(pv.sub(validation.target,axis=0)).abs().mean()
                    winner=min(CANDIDATES,key=lambda c:validation_mae[c])
                    pt=predictions(test,model)
                    errors=(pt.sub(test.target,axis=0)).abs()
                    for label in CANDIDATES+['rolling_selector']:
                        chosen=winner if label=='rolling_selector' else label
                        roomblocks.append({'house':house,'room':room,'block':start//BLOCK,
                                           'issue_start':start,'issue_end_exclusive':end,
                                           'model':label,'selected':winner,'validation_n':len(validation),
                                           'test_n':len(test),'validation_selected_mae_c':validation_mae[winner],
                                           'absolute_error_sum':errors[chosen].sum(),'mae_c':errors[chosen].mean()})
    d=pd.DataFrame(roomblocks)
    d.to_csv(OUT/'thermbuild_gating_room_blocks.csv',index=False,float_format='%.8f')
    room=d.groupby(['house','room','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('absolute_error_sum','sum'),blocks=('block','nunique'))
    room['mae_c']=room.error_sum/room.n
    room.to_csv(OUT/'thermbuild_gating_rooms.csv',index=False,float_format='%.8f')
    house=room.groupby(['house','model'],as_index=False).agg(rooms=('room','nunique'),mae_c=('mae_c','mean'))
    summary=house.groupby('model',as_index=False).agg(houses=('house','nunique'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'thermbuild_gating_summary.csv',index=False,float_format='%.8f')
    block=d.groupby(['house','block','model'],as_index=False).agg(rooms=('room','nunique'),mae_c=('mae_c','mean'))
    block[block.rooms==len(ROOMS)].to_csv(OUT/'thermbuild_gating_house_blocks.csv',index=False,float_format='%.8f')
    chosen=d[d.model.eq('rolling_selector')].groupby(['house','selected']).size().rename('room_blocks').reset_index()
    chosen.to_csv(OUT/'thermbuild_gating_choices.csv',index=False)
    print(summary.to_string(index=False)); print('\nChoices\n',chosen.to_string(index=False),flush=True)

if __name__ == '__main__': main()
