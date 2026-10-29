"""Exploratory cross-unit ridge baseline, one-hour mean forecast, no target fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS, TEST_START, TEST_END

TRAIN_END = pd.Timestamp('2020-03-01')
ALPHA = 100.0

def frames(unit, name, rooms):
    use = ['time', 'temp_amb', 'irrad'] + rooms
    df = pd.read_csv(DATA / name, usecols=use, low_memory=False)
    t = pd.to_datetime(df.pop('time'), errors='coerce')
    unique = t.notna() & ~t.duplicated(keep=False)
    df = df.loc[unique].apply(pd.to_numeric, errors='coerce')
    df.index = t[unique]
    mean = df.resample('1h').mean()
    count = df.resample('1h').count()
    mean = mean.where(count >= 45)
    data = []
    for room in rooms:
        temp = mean[room]
        hour = mean.index.hour.to_numpy()
        frame = pd.DataFrame({'temp_now':temp, 'temp_lag1':temp.shift(1), 'ambient_now':mean.temp_amb, 'solar_now':mean.irrad, 'hour_sin':np.sin(2*np.pi*hour/24), 'hour_cos':np.cos(2*np.pi*hour/24), 'target':temp.shift(-1)}, index=mean.index)
        frame['room'] = room
        frame['unit'] = unit
        data.append(frame.dropna())
    return pd.concat(data)

def main():
    data = pd.concat([frames(unit, item[2], item[3]) for unit,item in ITEMS.items()])
    cols = ['temp_now','temp_lag1','ambient_now','solar_now','hour_sin','hour_cos']
    output=[]
    for target in ITEMS:
        train = data[(data.unit != target) & (data.index < TRAIN_END)]
        # Feature timestamps precede the target by one hour, so test bounds apply to target time.
        target_time = data.index + pd.Timedelta(hours=1)
        test = data[(data.unit == target) & (target_time >= TEST_START) & (target_time < TEST_END)]
        x=train[cols].to_numpy(); y=train.target.to_numpy()
        mu=x.mean(axis=0); sd=x.std(axis=0); sd[sd==0]=1
        z=(x-mu)/sd; ymean=y.mean()
        beta=np.linalg.solve(z.T@z + ALPHA*np.eye(len(cols)), z.T@(y-ymean))
        for room, group in test.groupby('room'):
            actual=group.target.to_numpy()
            ridge=(group[cols].to_numpy()-mu)/sd @ beta + ymean
            persist=group.temp_now.to_numpy()
            for model,pred in [('ridge_source_only',ridge),('persistence',persist)]:
                err=pred-actual
                output.append({'target_unit':target,'room':room,'model':model,'test_pairs':len(err),'train_pairs':len(train),'mae_c':float(np.mean(np.abs(err))),'rmse_c':float(np.sqrt(np.mean(err**2)))})
        print(target, 'train', len(train), 'test', len(test), flush=True)
    pd.DataFrame(output).round(5).to_csv(OUT/'transfer_pilot.csv',index=False)

if __name__ == '__main__': main()
