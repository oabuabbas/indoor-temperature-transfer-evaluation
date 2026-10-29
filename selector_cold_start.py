"""Compare past-only selection at days 7, 14 and 30 on phase-matched rows."""
import zipfile
import pandas as pd
from audit_nest import DATA, OUT, ITEMS
from adaptation_stress import fit
from daily_residual import nest_source
from external_pleia import read_pleia
from thermbuild_external import ROOMS, FEATURES, hourly, evaluation_frame
from thermbuild_gating import CANDIDATES, predictions
from pleia_rolling_selector import make_frame
from selector_history_sensitivity import subset

H=6
PHASES=[('days_7_14',7,14,[7]),('days_14_30',14,30,[7,14]),('days_30_60',30,60,[7,14,30])]

def evaluate(site,house,entity,frame,origin,day,horizon,model):
    # Both source and selector must be constructible without pre-origin readings.
    frame=frame[frame.index >= origin + 18 * day / 24]
    result=[]
    for phase,start_day,end_day,lengths in PHASES:
        start,end=origin+start_day*day,origin+end_day*day
        history={n:subset(frame,start,end,n*day,horizon) for n in lengths}
        test=history[lengths[0]][1]
        if len(test)<24 or any(len(pair[0])<24 for pair in history.values()): continue
        if not all(test.index.equals(pair[1].index) for pair in history.values()): raise AssertionError('Unequal test rows')
        errors=predictions(test,model).sub(test.target,axis=0).abs()
        methods={c:(c,None) for c in CANDIDATES}
        for n,(valid,_) in history.items():
            mae=predictions(valid,model).sub(valid.target,axis=0).abs().mean()
            methods[f'select_{n}d']=(min(CANDIDATES,key=lambda x:mae[x]),len(valid))
        for method,(choice,valid_n) in methods.items():
            result.append({'site':site,'house':house,'entity':entity,'phase':phase,'method':method,
                           'selected':choice,'validation_n':valid_n,'test_n':len(test),
                           'mae_c':float(errors[choice].mean())})
    return result

def main():
    source=pd.concat([nest_source(name,item[3],H) for unit,item in ITEMS.items()
                      for name in [item[2],unit.lower()+'_2020-2021.csv']])
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    rows=[]
    with zipfile.ZipFile(DATA/'ThermBuild_measure_raw.zip') as z:
        names=[x for x in z.namelist() if x.endswith('_wetReal_raw.csv')]
        if len(names)!=2:raise ValueError('Expected two houses')
        houses=set()
        for name in names:
            house='Use1001_BSE1' if 'Use1001' in name else 'Use1002_BSE2' if 'Use1002' in name else None
            if house is None or house in houses: raise ValueError('Unexpected house')
            houses.add(house)
            with z.open(name) as f: raw=pd.read_csv(f,usecols=['TIME','consecutive_days']+ROOMS,low_memory=False)
            d=hourly(raw)
            for room in ROOMS:
                rows.extend(evaluate('ThermBuild',house,room,evaluation_frame(d[room],H),0,24,H,model))
    hour,_=read_pleia()
    origin=pd.Timestamp('2021-01-01',tz='UTC')
    for sensor,group in hour.groupby('IDdevice'):
        rows.extend(evaluate('PLEIA','site',str(sensor),make_frame(group),origin,pd.Timedelta(days=1),pd.Timedelta(hours=H),model))
    d=pd.DataFrame(rows)
    if d.empty:raise ValueError('No eligible cases')
    d.to_csv(OUT/'selector_cold_start_entities.csv',index=False,float_format='%.8f')
    house=d.groupby(['site','phase','house','method'],as_index=False).agg(entities=('entity','nunique'),mae_c=('mae_c','mean'))
    summary=house.groupby(['site','phase','method'],as_index=False).agg(houses=('house','nunique'),entities=('entities','sum'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'selector_cold_start_summary.csv',index=False,float_format='%.8f')
    counts=d[d.method.str.startswith('select_')].groupby(['site','phase','method','selected']).size().rename('entity_count').reset_index()
    counts.to_csv(OUT/'selector_cold_start_choices.csv',index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
