"""Compare 7/14/30-day selector histories on common evaluation rows."""
import zipfile
import numpy as np
import pandas as pd
from audit_nest import DATA, OUT, ITEMS
from adaptation_stress import fit
from daily_residual import nest_source
from external_pleia import read_pleia
from thermbuild_external import ROOMS, FEATURES, hourly, evaluation_frame
from thermbuild_gating import CANDIDATES, predictions
from pleia_rolling_selector import make_frame

H=6
LENGTHS=[7,14,30]
INTERVAL=pd.Timedelta(days=30)
PSTART=pd.Timestamp('2021-01-31',tz='UTC')
PEND=pd.Timestamp('2021-12-18',tz='UTC')

def subset(frame,start,end,days,horizon):
    target=frame.index+horizon
    valid=frame[(frame.index>=start-days)&(target<start)]
    test=frame[(frame.index>=start)&(target<end)]
    return valid,test

def evaluate(site,house,entity,frame,blocks,horizon,day,model):
    out=[]
    for number,start,end in blocks:
        histories={length:subset(frame,start,end,length*day,horizon) for length in LENGTHS}
        tests=[v[1] for v in histories.values()]
        if any(len(pair[0])<24 for pair in histories.values()) or len(tests[0])<24: continue
        if not all(t.index.equals(tests[0].index) for t in tests): raise AssertionError('Different test support')
        test=tests[0]
        errors=predictions(test,model).sub(test.target,axis=0).abs()
        choices={}
        for length,(valid,_) in histories.items():
            vmae=predictions(valid,model).sub(valid.target,axis=0).abs().mean()
            choices[f'select_{length}d']=(min(CANDIDATES,key=lambda c:vmae[c]),len(valid))
        for c in CANDIDATES: choices[c]=(c,None)
        for method,(choice,nvalid) in choices.items():
            err=errors[choice]
            out.append({'site':site,'house':house,'entity':entity,'block':number,'model':method,
                        'selected':choice,'validation_n':nvalid,'test_n':len(test),
                        'absolute_error_sum':float(err.sum()),'mae_c':float(err.mean())})
    return out

def main():
    source=pd.concat([nest_source(name,item[3],H) for unit,item in ITEMS.items()
                      for name in [item[2],unit.lower()+'_2020-2021.csv']])
    model=fit(source[FEATURES].to_numpy(),(source.target-source.seasonal_yesterday).to_numpy())
    rows=[]
    with zipfile.ZipFile(DATA/'ThermBuild_measure_raw.zip') as z:
        names=[x for x in z.namelist() if x.endswith('_wetReal_raw.csv')]
        if len(names)!=2: raise ValueError('Expected two houses')
        houses=set()
        for name in names:
            house='Use1001_BSE1' if 'Use1001' in name else 'Use1002_BSE2' if 'Use1002' in name else None
            if house is None or house in houses: raise ValueError('Unexpected house')
            houses.add(house)
            with z.open(name) as f: raw=pd.read_csv(f,usecols=['TIME','consecutive_days']+ROOMS,low_memory=False)
            d=hourly(raw)
            blocks=[(start//720,start,min(start+720,442*24)) for start in range(720,442*24,720)]
            for room in ROOMS:
                rows.extend(evaluate('ThermBuild',house,room,evaluation_frame(d[room],H),blocks,H,24,model))
    hour,_=read_pleia()
    blocks=[(i,start,min(start+INTERVAL,PEND)) for i,start in enumerate(pd.date_range(PSTART,PEND,freq=INTERVAL,inclusive='left'),1)]
    for sensor,group in hour.groupby('IDdevice'):
        rows.extend(evaluate('PLEIA','site',str(sensor),make_frame(group),blocks,pd.Timedelta(hours=H),pd.Timedelta(days=1),model))
    d=pd.DataFrame(rows)
    if d.empty: raise ValueError('No eligible blocks')
    d.to_csv(OUT/'selector_history_room_blocks.csv',index=False,float_format='%.8f')
    entity=d.groupby(['site','house','entity','model'],as_index=False).agg(n=('test_n','sum'),error_sum=('absolute_error_sum','sum'),blocks=('block','nunique'))
    entity['mae_c']=entity.error_sum/entity.n
    entity.to_csv(OUT/'selector_history_entities.csv',index=False,float_format='%.8f')
    house=entity.groupby(['site','house','model'],as_index=False).agg(entities=('entity','nunique'),mae_c=('mae_c','mean'))
    summary=house.groupby(['site','model'],as_index=False).agg(houses=('house','nunique'),entities=('entities','sum'),mae_c=('mae_c','mean'))
    summary.to_csv(OUT/'selector_history_summary.csv',index=False,float_format='%.8f')
    choices=d[d.model.str.startswith('select_')].groupby(['site','model','selected']).size().rename('entity_blocks').reset_index()
    choices.to_csv(OUT/'selector_history_choices.csv',index=False)
    print(summary.to_string(index=False));print('\nChoices\n',choices.to_string(index=False),flush=True)

if __name__=='__main__':main()
