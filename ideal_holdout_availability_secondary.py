"""Issue-time availability and retrospective scoring support, no new forecasts."""
import json
import numpy as np
import pandas as pd
from audit_nest import OUT
from ideal_development_prepare import select
from ideal_development_v1 import frame_from_series,windows,H,TEST_START,TEST_END
from ideal_holdout_prepare import HOURLY

def build_counts(series,index,history_ok):
    """Features use t, t-1, t-18; observed target is computed separately."""
    s=series.sort_index().asfreq('1h')
    now=s.reindex(index).notna().to_numpy()
    lag1=s.reindex(index-pd.Timedelta(hours=1)).notna().to_numpy()
    lag18=s.reindex(index-pd.Timedelta(hours=18)).notna().to_numpy()
    target=s.reindex(index+pd.Timedelta(hours=H)).notna().to_numpy()
    source=now & lag1 & lag18
    gate=source & bool(history_ok)
    fallback=gate | now | lag18
    return {'persistence_ready':now,'daily_ready':lag18,'source_ready':source,
            'selector_ready':gate,'selector_evaluable':gate & target,'source_evaluable':source & target,
            'fallback_ready':fallback,'fallback_evaluable':fallback & target}

def main():
    selected=select('holdout').drop(columns='dev_role').reset_index(drop=True)
    if not pd.read_csv(OUT/'ideal_holdout_sensors_v1.csv').equals(selected):raise ValueError('Holdout selection changed')
    quality=pd.read_csv(OUT/'ideal_sensor_quality.csv').set_index('sensor')
    orig=pd.read_csv(OUT/'ideal_holdout_v1_home_months.csv');orig=orig[orig.model.eq('rolling_selector')]
    exclusions=pd.read_csv(OUT/'ideal_holdout_v1_exclusions.csv')
    if orig.duplicated(['home','period_start']).any() or exclusions.duplicated(['home','period_start']).any():raise ValueError('Duplicated original window')
    orig=orig.set_index(['home','period_start']);exclusions=exclusions.set_index(['home','period_start'])
    starts=pd.date_range(TEST_START,TEST_END,freq='MS',inclusive='left');records=[]
    for item in selected.itertuples(index=False):
        raw=pd.read_csv(HOURLY/f'{item.sensor}.csv.gz',parse_dates=['hour_utc'])
        if raw.empty or raw.hour_utc.duplicated().any():raise ValueError('Missing or duplicated hourly series')
        s=raw.set_index('hour_utc').temp_c.sort_index().asfreq('1h')
        frame=frame_from_series(s)
        first=pd.Timestamp(quality.loc[item.sensor,'first_utc']).floor('h')
        last=pd.Timestamp(quality.loc[item.sensor,'last_utc']).floor('h')
        for start in starts:
            end=min(start+pd.offsets.MonthBegin(1),TEST_END)
            history,test=windows(frame,start,end);period=start.isoformat();key=(item.home,period)
            score_ok=len(history)>=24 and len(test)>=24
            if score_ok:
                if key not in orig.index:raise ValueError('Missing primary included window')
                row=orig.loc[key]
                if int(row.history_n)!=len(history) or int(row.test_n)!=len(test):raise ValueError('Primary count mismatch')
            else:
                if key not in exclusions.index:raise ValueError('Missing primary excluded window')
                row=exclusions.loc[key]
                if int(row.history_n)!=len(history) or int(row.test_n)!=len(test):raise ValueError('Excluded count mismatch')
            index=pd.date_range(start,end-pd.Timedelta(hours=H),freq='h',inclusive='left')
            flags=build_counts(s,index,len(history)>=24)
            within=((index>=first)&(index<=last))
            record={'home':int(item.home),'sensor':int(item.sensor),'period_start':period,
                    'history_n':len(history),'test_n':len(test),'primary_included':score_ok,
                    'calendar_n':len(index),'span_n':int(within.sum())}
            for name,flag in flags.items():
                record['calendar_'+name]=int(flag.sum())
                record['span_'+name]=int((flag & within).sum())
            record['primary_scored_rows']=len(test) if score_ok else 0
            if score_ok and record['calendar_selector_evaluable']!=len(test):raise ValueError('Reconstructed scoreable rows differ')
            records.append(record)
    d=pd.DataFrame(records)
    if len(d)!=73*9 or int(d.primary_included.sum())!=477 or int(d.primary_scored_rows.sum())!=239215:
        raise ValueError('Primary support does not reproduce')
    d.to_csv(OUT/'ideal_holdout_availability_home_months.csv',index=False)
    group=d.groupby('home',as_index=False).sum(numeric_only=True)
    group['span_selector_rate']=np.divide(group.span_selector_ready,group.span_n,
        out=np.full(len(group),np.nan),where=group.span_n.to_numpy()>0)
    group.to_csv(OUT/'ideal_holdout_availability_homes.csv',index=False,float_format='%.8f')
    totals={key:int(d[key].sum()) for key in d if key.startswith('calendar_') or key.startswith('span_') or key=='primary_scored_rows'}
    primary_windows=int(d.primary_included.sum())
    result={'homes':int(d.home.nunique()),'home_months':len(d),'history_eligible_home_months':int((d.history_n>=24).sum()),
            'primary_included_home_months':primary_windows,'primary_excluded_home_months':int((~d.primary_included).sum()),
            'totals':totals,'span_rates':{name:totals['span_'+name]/totals['span_n'] for name in
                ['persistence_ready','daily_ready','source_ready','selector_ready','selector_evaluable','fallback_ready','fallback_evaluable']},
            'calendar_rates':{name:totals['calendar_'+name]/totals['calendar_n'] for name in
                ['persistence_ready','daily_ready','source_ready','selector_ready','selector_evaluable','fallback_ready','fallback_evaluable']},
            'selector_evaluable_in_primary_excluded_windows':int(d.loc[~d.primary_included,'calendar_selector_evaluable'].sum()),
            'home_span_selector_rate_median':float(group.span_selector_rate.median()),
            'home_span_selector_rate_min':float(group.span_selector_rate.min()),
            'home_span_selector_rate_max':float(group.span_selector_rate.max()),
            'scope':'Retrospective recorded-span availability proxy, not live uptime or a probability for new sites.'}
    (OUT/'ideal_holdout_availability_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
