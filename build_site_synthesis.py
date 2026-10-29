"""Produce clearly labeled descriptive site results, never pooled across sites."""
import json
import pandas as pd
from audit_nest import OUT

def table(path,model_col='model'):
    df=pd.read_csv(OUT/path)
    if df[model_col].duplicated().any():raise ValueError('Duplicate model rows')
    return df.set_index(model_col).mae_c.to_dict()

def main():
    ideal=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf-8'))
    pleia=pd.read_csv(OUT/'pleia_rolling_summary.csv').query("population == 'all_eligible'")
    if pleia.model.duplicated().any():raise ValueError('Duplicate PLEIA model rows')
    pleia=pleia.set_index('model').mae_c.to_dict()
    germany=table('thermbuild_gating_summary.csv')
    us=pd.read_csv(OUT/'ornl_external_results.csv').query('horizon_h == 6')
    if len(us)!=210 or us[['case','room']].drop_duplicates().shape[0]!=70:raise ValueError('Unexpected ORNL cases')
    us=us.groupby('model').mae_c.mean().to_dict()
    rows=[]
    for site,status,source,unit,n,values in [
        ('IDEAL','locked held-out households','IDEAL earlier source houses','equal home',ideal['contributing_homes'],ideal['mean_mae_c']),
        ('PLEIA','exploratory','NEST','equal sensor',60,pleia),
        ('ThermBuild','exploratory','NEST','equal room within house; equal house',2,germany),
        ('ORNL','exploratory; source model includes hour features','NEST','equal case-room',70,us)]:
        for method in ['persistence','daily_seasonal','source_daily_ridge','rolling_selector','local_ridge_30d']:
            key='source_residual' if site=='ORNL' and method=='source_daily_ridge' else method
            rows.append({'site':site,'status':status,'source':source,'weight_unit':unit,'unit_count':n,
                         'horizon_h':6,'model':method,'mae_c':values.get(key)})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'study_synthesis_site_metrics.csv',index=False,float_format='%.8f')
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':main()
