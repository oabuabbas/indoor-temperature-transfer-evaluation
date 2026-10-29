"""Descriptive calendar-month breakdown; varying contributing house sets."""
import pandas as pd
from audit_nest import OUT

def main():
    d=pd.read_csv(OUT/'ideal_holdout_v1_home_months.csv')
    keep=d[d.model.isin(['rolling_selector','local_ridge_30d'])]
    if keep.duplicated(['home','period_start','model']).any():raise ValueError('Duplicate monthly record')
    p=keep.pivot(index=['home','period_start'],columns='model',values='mae_c')
    if p.isna().any().any():raise ValueError('Incomplete paired months')
    p['selector_minus_local_c']=p.rolling_selector-p.local_ridge_30d
    p=p.reset_index()
    m=p.groupby('period_start',as_index=False).agg(homes=('home','nunique'),
        selector_mae_c=('rolling_selector','mean'),local_mae_c=('local_ridge_30d','mean'),
        mean_diff_c=('selector_minus_local_c','mean'),median_diff_c=('selector_minus_local_c','median'))
    m.to_csv(OUT/'ideal_holdout_monthly_secondary.csv',index=False,float_format='%.8f')
    print(m.to_string(index=False),flush=True)

if __name__=='__main__':main()
