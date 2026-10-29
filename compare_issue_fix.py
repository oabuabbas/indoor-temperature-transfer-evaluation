"""Compare archived aggregate rows with corrected windows; no common-hour claim."""
import pandas as pd
from audit_nest import OUT

FILES={
'multiseason_results.csv':['unit','room','season','model','scenario'],
'horizon_results.csv':['horizon_h','unit','room','season','model'],
'pleia_external_results.csv':['horizon_h','sensor','model'],
'pleia_adaptation_results.csv':['horizon_h','sensor','period','model'],
'pleia_adaptation_min6_results.csv':['horizon_h','sensor','period','model'],
}
def main():
    summary=[]
    for file,keys in FILES.items():
        old=pd.read_csv(OUT/'legacy_pre_issue_fix'/file,dtype={'sensor':'string'})
        new=pd.read_csv(OUT/file,dtype={'sensor':'string'})
        paired=old.merge(new,on=keys,how='outer',suffixes=('_old','_new'),indicator=True,validate='one_to_one')
        both=paired[paired._merge=='both']
        cov=(both.coverage90_new-both.coverage90_old).dropna()
        test=(both.test_n_new-both.test_n_old).dropna()
        cal=(both.cal_n_new-both.cal_n_old).dropna()
        summary.append({'file':file,'old_rows':len(old),'new_rows':len(new),'paired_rows':len(both),'old_only':int((paired._merge=='left_only').sum()),'new_only':int((paired._merge=='right_only').sum()),'mean_coverage_old':both.coverage90_old.mean(),'mean_coverage_new':both.coverage90_new.mean(),'mean_abs_coverage_change_pp':cov.abs().mean()*100,'max_abs_coverage_change_pp':cov.abs().max()*100,'mean_test_n_change':test.mean(),'mean_cal_n_change':cal.mean()})
    pd.DataFrame(summary).round(6).to_csv(OUT/'issue_fix_comparison.csv',index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))

if __name__=='__main__':main()
