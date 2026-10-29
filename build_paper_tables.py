"""Rebuild descriptive manuscript tables from frozen output files."""
import json
import pandas as pd
from audit_nest import OUT

def main():
    root=OUT.parent/'paper';root.mkdir(exist_ok=True)
    primary=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf-8'))
    sec=json.loads((OUT/'ideal_holdout_range_secondary_summary.json').read_text(encoding='utf-8'))
    site=pd.read_csv(OUT/'study_synthesis_site_metrics.csv')
    labels={'persistence':'Persistence','daily_seasonal':'Previous-day reference','local_histgb_30d':'Local histogram gradient boosting',
            'local_ridge_30d':'Local ridge (30 days)','source_daily_ridge':'Transferred ridge','rolling_selector':'Rolling selector'}
    order=list(labels)
    lines=['# Tables for working manuscript','',
           'Generated from saved results; author must verify before submission. IDEAL is the held-out evaluation; site experiments are nonharmonized and exploratory.','',
           '## Table 1. IDEAL holdout performance','',
           '| Method | Equal-home MAE (°C) | Homes |','|---|---:|---:|']
    for method in order:lines.append(f"| {labels[method]} | {primary['mean_mae_c'][method]:.4f} | {primary['contributing_homes']} |")
    lines+=['',f"Primary selector minus local ridge: {primary['primary_selector_minus_local_c']:+.4f}°C; descriptive house-reweighting percentiles [{primary['descriptive_home_resample_interval_c'][0]:+.4f}, {primary['descriptive_home_resample_interval_c'][1]:+.4f}]°C, not a population/site CI.",'',
            '## Table 2. Secondary filter sensitivity on common issue times','',
            '| Method | 10–45°C | 15–40°C |','|---|---:|---:|']
    for method in ['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector']:
        lines.append(f"| {labels[method]} | {sec['mean_mae_c']['10_45'][method]:.4f} | {sec['mean_mae_c']['15_40'][method]:.4f} |")
    lines+=['',f"Post-primary exploratory analysis: {sec['homes']} homes, {sec['eligible_home_months']} home-months, {sec['test_issue_times']:,} common issue times; aggregated targets can differ.",'',
            '## Table 3. Site-specific results (IDEAL holdout; other sites exploratory)','',
            '| Site | Weighting unit | Persistence | Previous day | Transferred | Selector |','|---|---|---:|---:|---:|---:|']
    for name in ['IDEAL','PLEIA','ThermBuild','ORNL']:
        s=site[site.site.eq(name)].set_index('model')
        def value(method):
            x=s.loc[method,'mae_c']
            return 'Not evaluated' if pd.isna(x) else f'{x:.4f}'
        lines.append(f"| {name} | {s.iloc[0]['weight_unit']} | {value('persistence')} | {value('daily_seasonal')} | {value('source_daily_ridge')} | {value('rolling_selector')} |")
    lines+=['','Do not compare sites as a harmonized experiment or average their values. ORNL transferred model includes hour features; other experiments use temperature-only predictors.','']
    (root/'02_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote',root/'02_tables.md')

if __name__=='__main__':main()
