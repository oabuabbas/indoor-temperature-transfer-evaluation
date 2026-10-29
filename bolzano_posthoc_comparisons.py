"""Post-result descriptive matched-support contrasts; not part of the fixed primary contrast."""
from pathlib import Path
import pandas as pd
out=Path(__file__).resolve().parent/'outputs'
d=pd.read_csv(out/'bolzano_external_v1_nominal_mean_predictions.csv.gz')
rows=[]
for method in ['daily_seasonal','local_ridge_30d','source_daily_ridge']:
    x=d[d.target.notna()&d.fallback.notna()&d[method].notna()]
    rows.append({'comparison':method,'n':len(x),'fallback_mae_c':(x.fallback-x.target).abs().mean(),'comparator_mae_c':(x[method]-x.target).abs().mean(),'design':'post-result descriptive matched support'})
pd.DataFrame(rows).to_csv(out/'bolzano_external_v1_posthoc_comparisons.csv',index=False,float_format='%.10f')
print(pd.DataFrame(rows).to_string(index=False))
