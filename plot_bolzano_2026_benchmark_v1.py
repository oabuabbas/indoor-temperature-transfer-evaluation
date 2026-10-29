from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'outputs';s=json.loads((p/'bolzano_2026_benchmark_v1_summary.json').read_text())['results']['nominal_mean']
d=pd.DataFrame(s['common_metrics']).sort_values('mae_c',ascending=True)
labels={'persistence':'Persistence','daily_seasonal':'Previous day','source_daily_ridge':'Transferred ridge','local_ridge_30d':'Local 30-day ridge','rolling_selector':'Rolling selector','fallback':'Fallback policy','local_hist_gradient_boosting':'Local gradient boosting'}
fig,ax=plt.subplots(figsize=(8.4,4.8),layout='constrained'); bars=ax.barh([labels[x] for x in d.model],d.mae_c,color=['#1b6e70' if x=='fallback' else '#6685a0' for x in d.model]);ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.set_xlim(0,.86);ax.set_xlabel('Mean absolute error (°C)');ax.set_title('Bolzano Jan–Apr 2026: seven methods on common support')
fig.text(.5,-.05,'2,770 shared observed targets; one room, nominal unverified clock; fixed model settings.',ha='center',fontsize=9)
fig.savefig(p/'bolzano_2026_benchmark_v1.png',dpi=250,bbox_inches='tight');fig.savefig(p/'bolzano_2026_benchmark_v1.svg',bbox_inches='tight')
