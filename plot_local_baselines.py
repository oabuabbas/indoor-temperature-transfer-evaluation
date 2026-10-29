"""Publication-style exploratory plot; single site, conditional week resampling."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from audit_nest import OUT

d=pd.read_csv(OUT/'local_baselines_results.csv')
b=pd.read_csv(OUT/'local_baselines_bootstrap.csv')
labels={'daily_seasonal':'Previous-day reference','persistence':'Persistence','source_residual':'Source ridge','local_ridge':'Local ridge','local_histgb':'Local gradient boosting'}
order=list(labels);means=d.groupby('model').mae_c.mean().reindex(order)
fig,(ax,ax2)=plt.subplots(1,2,figsize=(11.8,4.6),gridspec_kw={'width_ratios':[1,1.05]})
colors=['#165d72','#79949a','#79949a','#79949a','#79949a']
ax.barh(range(len(order)),means, color=colors);ax.set_yticks(range(len(order)),[labels[x] for x in order]);ax.invert_yaxis();ax.set_xlabel('Mean absolute error (°C)');ax.set_title('Six-hour forecast error',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
for i,v in enumerate(means):ax.text(v+.006,i,f'{v:.3f}',va='center',fontsize=9)
ax.set_xlim(0,max(means)*1.22)
names=['persistence','source_residual','local_ridge','local_histgb']
for i,name in enumerate(names):
    row=b[b.comparison==name+' minus daily_seasonal'].iloc[0]
    ax2.plot([row.q025_c,row.q975_c],[i,i],color='#447785',lw=2)
    ax2.scatter([row.observed_diff_c],[i],color='#133f51',s=30,zorder=3)
ax2.axvline(0,color='#555555',lw=1,ls='--');ax2.set_yticks(range(len(names)),[labels[x] for x in names]);ax2.invert_yaxis();ax2.set_xlabel('MAE difference vs previous day (°C)');ax2.set_title('Resampled calendar-week ranges',loc='left',fontweight='bold');ax2.grid(axis='x',alpha=.2);ax2.set_axisbelow(True)
fig.suptitle('PLEIAData: exploratory single-building comparison',x=.05,ha='left',fontsize=14,fontweight='bold')
fig.text(.05,.01,'59 sensors in one building · March–17 December 2021 · ranges are conditional week resampling, not new-site confidence intervals',fontsize=8,color='#444444')
fig.tight_layout(rect=[0,.05,1,.91]);fig.savefig(OUT/'local_baselines_comparison.png',dpi=220,bbox_inches='tight');print(OUT/'local_baselines_comparison.png')
