from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'outputs'
d=pd.read_csv(out/'bolzano_external_v1_nominal_mean_monthly.csv');d['difference']=d.fallback_mae_c-d.persistence_mae_c
fig,ax=plt.subplots(figsize=(10.5,4.6),layout='constrained')
colors=['#1b6e70' if x<0 else '#d1833f' if x>0 else '#aaaaaa' for x in d.difference]
ax.bar(range(len(d)),d.difference,color=colors);ax.axhline(0,color='black',lw=.8)
ax.set_xticks(range(len(d)),[x[:7] for x in d.month],rotation=65,ha='right',fontsize=8)
ax.set_ylabel('Fallback MAE − persistence MAE (°C)')
ax.set_title('Bolzano: monthly paired differences, nominal unverified clock')
fig.text(.5,-.04,'Negative favors fallback; one room, correlated months, same observed targets within each month.',ha='center',fontsize=9)
fig.savefig(out/'bolzano_external_v1_monthly.png',dpi=250,bbox_inches='tight')
fig.savefig(out/'bolzano_external_v1_monthly.svg',bbox_inches='tight')
