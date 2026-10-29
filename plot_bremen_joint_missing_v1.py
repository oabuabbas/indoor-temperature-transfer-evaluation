from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'outputs';d=pd.read_csv(out/'bremen_joint_missing_v1_replicates.csv')
order=['common_iid10','common_block24_14d','temperature_dependent'];names=['Shared IID 10%','Shared 24 h / 14 d','Temperature-dependent']
g=d.groupby('scenario').mean(numeric_only=True).loc[order];x=np.arange(3);w=.36
fig,(ax,ax2)=plt.subplots(2,1,figsize=(9,7.5),layout='constrained')
a=ax.bar(x-w/2,100*g.fallback_ready/g.opportunities,w,color='#1b6e70');b=ax.bar(x+w/2,100*g.persistence_ready/g.opportunities,w,color='#d1833f')
for z in (a,b):ax.bar_label(z,fmt='%.1f',padding=2,fontsize=9)
ax.set_ylim(0,111);ax.set_xticks(x,names);ax.set_ylabel('Ready issue opportunities (%)');ax.set_title('Shared and state-dependent synthetic outages')
fig.legend([a,b],['Fallback policy','Persistence'],loc='upper center',bbox_to_anchor=(.5,-.005),ncol=2,frameon=False)
a=ax2.bar(x-w/2,g.oracle_extra_scored,w,color='#496c9c');b=ax2.bar(x+w/2,g.same_sensor_extra_scored,w,color='#b66060')
for z in (a,b):ax2.bar_label(z,fmt='%.0f',padding=3,fontsize=9)
ax2.set_ylim(0,3200);ax2.set_xticks(x,names);ax2.set_ylabel('Extra forecasts scored (count)');ax2.legend([a,b],['Unmasked target oracle','Same sensor survives at target'],frameon=False,loc='upper right')
fig.text(.5,-.09,'Means of five synthetic masks; four correlated rooms; target visibility is an assumed observation model.',ha='center',fontsize=9)
fig.savefig(out/'bremen_joint_missing_v1.png',dpi=250,bbox_inches='tight')
fig.savefig(out/'bremen_joint_missing_v1.svg',bbox_inches='tight')
