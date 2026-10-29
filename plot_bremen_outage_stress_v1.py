from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'outputs'
d=pd.read_csv(out/'bremen_outage_stress_v1_replicates.csv')
order=['none','iid05','iid20','block6h_7d','block24h_14d','block72h_30d']
labels=['None','IID 5%','IID 20%','6 h / 7 d','24 h / 14 d','72 h / 30 d']
g=d.groupby('scenario').mean(numeric_only=True).loc[order]
x=np.arange(len(order));w=.35
fig,(ax,ax2)=plt.subplots(2,1,figsize=(9,7.1),layout='constrained')
a=ax.bar(x-w/2,100*g.fallback_ready/g.opportunities,w,label='Fallback policy',color='#1b6e70')
b=ax.bar(x+w/2,100*g.persistence_ready/g.opportunities,w,label='Persistence',color='#d1833f')
for z in (a,b):ax.bar_label(z,fmt='%.1f',padding=2,fontsize=8)
ax.set_ylim(0,112);ax.set_ylabel('Ready issue opportunities (%)');ax.set_xticks(x,labels);ax.set_title('Synthetic input outages: readiness and paired forecast accuracy')
a=ax2.bar(x-w/2,g.paired_fallback_equal_room_mae,w,label='Fallback policy',color='#1b6e70')
b=ax2.bar(x+w/2,g.paired_persistence_equal_room_mae,w,label='Persistence',color='#d1833f')
for z in (a,b):ax2.bar_label(z,fmt='%.3f',padding=2,fontsize=8)
ax2.set_ylim(0,.75);ax2.set_ylabel('Paired MAE (°C)');ax2.set_xticks(x,labels)
fig.legend([a,b],['Fallback policy','Persistence'],loc='lower center',bbox_to_anchor=(.5,-.055),ncol=2,frameon=False)
fig.text(.5,-.09,'Five mask replicates per outage scenario; four correlated rooms; target is an unmasked retrospective oracle.',ha='center',fontsize=9)
fig.savefig(out/'bremen_outage_stress_v1.png',dpi=250,bbox_inches='tight')
fig.savefig(out/'bremen_outage_stress_v1.svg',bbox_inches='tight')
