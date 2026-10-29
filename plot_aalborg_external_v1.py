from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'outputs';d=pd.read_csv(p/'aalborg_external_v1_rooms.csv');x=d.pivot(index='room',columns='model',values='mae_c').sort_index();pos=np.arange(len(x));w=.2
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
for i,(model,label,color) in enumerate([('source_daily_ridge','Transferred ridge','#4d7e9f'),('rolling_selector','Rolling selector','#1b6e70'),('local_ridge_30d','Local ridge','#d1833f'),('hgb_ideal_selected','Development-selected boosting','#ab6868')]):
 b=ax.bar(pos+(i-1.5)*w,x[model],w,color=color,label=label);ax.bar_label(b,fmt='%.2f',padding=2,rotation=90,fontsize=7)
ax.set_xticks(pos,x.index);ax.set_ylim(0,1.3);ax.set_ylabel('MAE (°C)');ax.set_title('Aalborg office rooms: matched-support forecast errors');ax.legend(frameon=False,ncol=2,loc='upper right')
fig.text(.5,-.05,'39,270 shared issue-room observations; six rooms in one building; selected models used past local labels.',ha='center',fontsize=9)
fig.savefig(p/'aalborg_external_v1.png',dpi=250,bbox_inches='tight');fig.savefig(p/'aalborg_external_v1.svg',bbox_inches='tight')
