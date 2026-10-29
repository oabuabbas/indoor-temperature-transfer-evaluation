from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'outputs';d=pd.read_csv(out/'bremen_nonlinear_benchmark_v1_rooms.csv')
p=d.pivot(index='sensor',columns='model',values='mae_c').sort_index();x=np.arange(len(p));w=.19
fig,ax=plt.subplots(figsize=(9.1,4.8),layout='constrained')
for offset,(col,label,color) in enumerate([('source_daily_ridge','Transferred ridge','#497ca6'),('rolling_selector','Rolling selector','#1b6e70'),('local_ridge_30d','Local ridge','#d1833f'),('local_hist_gradient_boosting','Local gradient boosting','#ae6565')]):
 bars=ax.bar(x+(offset-1.5)*w,p[col],w,label=label,color=color);ax.bar_label(bars,fmt='%.2f',padding=2,fontsize=7,rotation=90)
ax.set_xticks(x,p.index);ax.set_ylim(0,1.12);ax.set_ylabel('MAE (°C)');ax.set_title('Bremen room forecasts on identical observed support');ax.legend(frameon=False,ncol=2,loc='upper center')
fig.text(.5,-.05,'28,948 paired issue times; four correlated rooms; fixed local model settings.',ha='center',fontsize=9)
fig.savefig(out/'bremen_nonlinear_benchmark_v1.png',dpi=250,bbox_inches='tight');fig.savefig(out/'bremen_nonlinear_benchmark_v1.svg',bbox_inches='tight')
