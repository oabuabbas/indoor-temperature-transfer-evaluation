from pathlib import Path
import json
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/'outputs';s=json.loads((p/'ideal_nonlinear_holdout_secondary_v1_summary.json').read_text())
labels={'rolling_selector':'Rolling selector','source_daily_ridge':'Transferred ridge','local_ridge_30d':'Local ridge','selected_boosting':'Development-selected boosting','local_histgb_30d':'Earlier fixed boosting','daily_seasonal':'Previous day','persistence':'Persistence'}
d=sorted([(v,labels[k],k) for k,v in s['equal_home_mae_c'].items()]);fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
bars=ax.barh([x[1] for x in d],[x[0] for x in d],color=['#b36d6d' if x[2]=='selected_boosting' else '#547d99' for x in d]);ax.invert_yaxis();ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax.set_xlim(0,1.28);ax.set_xlabel('Equal-home mean MAE (°C)');ax.set_title('IDEAL: later-period secondary comparison')
fig.text(.5,-.05,'72 homes, Feb–Jun 2018; shared eligible windows; prior holdout outcomes were already known.',ha='center',fontsize=9)
fig.savefig(p/'ideal_nonlinear_holdout_secondary_v1.png',dpi=250,bbox_inches='tight');fig.savefig(p/'ideal_nonlinear_holdout_secondary_v1.svg',bbox_inches='tight')
