"""Descriptive display of locked IDEAL holdout mean errors."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=json.loads((OUT/'ideal_holdout_v1_summary.json').read_text(encoding='utf-8'))['mean_mae_c']
    order=['persistence','daily_seasonal','local_histgb_30d','local_ridge_30d','source_daily_ridge','rolling_selector']
    labels=['Persistence','Previous day','Local gradient boosting','Local ridge','Transferred ridge','Rolling selector']
    vals=[d[k] for k in order]
    fig,ax=plt.subplots(figsize=(8.8,5),layout='constrained')
    bars=ax.barh(labels,vals,color=['#909eaa','#909eaa','#a78d74','#6ca593','#5c83a2','#b77442'])
    ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.22);ax.spines[['top','right']].set_visible(False)
    ax.set_title('IDEAL held-out homes: 6-hour forecast\n73 homes; 477 eligible home-months; one geographic region',loc='left',fontsize=11)
    ax.set_xlabel('Mean absolute error (°C), equal weight per home')
    for b,v in zip(bars,vals):ax.text(v+.015,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center')
    fig.savefig(OUT/'ideal_holdout_v1_comparison.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'ideal_holdout_v1_comparison.svg',bbox_inches='tight')

if __name__=='__main__':main()
