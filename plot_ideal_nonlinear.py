"""Exploratory development comparison including nonlinear local baseline."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=pd.read_csv(OUT/'ideal_dev_nonlinear_summary.csv').query('min_months == 0').set_index('model')
    order=['persistence','daily_seasonal','local_histgb_30d','source_daily_ridge','local_ridge_30d','rolling_selector']
    labels=['Persistence','Previous day','Local gradient boosting','Transferred ridge','Local ridge','Rolling selector']
    vals=[float(d.loc[k,'mae_c']) for k in order]
    fig,ax=plt.subplots(figsize=(8.8,5),layout='constrained')
    bars=ax.barh(labels,vals,color=['#909eaa','#909eaa','#a78d74','#5c83a2','#6ca593','#b77442'])
    ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.22);ax.spines[['top','right']].set_visible(False)
    ax.set_title('IDEAL development: 6-hour forecast\n49 homes; 350 home-months; not the held-out test',loc='left',fontsize=11)
    ax.set_xlabel('Mean absolute error (°C), equal weight per home')
    for b,v in zip(bars,vals):ax.text(v+.015,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center')
    fig.savefig(OUT/'ideal_dev_nonlinear_comparison.png',dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
