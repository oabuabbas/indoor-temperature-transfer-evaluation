"""Descriptive plot of exploratory development-only MAE, no uncertainty claim."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=pd.read_csv(OUT/'ideal_dev_local_summary.csv').set_index('model')
    order=['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector']
    labels=['Persistence','Previous day','Transferred ridge','Local ridge (30 d)','Rolling selector']
    values=[float(d.loc[x,'mae_c']) for x in order]
    fig,ax=plt.subplots(figsize=(8.6,4.7),layout='constrained')
    bars=ax.barh(labels,values,color=['#8997a4','#8997a4','#5c83a2','#6ca593','#b77442'])
    ax.invert_yaxis();ax.set_xlabel('Mean absolute error (°C), equal weight per home')
    ax.set_title('IDEAL: exploratory development homes, 6-hour forecast\n49 homes; 350 eligible home-months; not the held-out test',loc='left',fontsize=11)
    ax.set_xlim(0,max(values)*1.22);ax.spines[['top','right']].set_visible(False)
    for bar,value in zip(bars,values):ax.text(value+.015,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center')
    fig.savefig(OUT/'ideal_development_v1_comparison.png',dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
