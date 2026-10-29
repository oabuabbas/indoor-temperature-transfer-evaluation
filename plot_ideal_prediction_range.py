"""Plot exploratory paired-issue sensitivity; targets may differ by filter."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=pd.read_csv(OUT/'ideal_prediction_range_summary.csv').pivot(index='model',columns='filter',values='mae_c')
    order=['persistence','daily_seasonal','source_daily_ridge','local_ridge_30d','rolling_selector']
    labels=['Persistence','Previous day','Transferred ridge','Local ridge','Rolling selector']
    y=np.arange(len(order));fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.barh(y-.18,d.loc[order,'10_45'],height=.34,label='10–45°C',color='#557f9e')
    ax.barh(y+.18,d.loc[order,'15_40'],height=.34,label='15–40°C',color='#be8754')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,1.25)
    ax.set_xlabel('Mean absolute error (°C), equal weight per home')
    ax.set_title('IDEAL development: common issue times, 6-hour forecast\n49 homes; aggregated targets may differ between filters',loc='left',fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right',frameon=False)
    fig.savefig(OUT/'ideal_prediction_range_comparison.png',dpi=180,bbox_inches='tight')

if __name__=='__main__':main()
