"""Post-result descriptive paired difference by month with changing cohorts."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=pd.read_csv(OUT/'ideal_holdout_monthly_secondary.csv')
    t=pd.to_datetime(d.period_start,utc=True)
    labels=[f'{v.strftime("%Y-%m")} (n={n})' for v,n in zip(t,d.homes)]
    color=['#567f9e' if v<0 else '#b67549' for v in d.mean_diff_c]
    fig,ax=plt.subplots(figsize=(9,5.3),layout='constrained')
    bars=ax.barh(labels,d.mean_diff_c,color=color)
    ax.axvline(0,color='#343434',linewidth=.9);ax.invert_yaxis()
    ax.set_xlabel('Selector MAE − local ridge MAE (°C); negative favors selector')
    ax.set_title('IDEAL held-out homes: post-result monthly breakdown\nContributing homes change across months; no causal comparison',loc='left',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(min(d.mean_diff_c.min()*1.5,-.04),max(d.mean_diff_c.max()*1.5,.04))
    for b,v in zip(bars,d.mean_diff_c):
        ax.text(v+(.0007 if v>=0 else -.0007),b.get_y()+b.get_height()/2,f'{v:+.3f}',
                va='center',ha='left' if v>=0 else 'right',fontsize=9)
    fig.savefig(OUT/'ideal_holdout_monthly_secondary.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'ideal_holdout_monthly_secondary.svg',bbox_inches='tight')

if __name__=='__main__':main()
