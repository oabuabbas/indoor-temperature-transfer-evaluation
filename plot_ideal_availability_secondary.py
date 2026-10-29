"""Plot observed input-readiness proxy, not real-world uptime or accuracy."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_nest import OUT

def main():
    d=json.loads((OUT/'ideal_holdout_availability_summary.json').read_text(encoding='utf-8'))
    rates=d['span_rates'];order=['selector_ready','source_ready','daily_ready','persistence_ready','fallback_ready']
    labels=['Rolling selector','Transferred ridge','Previous day','Persistence','Hypothetical fallback']
    vals=[100*rates[k] for k in order]
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    bars=ax.barh(labels,vals,color=['#b77442','#5c83a2','#909eaa','#909eaa','#6ca593'])
    ax.invert_yaxis();ax.set_xlim(0,103);ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Opportunities with required inputs available (%)')
    ax.set_title('IDEAL held-out record: retrospective input-readiness audit\n369,479 opportunities within recorded spans; fallback is post-hoc and not accuracy-tested',loc='left',fontsize=11)
    for bar,value in zip(bars,vals):ax.text(value+1,bar.get_y()+bar.get_height()/2,f'{value:.1f}%',va='center')
    fig.savefig(OUT/'ideal_holdout_availability_secondary.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'ideal_holdout_availability_secondary.svg',bbox_inches='tight')

if __name__=='__main__':main()
