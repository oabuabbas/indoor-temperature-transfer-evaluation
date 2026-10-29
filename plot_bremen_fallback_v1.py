from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent / "outputs"
d=pd.read_csv(out / "bremen_fallback_v1_common.csv").sort_values("sensor")
x=np.arange(len(d)); w=.34
fig,ax=plt.subplots(figsize=(8.4,4.8),layout="constrained")
a=ax.bar(x-w/2,d.fallback_mae_c,w,label="Fallback policy",color="#1b6e70")
b=ax.bar(x+w/2,d.persistence_mae_c,w,label="Persistence",color="#d1833f")
for bars in (a,b): ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=9)
ax.set_xticks(x,d.sensor);ax.set_ylabel("Mean absolute error (°C)")
ax.set_ylim(0, max(d.persistence_mae_c)*1.18)
ax.set_title("Bremen rooms: 6-hour forecasts on paired observed targets")
ax.legend(frameon=False,ncol=2,loc="upper right")
ax.text(.5,-.18,"29,022 paired predictions; 4 rooms in one dataset, not independent sites",transform=ax.transAxes,ha="center",fontsize=9)
fig.savefig(out/"bremen_fallback_paired.png",dpi=250,bbox_inches="tight")
fig.savefig(out/"bremen_fallback_paired.svg",bbox_inches="tight")
