"""Run saved-artifact checks; create traceable evidence ledger without re-estimating models."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pandas as pd
p=Path(__file__).resolve().parent;o=p/'outputs'
def read(n):return json.loads((o/(n+'.json')).read_text(encoding='utf8'))
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
checks=['verify_paper_outputs.py','verify_bremen_fallback.py','verify_bremen_outage_stress.py','verify_bremen_joint_missing_v1.py','verify_bolzano_external_v1.py','verify_bolzano_2026_benchmark_v1.py','verify_bremen_nonlinear_benchmark_v1.py','verify_ideal_nonlinear_tuning_v1.py','verify_aalborg_external_v1.py']
env=os.environ.copy();env['OMP_NUM_THREADS']='1';results=[]
for script in checks:
 x=subprocess.run([sys.executable,script],cwd=p,env=env,capture_output=True,text=True,encoding='utf8',errors='replace');results.append({'script':script,'returncode':x.returncode,'output':(x.stdout+'\n'+x.stderr).strip()});print(script,'PASS' if x.returncode==0 else 'FAIL',flush=True)
unit=subprocess.run([sys.executable,'-m','unittest','discover','-p','test_*.py'],cwd=p,env=env,capture_output=True,text=True,encoding='utf8',errors='replace');results.append({'script':'unittest discover','returncode':unit.returncode,'output':(unit.stdout+'\n'+unit.stderr).strip()});print('unittest',unit.returncode,flush=True)
if any(x['returncode']!=0 for x in results):raise ValueError('Evidence verification failed; inspect output')
a=read('ideal_holdout_v1_summary');z=pd.read_csv(o/'ideal_holdout_v1_home_months.csv');n=int(z[z.model.eq('rolling_selector')].test_n.sum())
b=read('bremen_fallback_v1_summary');bp=b['paired_fallback_vs_persistence']
c=read('bolzano_external_v1_summary')['results']['nominal_mean'];d=read('bolzano_2026_benchmark_v1_summary')['results']['nominal_mean'];e=read('aalborg_external_v1_summary')
rows=[{'study':'IDEAL original holdout','support':'common original home-months','n':n,'units':'73 houses','selector_mae':a['mean_mae_c']['rolling_selector'],'comparator':'local ridge','comparator_mae':a['mean_mae_c']['local_ridge_30d'],'weight':'equal home','status':'local protocol before original holdout errors; earlier exploratory work known'},
{'study':'Bremen external','support':'fallback/persistence paired','n':bp['common_issue_times'],'units':'4 rooms, one dataset','selector_mae':bp['equal_room_fallback_mae_c'],'comparator':'persistence','comparator_mae':bp['equal_room_persistence_mae_c'],'weight':'equal room','status':'protocol before Bremen errors; policy post IDEAL'},
{'study':'Bolzano 2024-25','support':'fallback/persistence paired; nominal clock','n':c['fallback_vs_persistence']['n'],'units':'1 room','selector_mae':c['fallback_vs_persistence']['fallback_mae_c'],'comparator':'persistence','comparator_mae':c['fallback_vs_persistence']['persistence_mae_c'],'weight':'single room','status':'protocol before Bolzano errors; prior datasets known'},
{'study':'Bolzano Jan-Apr 2026','support':'all 7 methods and target; nominal clock','n':d['common_n'],'units':'1 room','selector_mae':next(x['mae_c'] for x in d['common_metrics'] if x['model']=='rolling_selector'),'comparator':'source ridge','comparator_mae':next(x['mae_c'] for x in d['common_metrics'] if x['model']=='source_daily_ridge'),'weight':'single room','status':'protocol after 2024-25 results; before 2026 errors'},
{'study':'Aalborg','support':'all 8 methods and target','n':e['common_issue_times'],'units':'6 rooms, one building','selector_mae':next(x['equal_room_mae_c'] for x in e['equal_room_metrics'] if x['model']=='rolling_selector'),'comparator':'source ridge','comparator_mae':next(x['equal_room_mae_c'] for x in e['equal_room_metrics'] if x['model']=='source_daily_ridge'),'weight':'equal room','status':'protocol before Aalborg errors; other datasets known'}]
files=['ideal_holdout_v1_summary.json','bremen_fallback_v1_summary.json','bolzano_external_v1_summary.json','bolzano_2026_benchmark_v1_summary.json','aalborg_external_v1_summary.json']
result={'checks':results,'evidence':rows,'input_sha256':{x:digest(o/x) for x in files},'warning':'Not independent model refitting, not publication confirmation; comparisons have different populations, periods, preprocessing, weights and support.'}
(o/'submission_evidence_audit_v1.json').write_text(json.dumps(result,indent=2),encoding='utf8');pd.DataFrame(rows).to_csv(o/'submission_evidence_ledger_v1.csv',index=False,float_format='%.10f')
print('Evidence ledger created.',flush=True)
if __name__=='__main__':pass
