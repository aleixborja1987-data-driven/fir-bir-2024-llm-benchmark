"""Regenerate paired descriptive comparisons and explicitly defined latency summaries."""
import math,json,platform
from pathlib import Path
import numpy as np
import pandas as pd

def sign_test(d):
    d=np.asarray(d);positive=int((d>1e-10).sum());negative=int((d< -1e-10).sum());n=positive+negative
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(positive,negative)+1))/2**n) if n else 1.
    return {'n_positive':positive,'n_negative':negative,'n_ties':len(d)-n,'two_sided_exact_p':p}

def build(root):
    tables=root/'results/tables';eff=pd.read_csv(tables/'efficiency_by_condition.csv');scores=pd.read_csv(tables/'condition_summary.csv');rows=[]
    for exam in ['FIR','BIR']:
        for metric,df in [('final_attempt_latency_seconds',eff),('logged_latency_seconds',eff),('total_tokens',eff),('mean_accuracy_pct',scores),('mean_score',scores)]:
            a=df[df.exam==exam].pivot(index='model',columns='strategy',values=metric)
            diff=a['whole-exam']-a['item-wise'];red=100*(1-a['whole-exam']/a['item-wise'])
            rows.append({'exam':exam,'metric':metric,'item_wise_mean':a['item-wise'].mean(),'whole_exam_mean':a['whole-exam'].mean(),'median_percent_reduction':red.median(),'minimum_percent_reduction':red.min(),'maximum_percent_reduction':red.max(),'difference_direction':'whole-exam minus item-wise',**sign_test(diff)})
    pd.DataFrame(rows).to_csv(tables/'paired_sign_tests_and_reductions.csv',index=False)
    calls=pd.read_csv(root/'data/request_records.csv');calls=calls[calls.is_final_record & calls.strategy.eq('item-wise')];rows=[]
    for keys,g in calls.groupby(['condition_id','model','exam']):
        a=pd.to_numeric(g.final_attempt_latency_seconds,errors='coerce').dropna()
        rows.append(dict(zip(['condition_id','model','exam'],keys))|{'n_requests':len(a),'pooling':'420 final requests; two replicates pooled; includes reserves','q1_seconds':a.quantile(.25),'median_seconds':a.median(),'q3_seconds':a.quantile(.75),'iqr_seconds':a.quantile(.75)-a.quantile(.25)})
    pd.DataFrame(rows).to_csv(tables/'item_latency_distribution.csv',index=False)
    import matplotlib
    (root/'provenance/analysis_environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'scope':'Current offline reanalysis environment; see original_environment.json for metadata recovered from archived venv.'},indent=2)+'\n')
