"""Read-only independent verification of the supplied package and regenerated tables.
Usage: python independent_audit.py SOURCE_ROOT EXECUTION_ROOT OUTPUT_DIR
No provider calls. Does not import the supplied scoring/bootstrapping implementation.
"""
import sys,json,re,csv,hashlib,math
from pathlib import Path
import numpy as np
import pandas as pd
src,exe,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True,parents=True)
runs=json.loads((src/'config/run_manifest.json').read_text());rows=[];coverage=[]
for run in runs:
 records=[json.loads(l) for l in (src/run['raw_file']).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
 retained={}
 for line,r in enumerate(records,1):
  rep=int(r.get('replicate_id',r.get('replicate')));q=r.get('original_question_number',r.get('question_number',r.get('question_id',0)))
  q=int(str(q).lstrip('Qq')) if run['strategy']=='item-wise' else 0
  retained[rep,q]=(line,r)
 for (rep,q),(line,r) in retained.items():
  if isinstance(r.get('raw_tool_input'),dict):obj=r['raw_tool_input']
  else:
   text=next(r[k] for k in ['raw_output','raw_text','raw_content'] if isinstance(r.get(k),str))
   text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
   try:obj=json.loads(text)
   except json.JSONDecodeError:
    m=re.match(r'^\s*\{\s*"answer"\s*:\s*"([1-4]|blank)"\s*,\s*"rationale"\s*:',text)
    assert m,(run['condition_id'],line);obj={'answer':m[1]}
  answers=obj['answers'] if run['strategy']=='whole-exam' else [{'question_number':q,**obj}]
  if run['strategy']=='whole-exam':assert sorted(int(x['question_number']) for x in answers)==list(range(1,211))
  for ans in answers:
   v=ans['answer'];v='blank' if isinstance(v,list) or re.fullmatch(r'\s*[1-4](?:\s*[,;/]\s*[1-4])+\s*',str(v)) else str(v).lower().strip()
   if re.fullmatch(r'[1-4](?:\.0+)?',v):v=str(int(float(v)))
   rows.append(dict(condition_id=run['condition_id'],model=run['model'],exam=run['exam'],strategy=run['strategy'],replicate=rep,question_number=int(ans['question_number']),answer=v,raw_line=line))
 coverage.append(dict(condition_id=run['condition_id'],raw_records=len(records),retained_requests=len(retained),returned_models='|'.join(sorted({str(r.get('model_returned')) for r in records}))))
a=pd.DataFrame(rows);keys=[]
for exam in ['BIR','FIR']:
 k=pd.read_csv(src/f'source_data/inputs/{exam.lower()}2024_answer_key_210.csv',keep_default_na=False)
 q=pd.read_csv(src/f'source_data/inputs/{exam.lower()}2024_questions_210.csv',keep_default_na=False)
 k=k.sort_values('original_question_number');k=k[k.correct_answer.astype(str).str.match(r'^[1-4](?:\.0+)?$')].head(200)
 for _,row in k.iterrows():
  qn=int(row.original_question_number);has=str(q.loc[q.original_question_number==qn,'has_image'].iloc[0]).lower()=='true'
  keys.append(dict(exam=exam,question_number=qn,key=str(int(float(row.correct_answer))),has_image=has))
a=a.merge(pd.DataFrame(keys),on=['exam','question_number'],validate='many_to_one');a['correct']=a.answer==a.key;a['blank']=a.answer=='blank';a['incorrect']=~a.correct & ~a.blank;a['score']=np.where(a.correct,1,np.where(a.blank,0,-1/3));assert len(a)==9600
b=pd.read_csv(exe/'data/official_200_scored_items.csv',keep_default_na=False);c=a.merge(b,on=['condition_id','replicate','question_number'],suffixes=('_ind','_pkg'),validate='one_to_one')
assert (c.answer==c.rescored_answer).all();assert (c.correct==c.is_correct).all();assert np.allclose(c.score_ind,c.score_pkg)
boot=[]
for subset in ['official_200','fir_text_only_180','fir_image_20']:
 sub=a if subset=='official_200' else a[(a.exam=='FIR')&(a.has_image==(subset=='fir_image_20'))]
 expected=pd.read_csv(exe/f'results/tables/bootstrap_{subset}.csv')
 for (model,exam,strategy),g in sub.groupby(['model','exam','strategy']):
  p=g.pivot(index='question_number',columns='replicate',values='correct').astype(float).sort_index();s=g.pivot(index='question_number',columns='replicate',values='score').sort_index();assert list(p.columns)==[1,2]
  rng=np.random.default_rng(2026);idx=rng.integers(0,len(p),size=(10000,len(p)))
  acc=p.to_numpy()[idx,:].mean(axis=(1,2))*100;sc=s.to_numpy()[idx,:].sum(axis=1).mean(axis=1)
  lo,hi=np.percentile(acc,[2.5,97.5]);sl,sh=np.percentile(sc,[2.5,97.5]);v=expected[(expected.model==model)&(expected.exam==exam)&(expected.strategy==strategy)].iloc[0]
  assert np.allclose([lo,hi,sl,sh],[v.acc_ci_low_pct,v.acc_ci_high_pct,v.score_ci_low,v.score_ci_high],atol=1e-10)
  boot.append(dict(model=model,exam=exam,strategy=strategy,subset=subset,acc_low=lo,acc_high=hi,score_low=sl,score_high=sh,match=True))
pd.DataFrame(boot).to_csv(out/'independent_bootstrap_check.csv',index=False);pd.DataFrame(coverage).to_csv(out/'run_coverage.csv',index=False)
a.groupby(['condition_id','exam','strategy','replicate']).agg(n=('answer','size'),correct=('correct','sum'),incorrect=('incorrect','sum'),blank=('blank','sum'),score=('score','sum')).to_csv(out/'independent_run_scores.csv')
print('PASS: independently extracted/scored 9600 responses; explicit 2-replicate cluster bootstrap matched all 48 model/exam/strategy/subset rows.')
