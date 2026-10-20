#!/usr/bin/env python3
"""Offline reconstruction of the supplied FIR/BIR API logs (no model/API calls).

This adapter was newly written for the 2026-10-08 reproducibility package. Original
acquisition and analysis scripts are separately retained in source_scripts/.
The answer key is joined only after final-answer extraction. Earlier incorrect
CSV correctness flags are never used to change a model answer.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def load_json(path:Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def first(obj:dict,*keys,default=None):
    for key in keys:
        if key in obj and obj[key] is not None:return obj[key]
    return default


def as_bool(value)->bool:
    if isinstance(value,bool):return value
    if value is None:return False
    if isinstance(value,(int,float)):return bool(value) if not pd.isna(value) else False
    return str(value).strip().lower() in ('true','1','yes')


def norm_answer(value):
    """Value normalization only; no official key is consulted."""
    if value is None:return None
    if isinstance(value,list):
        return None  # Explicit multiple choices are handled separately.
    text=str(value).strip().lower()
    if re.fullmatch(r'[1-4](?:\.0+)?',text):return str(int(float(text)))
    if text in ('blank','blanco'):return 'blank'
    return None


def answer_object(record:dict):
    """Read emitted structured output; tolerate only an unambiguous leading answer.

    In the one truncated JSON response, auxiliary rationale text is incomplete.
    Recovery extracts the already-emitted leading answer field, not the rationale.
    """
    tool=record.get('raw_tool_input')
    if isinstance(tool,dict):return tool,json.dumps(tool,ensure_ascii=False),'tool_json'
    text=first(record,'raw_output','raw_text','raw_content',default='')
    if not isinstance(text,str):raise ValueError('Raw output is not a string')
    clean=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip(),flags=re.I)
    try:return json.loads(clean),text,'json'
    except json.JSONDecodeError:
        match=re.match(r'^\s*\{\s*"answer"\s*:\s*"([1-4]|blank)"\s*,\s*"rationale"\s*:',clean)
        if match:return {'answer':match.group(1)},text,'unique_leading_answer_recovery'
        raise ValueError('Output cannot be parsed without an unsupported recovery')


def selected_answer(value,record,parse_mode):
    if isinstance(value,list) or (isinstance(value,str) and re.fullmatch(r'\s*[1-4](?:\s*[,;/]\s*[1-4])+\s*',value)):
        return 'blank','multiple_options_to_blank'
    ans=norm_answer(value)
    if ans is None:return None,'missing_or_invalid'
    if parse_mode=='unique_leading_answer_recovery' or record.get('parse_error') or as_bool(record.get('rationale_length_violation_recovered')):
        return ans,'unique_answer_recovery'
    return ans,'as_emitted'


def read_inputs(root:Path):
    parts=[]
    for exam in ['BIR','FIR']:
        prefix=exam.lower()+'2024'
        key=pd.read_csv(root/'source_data/inputs'/f'{prefix}_answer_key_210.csv',keep_default_na=False)
        questions=pd.read_csv(root/'source_data/inputs'/f'{prefix}_questions_210.csv',keep_default_na=False)
        kn='original_question_number'
        key['question_number']=pd.to_numeric(key[kn]).astype(int)
        questions['question_number']=pd.to_numeric(questions[kn]).astype(int)
        if sorted(key.question_number)!=list(range(1,211)) or sorted(questions.question_number)!=list(range(1,211)):raise ValueError('Inputs must contain each question ID 1..210 exactly once')
        official_col='has_official_answer' if 'has_official_answer' in key else 'official_answer_available'
        key['has_official_answer']=key[official_col].map(as_bool)
        key['correct_answer']=key['correct_answer'].map(norm_answer)
        if (key.loc[key.has_official_answer,'correct_answer'].isna()).any():raise ValueError('Invalid official answer')
        if (key.loc[~key.has_official_answer,'correct_answer'].notna()).any():raise ValueError('Contradictory official-answer flag')
        image_map=dict(zip(questions.question_number,questions.has_image.map(as_bool)))
        key['has_image']=key.question_number.map(image_map)
        key['exam']=exam
        key=key.sort_values('question_number')
        # Selects the first 200 valid keyed questions, including replacement reserves.
        selected=key.loc[key.has_official_answer,'question_number'].head(200).tolist()
        if len(selected)!=200:raise ValueError('Expected 200 valid scored items')
        key['included_in_official_200']=key.question_number.isin(selected)
        key['official200_position']=key.question_number.map({q:i+1 for i,q in enumerate(selected)})
        key['is_reserve_original']=key.question_number>200
        key['selection_reason']=np.where(~key.has_official_answer,'no_answer_in_supplied_key',np.where(key.included_in_official_200,np.where(key.is_reserve_original,'used_reserve','scored_main'),'unused_reserve'))
        parts.append(key[['exam','question_number','correct_answer','has_official_answer','has_image','is_reserve_original','included_in_official_200','official200_position','selection_reason']])
    return pd.concat(parts,ignore_index=True)


def tokens(record:dict):
    usage=first(record,'usage','usage_raw',default={}) or {}
    if not isinstance(usage,dict):usage={}
    inp=first(record,'input_tokens','prompt_tokens',default=first(usage,'input_tokens','prompt_tokens'))
    out=first(record,'output_tokens','completion_tokens',default=first(usage,'output_tokens','completion_tokens'))
    total=first(record,'total_tokens',default=usage.get('total_tokens'))
    if total is None and inp is not None and out is not None:total=inp+out
    reasoning=first(record,'reasoning_tokens','thinking_tokens')
    if reasoning is None:
        for k in ['output_tokens_details','completion_tokens_details']:
            details=usage.get(k) or {}
            reasoning=first(details,'reasoning_tokens','thinking_tokens')
            if reasoning is not None:break
    cached=first(record,'cached_input_tokens','cached_tokens','prompt_cache_hit_tokens','cache_read_input_tokens')
    if cached is None:
        cached=first(usage,'cache_read_input_tokens','prompt_cache_hit_tokens','cached_tokens')
        if cached is None:cached=(usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'input_tokens':inp,'output_tokens':out,'total_tokens':total,'reasoning_tokens':reasoning,'cached_input_tokens':cached,'cache_creation_input_tokens':first(record,'cache_creation_input_tokens',default=usage.get('cache_creation_input_tokens'))}


def replicate(record:dict)->int:
    return int(first(record,'replicate_id','replicate'))


def question(record:dict)->int:
    q=first(record,'original_question_number','question_number','question_id')
    return int(str(q).lstrip('Qq'))


def metric_row(record:dict,run:dict,line:int,is_final:bool):
    # Preserve distinctions between final-attempt and including-retry durations.
    final=first(record,'final_attempt_latency_seconds','attempt_latency_seconds','latency_seconds','elapsed_seconds')
    inclusive=first(record,'latency_seconds_including_retries','elapsed_seconds_including_retries')
    if inclusive is None and run['model_folder']=='Kimi-k2.6' and run['exam']=='BIR' and run['strategy']=='item-wise':
        inclusive=record.get('latency_seconds')
    logged=first(record,'latency_seconds','elapsed_seconds','final_attempt_latency_seconds')
    attempts=first(record,'attempts_used',default=1)
    return {'condition_id':run['condition_id'],'model':run['model'],'exam':run['exam'],'strategy':run['strategy'],'replicate':replicate(record),'question_number':question(record) if run['strategy']=='item-wise' else '',
            'raw_file':run['raw_file'],'raw_line':line,'is_final_record':is_final,'attempt_number':record.get('attempt'),'attempts_used_counter':attempts,
            'final_attempt_latency_seconds':final,'elapsed_including_retries_seconds':inclusive,'logged_latency_seconds':logged,
            'request_time_utc':first(record,'request_time_utc','attempt_started_utc'),'response_time_utc':first(record,'response_time_utc','attempt_completed_utc'),
            'parse_error':record.get('parse_error'),'api_error':first(record,'api_error','exception_type','error_type'),
            'protocol_violation':record.get('protocol_violation'),'finish_reason':first(record,'finish_reason','stop_reason'),
            'model_returned':record.get('model_returned'),**tokens(record)}


def extract(root:Path,runs:list):
    rows=[];attempts=[];audits=[];runinfo=[]
    for run in runs:
        raw=[(i,json.loads(s)) for i,s in enumerate((root/run['raw_file']).read_text(encoding='utf-8-sig').splitlines(),1) if s.strip()]
        retained={}
        for line,r in raw:
            k=(replicate(r),question(r) if run['strategy']=='item-wise' else 0)
            retained[k]=(line,r)  # A per-attempt log's last record is its retained attempt.
        expected=420 if run['strategy']=='item-wise' else 2
        expected_ids={(rep,q) for rep in (1,2) for q in (range(1,211) if run['strategy']=='item-wise' else [0])}
        if set(retained)!=expected_ids:raise ValueError(f"{run['condition_id']}: unexpected or missing replicate/question IDs")
        final_lines={v[0] for v in retained.values()}
        for line,r in raw:attempts.append(metric_row(r,run,line,line in final_lines))
        for (rep,q),(line,r) in sorted(retained.items()):
            obj,text,mode=answer_object(r)
            if run['strategy']=='whole-exam':
                answers=obj.get('answers') if isinstance(obj,dict) else obj
                if not isinstance(answers,list):raise ValueError('No answer list in whole-exam response')
                qs=[question(x) for x in answers]
                if sorted(qs)!=list(range(1,211)):raise ValueError('Whole-exam answer list must contain 1..210 exactly once')
            else:answers=[{'question_number':q,**obj}]
            for answer in answers:
                qn=question(answer);value=answer.get('answer');ans,rule=selected_answer(value,r,mode)
                # For whole-exam JSON errors are tracked at call level; any parsed valid answer is direct.
                initial=first(r,'parsed_answer','model_answer','answer') if run['strategy']=='item-wise' else value
                record={'condition_id':run['condition_id'],'model':run['model'],'model_requested':run['model_id'],'model_returned':r.get('model_returned'),'exam':run['exam'],'strategy':run['strategy'],'replicate':rep,'question_number':qn,
                        'raw_answer_value':json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value,'initial_parsed_answer':initial,'rescored_answer':ans,'model_answer':ans,'recode_rule':rule,
                        'raw_file':run['raw_file'],'raw_line':line,'rationale':answer.get('rationale',''),'source_parse_error':r.get('parse_error'),'output_sha256':hashlib.sha256(text.encode()).hexdigest()}
                rows.append(record)
                if rule in ['unique_answer_recovery','multiple_options_to_blank']:
                    audits.append({**record,'raw_output':text,'recovery_mode':mode})
        meta=load_json(root/run['metadata_file'])
        runinfo.append({**run,'raw_records':len(raw),'retained_request_records':len(retained),
                        'minimum_attempts_from_counters_or_rows':max(len(raw),sum(int(first(r,'attempts_used',default=1)) for _,r in retained.values())),
                        'logged_prespecified_complete':meta.get('definitive_run_completed_as_prespecified','not recorded'),
                        'generation_parameters_json':json.dumps(meta.get('generation_parameters',{k:meta[k] for k in ['temperature','top_p','max_tokens','thinking','stream'] if k in meta}),ensure_ascii=False),
                        'created_utc':first(meta,'created_utc','created_at_utc'),'completed_utc':meta.get('completed_utc')})
    return pd.DataFrame(rows),pd.DataFrame(audits),pd.DataFrame(attempts),pd.DataFrame(runinfo)


def rescore(df:pd.DataFrame,key:pd.DataFrame):
    out=df.merge(key,on=['exam','question_number'],validate='many_to_one')
    def calc(r):
        if not r.has_official_answer:return 'not_evaluable',False,False,np.nan
        ans=r.rescored_answer
        if ans=='blank':return 'blank',False,True,0.
        if ans not in ['1','2','3','4']:return 'missing_or_invalid',False,False,0.
        ok=ans==r.correct_answer
        return ('correct' if ok else 'incorrect'),ok,False,(1. if ok else -1/3)
    status=pd.DataFrame([calc(r) for r in out.itertuples()],columns=['rescored_status','is_correct','is_blank','score'],index=out.index)
    return pd.concat([out,status],axis=1).sort_values(['condition_id','replicate','question_number'])


def aggregate(df:pd.DataFrame):
    groups=['condition_id','model','exam','strategy','replicate']
    rep=df.groupby(groups,sort=True).agg(n_items=('question_number','size'),correct=('is_correct','sum'),score=('score','sum'),blank=('is_blank','sum'),incorrect=('rescored_status',lambda x:(x=='incorrect').sum()),missing_or_invalid=('rescored_status',lambda x:(x=='missing_or_invalid').sum())).reset_index()
    rep['accuracy_pct']=100*rep.correct/rep.n_items
    mean=rep.groupby(groups[:-1],sort=True).agg(n_replicates=('replicate','size'),mean_correct=('correct','mean'),mean_accuracy_pct=('accuracy_pct','mean'),mean_score=('score','mean'),mean_blank=('blank','mean'),mean_incorrect=('incorrect','mean'),accuracy_sd=('accuracy_pct','std')).reset_index()
    return rep,mean


def agreement(df:pd.DataFrame):
    rep_rows=[];pair_rows=[]
    for (cid,model,exam,strategy),g in df.groupby(['condition_id','model','exam','strategy'],sort=True):
        a=g.pivot(index='question_number',columns='replicate',values='rescored_answer').fillna('INVALID')
        rep_rows.append({'condition_id':cid,'model':model,'exam':exam,'strategy':strategy,'n_items':len(a),'n_discordant':int((a[1]!=a[2]).sum()),'exact_answer_agreement_pct':100*(a[1]==a[2]).mean()})
    for (model,exam),g in df.groupby(['model','exam'],sort=True):
        a=g.pivot(index=['question_number','replicate'],columns='strategy',values='is_correct').astype(int)
        observed=(a['item-wise']==a['whole-exam']).mean()
        pi=a.to_numpy().mean();expected=2*pi*(1-pi);ac1=(observed-expected)/(1-expected)
        pair_rows.append({'model':model,'exam':exam,'n_question_replicate_pairs':len(a),'correctness_agreement_pct':100*observed,'gwet_ac1_binary_two_raters':ac1,'interpretation':'descriptive; paired replicates retained; no superiority testing'})
    return pd.DataFrame(rep_rows),pd.DataFrame(pair_rows)


def image_audit(root:Path):
    rows=[]
    key=read_inputs(root)
    image_ids=set(key.loc[(key.exam=='FIR') & key.has_image,'question_number'])
    if len(image_ids)!=20:raise ValueError('Expected 20 FIR image inputs')
    source=pd.read_csv(root/'provenance/source_inventory.csv',keep_default_na=False)
    for q in image_ids:
        rel=f'source_data/inputs/images/fir/Q{q:03d}.png'
        p=root/rel
        expected_source=source.loc[source.package_path==rel,'public_sha256'].tolist()
        if len(expected_source)!=1 or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected_source[0]:
            raise ValueError(f'Missing or changed original image: {rel}')
    for run in load_json(root/'config/run_manifest.json'):
        if run['exam']!='FIR' or run['model_folder']=='DeepSeekV4Pro':continue
        raw=[json.loads(l) for l in (root/run['raw_file']).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        expected={}
        for r in raw:
            q=first(r,'original_question_number','question_id')
            if q is not None and r.get('image_sha256'):
                expected[int(str(q).lstrip('Qq'))]=r['image_sha256']
            if q is not None and r.get('image_sha256s'):
                vals=r['image_sha256s'];vals=json.loads(vals) if isinstance(vals,str) and vals.startswith('[') else vals
                if isinstance(vals,list) and len(vals)==1:expected[int(str(q).lstrip('Qq'))]=vals[0]
        manifest=root/Path(run['raw_file']).parent/'image_manifest.csv'
        if manifest.exists():
            with manifest.open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    q=first(row,'original_question_number','question_number','question_id')
                    h=first(row,'image_sha256','sha256')
                    if q and h:expected[int(float(str(q).lstrip('Qq')))]=h
        if set(expected) and set(expected)!=image_ids:raise ValueError(f"Incomplete logged image hash coverage: {run['condition_id']}")
        if not expected and run['condition_id']!='kimi_fir_item_wise':raise ValueError(f"Unexpected absent image hashes: {run['condition_id']}")
        for q in sorted(image_ids):
            h=expected.get(q,'')
            p=root/'source_data/inputs/images/fir'/f'Q{q:03d}.png'
            actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''
            rows.append({'condition_id':run['condition_id'],'question_number':q,'file':str(p.relative_to(root)),'logged_sha256':h,'provided_sha256':actual,'matches':str(h).lower()==actual if h else None,'evidence':'logged_hash' if h else 'original_file_verified; request_hash_not_logged'})
    out=pd.DataFrame(rows)
    if len(out)!=200 or out.condition_id.nunique()!=10:raise ValueError('Incomplete multimodal condition coverage')
    logged=out[out.logged_sha256!='']
    if len(logged)!=180 or not logged.matches.eq(True).all():raise ValueError('Logged image hash mismatch or missing coverage')
    return out


def efficiency(attempts:pd.DataFrame,runs:pd.DataFrame,scored:pd.DataFrame,root:Path):
    final=attempts[attempts.is_final_record].copy()
    numeric=['input_tokens','output_tokens','total_tokens','reasoning_tokens','cached_input_tokens','cache_creation_input_tokens','final_attempt_latency_seconds','elapsed_including_retries_seconds','logged_latency_seconds']
    for c in numeric:final[c]=pd.to_numeric(final[c],errors='coerce')
    group=['condition_id','model','exam','strategy','replicate']
    result=final.groupby(group)[numeric].sum(min_count=1).reset_index()
    complete=final.groupby(group)[['input_tokens','output_tokens','reasoning_tokens']].count().reset_index().rename(columns={x:x+'_n_observed' for x in ['input_tokens','output_tokens','reasoning_tokens']})
    result=result.merge(complete,on=group)
    result['retained_calls']=result.strategy.map({'item-wise':210,'whole-exam':1})
    result['total_tokens_per_scored_question']=result.total_tokens/200
    result['final_attempt_latency_per_scored_question']=result.final_attempt_latency_seconds/200
    result['output_tokens_per_end_to_end_second']=result.output_tokens/result.final_attempt_latency_seconds
    counts=scored.groupby(['condition_id','replicate']).agg(correct=('is_correct','sum'),exam_score=('score','sum')).reset_index()
    result=result.merge(counts,on=['condition_id','replicate'])
    result['total_tokens_per_correct_answer']=result.total_tokens/result.correct
    result['total_tokens_per_exam_point']=result.total_tokens/result.exam_score
    # Missing dated unit prices must not be inferred from rounded cost totals.
    rates=pd.read_csv(root/'config/api_prices.csv',keep_default_na=False)
    prices={r.model:(r.input_usd_per_million,r.output_usd_per_million,r.status) for r in rates.itertuples()}
    costs=[]
    for r in result.itertuples():
        pi,po,status=prices[r.model]
        costs.append((r.input_tokens*float(pi)+r.output_tokens*float(po))/1e6 if pi!='' and po!='' and status=='author_verified' else np.nan)
    result['estimated_api_cost_usd']=costs
    result['estimated_api_cost_per_scored_question_usd']=result.estimated_api_cost_usd/200
    result['cost_status']=np.where(result.estimated_api_cost_usd.isna(),'historical_unit_prices_not_supplied','standard_uncached_list_price_estimate')
    means=result.groupby(group[:-1])[numeric+['total_tokens_per_scored_question','total_tokens_per_correct_answer','total_tokens_per_exam_point','output_tokens_per_end_to_end_second','estimated_api_cost_usd','estimated_api_cost_per_scored_question_usd']].mean().reset_index()
    return result,means


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--skip-bootstrap',action='store_true')
    args=parser.parse_args();root=args.root.resolve();data=root/'data';tables=root/'results/tables'
    data.mkdir(exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    (root/'results/validation.json').write_text(json.dumps({'status':'running','bootstrap_requested':not args.skip_bootstrap})+'\n')
    sources=pd.read_csv(root/'provenance/source_inventory.csv',keep_default_na=False)
    for row in sources.itertuples():
        source=root/row.package_path
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=row.public_sha256:
            raise ValueError(f'Missing or changed deposited source: {row.package_path}')
    runs=load_json(root/'config/run_manifest.json');key=read_inputs(root)
    if len(runs)!=24 or len({r['condition_id'] for r in runs})!=24:raise ValueError('Expected 24 unique conditions')
    cells={(r['model'],r['exam'],r['strategy']) for r in runs}
    expected_cells={(m,e,s) for m in {r['model'] for r in runs} for e in ['BIR','FIR'] for s in ['item-wise','whole-exam']}
    if len({r['model'] for r in runs})!=6 or cells!=expected_cells:raise ValueError('Incomplete factorial design')
    ia=image_audit(root)
    items,audit,attempts,runinfo=extract(root,runs)
    all_items=rescore(items,key);scored=all_items[all_items.included_in_official_200].copy()
    ident=['condition_id','replicate','question_number']
    assert len(all_items)==10080 and not all_items.duplicated(ident).any()
    assert len(scored)==9600 and scored.groupby(['condition_id','replicate']).size().eq(200).all()
    assert scored.groupby(['exam','model','strategy','replicate']).has_image.sum().loc['FIR'].eq(20).all()
    if len(audit):
        audit=audit.merge(scored[ident+['correct_answer','is_correct','score']],on=ident,how='left',validate='one_to_one')
        audit['included_in_official_200']=audit['score'].notna()
    saved_audit=audit[audit.included_in_official_200]
    assert (saved_audit.recode_rule=='unique_answer_recovery').sum()==43
    assert (saved_audit.recode_rule=='multiple_options_to_blank').sum()==11
    for file,df in [('official_answer_key_210.csv',key),('all_210_responses_rescored.csv',all_items),('official_200_scored_items.csv',scored),('reclassification_audit_all.csv',audit),('reclassification_audit_scored.csv',saved_audit),('request_records.csv',attempts),('run_manifest.csv',runinfo)]:df.to_csv(data/file,index=False)
    rep,summary=aggregate(scored)
    rep.to_csv(tables/'replicate_summary.csv',index=False);summary.to_csv(tables/'condition_summary.csv',index=False)
    a,b=agreement(scored);a.to_csv(tables/'replicate_agreement.csv',index=False);b.to_csv(tables/'strategy_agreement_ac1.csv',index=False)
    er,em=efficiency(attempts,runinfo,scored,root);er.to_csv(tables/'efficiency_by_replicate.csv',index=False);em.to_csv(tables/'efficiency_by_condition.csv',index=False)
    ia.to_csv(tables/'image_hash_verification.csv',index=False)
    if not args.skip_bootstrap:
        spec=importlib.util.spec_from_file_location('supplied_bootstrap',root/'source_scripts/analysis/bootstrap_ci.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for label,subset in [('official_200',scored),('fir_text_only_180',scored[(scored.exam=='FIR')&~scored.has_image]),('fir_image_20',scored[(scored.exam=='FIR')&scored.has_image])]:
            out=module.run_subset(subset,label,10000,2026)
            out.to_csv(tables/f'bootstrap_{label}.csv',index=False)
    validation={'status':'offline_data_checks_passed','generated_answer_positions':len(all_items),'scored_responses':len(scored),'conditions':len(runs),'replicate_runs':len(rep),'unique_answer_recoveries_scored':int((saved_audit.recode_rule=='unique_answer_recovery').sum()),'correct_recovered':int(saved_audit.loc[saved_audit.recode_rule=='unique_answer_recovery','is_correct'].sum()),'multiple_option_recodes_scored':int((saved_audit.recode_rule=='multiple_options_to_blank').sum()),'unused_reserve_recoveries':int((~audit.included_in_official_200).sum()),'minimum_api_attempts_from_supplied_evidence':int(runinfo.minimum_attempts_from_counters_or_rows.sum()),'original_image_files_verified':20,'multimodal_condition_image_rows':len(ia),'image_hash_comparisons':int(ia.logged_sha256.ne('').sum()),'image_hash_mismatches':int(ia.matches.eq(False).sum()),'request_image_hashes_not_logged':int(ia.logged_sha256.eq('').sum()),'bootstrap':{'executed':not args.skip_bootstrap,'resamples':10000,'seed':2026,'method':'item-cluster percentile; source-supplied implementation'},'limitations':['Estimated costs use documented standard uncached rates; they are not a billing ledger.','No official final-answer-key PDF included: official-200 selection follows supplied answer-key CSV and supplied scoring algorithm.','Historical API acquisition is not re-executed; source logs and provenance are retained.','This is a reanalysis/reproducibility package, not an author-approved publication release.']}
    (root/'results/validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':main()
