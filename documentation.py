"""Generate source-linked supplementary tables and documentation indexes offline."""
from pathlib import Path
import ast, csv, hashlib, json, re
import pandas as pd


def build(root:Path):
    runs=json.loads((root/'config/run_manifest.json').read_text())
    out=root/'data/prompts';out.mkdir(exist_ok=True)
    index=[];settings=[]
    for run in runs:
        source=root/run['raw_file'];meta=json.loads((root/run['metadata_file']).read_text(encoding='utf-8-sig'))
        raw=[json.loads(l) for l in source.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        folder=out/run['condition_id'];folder.mkdir(exist_ok=True)
        # Copy full prompts/preambles already saved in the original records/metadata.
        for f in source.parent.glob('*.txt'):
            (folder/f.name).write_bytes(f.read_bytes())
            index.append({'condition_id':run['condition_id'],'file':str((folder/f.name).relative_to(root)),'type':'source_logged_text','source':str(f.relative_to(root))})
        prompt_records=[]
        for line,r in enumerate(raw,1):
            for key in ['prompt','prompt_text','user_prompt','system_prompt']:
                if isinstance(r.get(key),str) and r[key]:
                    prompt_records.append({'raw_line':line,'replicate':r.get('replicate_id',r.get('replicate')),'question_number':r.get('original_question_number'),'role_or_field':key,'text':r[key]})
        if prompt_records:
            f=folder/'logged_prompts.jsonl';f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in prompt_records))
            index.append({'condition_id':run['condition_id'],'file':str(f.relative_to(root)),'type':'extracted_verbatim_from_logs','source':run['raw_file']})
        for key in ['system_prompt','prompt_preamble']:
            if meta.get(key):
                f=folder/(key+'_from_metadata.txt');f.write_text(meta[key],encoding='utf-8')
                index.append({'condition_id':run['condition_id'],'file':str(f.relative_to(root)),'type':'source_metadata_verbatim','source':run['metadata_file']+'#'+key})
        # Original request/config data are authoritative, not an invented common template.
        gen=meta.get('generation_parameters',{})
        settings.append({'condition_id':run['condition_id'],'model':run['model'],'exam':run['exam'],'strategy':run['strategy'],
                         'metadata_file':run['metadata_file'],'raw_file':run['raw_file'],
                         'temperature':gen.get('temperature',meta.get('temperature','not in metadata')),'top_p':gen.get('top_p',meta.get('top_p','not in metadata')),
                         'generation_parameters_json':json.dumps(gen,ensure_ascii=False),'thinking_config_json':json.dumps(meta.get('thinking',{}),ensure_ascii=False),
                         'output_budget':gen.get('max_output_tokens',gen.get('max_tokens',meta.get('max_tokens','not in metadata'))),
                         'whole_exam_rationale_observed':run['strategy']=='whole-exam' and run['model_folder']=='DeepSeekV4Pro',
                         'image_input':run['exam']=='FIR' and run['model_folder']!='DeepSeekV4Pro',
                         'raw_provider_endpoints_json':json.dumps(sorted({str(r.get('api_endpoint',meta.get('api_endpoint',meta.get('endpoint','not in metadata')))) for r in raw})),
                         'workflow_note':'Original scripts use serial request loops; streaming is not concurrent dispatch. Sleep/retry waits are not uniformly included in logged latency.'})
    # Materialize DeepSeek's otherwise unlogged full prompts from ONLY its pure
    # formatting functions. No acquisition code, imports or API calls execute.
    original=root/'source_scripts/acquisition/deepseek/run_deepseek_v4pro_whole_text_2runs.py'
    tree=ast.parse(original.read_text(encoding='utf-8-sig'))
    wanted={'clean_text','bool_like','normalize_question_number','detect_question_col','load_questions','build_exam_text','build_prompt'}
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    astmod=ast.Module(body=funcs,type_ignores=[])
    from typing import Any,Optional
    ns={'pd':pd,'Path':Path,'Any':Any,'Optional':Optional}
    exec(compile(astmod,str(original),'exec'),ns)
    for exam in ['BIR','FIR']:
        qs=ns['load_questions'](root/'source_data/inputs'/f'{exam.lower()}2024_questions_210.csv')
        name='Spanish Biology residency access examination' if exam=='BIR' else 'Spanish Pharmacy residency access examination'
        text=ns['build_prompt'](name,exam+'2024',qs)
        cid=f'deepseek_{exam.lower()}_whole_exam';f=out/cid/'prompt_RECONSTRUCTED_FROM_SUPPLIED_SCRIPT.txt';f.write_text(text,encoding='utf-8')
        index.append({'condition_id':cid,'file':str(f.relative_to(root)),'type':'reconstructed_from_supplied_pure_prompt_functions; not a captured request payload','source':str(original.relative_to(root))})
    pd.DataFrame(index).to_csv(root/'data/prompt_index.csv',index=False)
    pd.DataFrame(settings).to_csv(root/'results/tables/api_configuration_index.csv',index=False)
    # Provenance of new adapters and original source analysis code is explicit.
    # Source hash matches are checked against acquisition metadata when reported.
    inp=[]
    for run in runs:
        meta=json.loads((root/run['metadata_file']).read_text(encoding='utf-8-sig'))
        for kind in ['questions','answer_key']:
            expected=meta.get(f'{kind}_file_sha256')
            f=root/'source_data/inputs'/f'{run["exam"].lower()}2024_{kind}_210.csv'
            actual=hashlib.sha256(f.read_bytes()).hexdigest()
            inp.append({'condition_id':run['condition_id'],'input':str(f.relative_to(root)),'logged_sha256':expected,'provided_sha256':actual,'matches_if_reported':expected==actual if expected else 'not recorded'})
    pd.DataFrame(inp).to_csv(root/'results/tables/input_hash_verification.csv',index=False)

def reconstruct_additional(root):
    """Evaluate only reviewed pure formatting functions; never import acquisition modules."""
    import typing
    out=root/'data/prompts';index=pd.read_csv(root/'data/prompt_index.csv').to_dict('records')
    qpath=root/'source_scripts/acquisition/qwen/run_bir2024_qwen235b_whole_text_answers_only_2runs_STRICT.py'
    tree=ast.parse(qpath.read_text(encoding='utf-8-sig'));ns={'pd':pd}
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_PREAMBLE' for t in n.targets):ns['PROMPT_PREAMBLE']=ast.literal_eval(n.value)
        if isinstance(n,ast.FunctionDef) and n.name in ['format_question','build_whole_exam_prompt']:nodes.append(n)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(qpath),'exec'),ns)
    qs=pd.read_csv(root/'source_data/inputs/bir2024_questions_210.csv').sort_values('original_question_number')
    prompt=ns['build_whole_exam_prompt'](qs);cid='qwen_bir_whole_exam'
    meta=json.loads((root/f'source_data/runs/{cid}/metadata.json').read_text())
    if hashlib.sha256(prompt.encode()).hexdigest()!=meta['prompt_hash_sha256']:raise ValueError('Reconstructed Qwen prompt hash mismatch')
    f=out/cid/'user_prompt_RECONSTRUCTED_HASH_VERIFIED.txt';f.write_text(prompt)
    index.append({'condition_id':cid,'file':str(f.relative_to(root)),'type':'reconstructed_user_text; recorded_full_prompt_hash_verified','source':str(qpath.relative_to(root))})
    kp=root/'source_scripts/acquisition/kimi/08_run_fir2024_kimi_k2_6_itemwise_images_nonthinking_2reps.py'
    tree=ast.parse(kp.read_text(encoding='utf-8-sig'));ns={'pd':pd,'Path':Path,'Any':typing.Any,'Sequence':typing.Sequence}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['safe_str','build_question_text']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(kp),'exec'),ns)
    qs=pd.read_csv(root/'source_data/inputs/fir2024_questions_210.csv');cid='kimi_fir_item_wise';records=[]
    for _,r in qs.iterrows():
        q=int(r.original_question_number);im=root/f'source_data/inputs/images/fir/Q{q:03d}.png';paths=[im] if im.exists() else []
        records.append({'question_number':q,'role':'user','text':ns['build_question_text'](r,paths),'image_files':[str(p.relative_to(root)) for p in paths],'evidence':'reconstructed from original formatting function and provided inputs; no recorded request hash'})
    f=out/cid/'user_prompts_RECONSTRUCTED.jsonl';f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    index.append({'condition_id':cid,'file':str(f.relative_to(root)),'type':'reconstructed_user_text_and_image_references; not a captured request','source':str(kp.relative_to(root))})
    pd.DataFrame(index).to_csv(root/'data/prompt_index.csv',index=False)
