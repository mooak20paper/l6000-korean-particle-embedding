"""User-requested, frozen post-final follow-up evaluation; no training."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, zipfile, shutil, importlib.metadata as metadata

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evaluation_only/test1200_epoch1_20261018'
OLD=ROOT/'evaluation_only/final_evaluation_baseline_C_20261018'
RUN='small1200_lr1em4_C_seed17_20261018'
NAME='C1200_lr1em4_seed17_epoch1'
PACKAGE=ROOT/'experiments/small600_1200_20261018/training_packages'/RUN
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def info(p): return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def verified(ref):
    p=ROOT/ref['path'];assert sha(p)==ref['sha256'],str(p);return p

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Do not overwrite a frozen protocol.'
    old=read(OLD/'protocol.json');old_manifest=read(OLD/'manifest.json')
    assert sha(OLD/'protocol.json')=='7a36541942ca4f9eb7e3a4b083a476a5236993244b46421f2ff8f33d7d8ede26'
    assert sha(OLD/'manifest.json')=='77bc756061a7fa33427f834a816911ea1e055b9cf054e3dffc00d39d51fc118e'
    for k in ['test_manifest','test_data','klue_manifest','klue_data','selected_result','baseline_weights','sentence_index']: verified(old[k])
    test=rows(verified(old['test_data']));sts=read(verified(old['klue_data']))
    assert len(test)==600 and len(sts)==519 and all(r['human_review']=='complete' and r['evaluation_allowed'] for r in test)
    assert read(ROOT/'evaluation_only/current_final_test.json')['compact_triplets']==old['test_data']
    index=read(verified(old['sentence_index']));sentences=index['sentences']
    assert sentences==list(dict.fromkeys([r[k] for r in test for k in ['anchor','positive','hard_negative']]+[r[k] for r in sts for k in ['sentence1','sentence2']]))
    archive=ROOT/'experiments/small600_1200_20261018/downloads'/(RUN+'_results.zip')
    checked=read(ROOT/'experiments/small600_1200_20261018/verified'/(RUN+'.json'))
    assert checked['result']=='PASS' and sha(archive)==checked['archive_sha256']=='184da8034f8eac7e6b1e1a2fcc652e867c11cedc791024ba9aed5e6bfbc92965'
    assert sha(PACKAGE/'manifest.json')=='46ff527b83de13a72543441c8c03eff8b331f72f444e59911118ee050c883307'
    package_manifest=read(PACKAGE/'manifest.json')
    for name,ref in package_manifest['files'].items(): assert sha(PACKAGE/name)==ref['sha256'],name
    adapter=OUT/'adapter_epoch1';adapter.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        rm=json.loads(z.read('result_manifest.json'));assert rm['run_id']==RUN
        assert z.read('input_manifest.json')==(PACKAGE/'manifest.json').read_bytes()
        for name in ['adapter_config.json','adapter_model.safetensors']:
            member='adapter_epoch1/'+name;data=z.read(member)
            assert hashlib.sha256(data).hexdigest()==rm['files'][member]['sha256']
            (adapter/name).write_bytes(data)
    assert sha(adapter/'adapter_model.safetensors')=='fbb299091b5aa280a0690753d1b7dc8288f9c83a5c890f4d3b09c04e2e56405a'
    config=read(adapter/'adapter_config.json');assert (config['r'],config['lora_alpha'],config['lora_dropout'])==(16,32,.1)
    sys.path.insert(0,str(ROOT/'scripts'))
    from evaluation_guard import registry,check_values,strings,load_records,norm
    urls,texts=registry();train=rows(PACKAGE/'train.jsonl');val=rows(PACKAGE/'validation600.jsonl')
    issues=check_values(list(strings(load_records(PACKAGE/'train.jsonl'))),urls,texts)
    assert not issues,issues
    from external_sts_guard import load_protection
    ext=load_protection();separation={}
    for label,data in [('train1200',train),('validation600',val)]:
        values=[r[k] for r in data for k in ['anchor','positive','hard_negative']]
        extissues=check_values(values,set(),ext);assert not extissues,extissues
        overlap=set(map(norm,values))&{norm(r[k]) for r in test for k in ['anchor','positive','hard_negative']}
        assert not overlap
        separation[label]={'rows':len(data),'external_guard_issues':extissues,'test_sentence_overlap':len(overlap)}
    save(OUT/'separation_check.json',{'status':'PASS','training_guard_issues':issues,'explicit_inputs':separation,'limitation':'Existing group and lineage reviews retained; not proof of unseen pretraining or all event independence.'})
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-m3'),local_files_only=True)
    lengths=[len(x) for x in tok(sentences,truncation=False)['input_ids']]
    assert lengths==index['token_lengths'] and max(lengths)<=160
    save(OUT/'sentence_index.json',{'sentences':sentences,'token_lengths':lengths,'inference_order':sorted(range(len(sentences)),key=lambda i:(lengths[i],i))})
    reference_files={}
    for model in ['baseline','selected_C6000_epoch1']:
        reference_files[model]={}
        for suffix in ['_embeddings.npy','_inference.json','_test_scores.json']:
            name=model+suffix;reference_files[model][suffix]=old_manifest['files'][name];verified(reference_files[model][suffix])
    for name in ['summary.json','KLUE_STS519_문항별점수.csv']: verified(old_manifest['files'][name])
    fixed_keys=['pooling','dimension','normalize','prefix','precision','max_tokens','truncation','batch_size','threads','seed','tie_epsilon','test_primary_metric','mcnemar','test_bootstrap','sts_primary_metric','sts_secondary_metric','sts_primary_label_field','sts_sensitivity_label_field','sts_bootstrap','sts_max_relative_decline']
    protocol={k:old[k] for k in fixed_keys}
    protocol.update(status='FROZEN_BEFORE_NEW_MODEL_INFERENCE',registered_at_utc=now(),user_instruction='1200개 epoch 1로 test해줘',kind='Post-final-evaluation follow-up with repeated Test and KLUE exposure; not independent confirmation',model=NAME,run_id=RUN,epoch=1,train_n=1200,learning_rate=.0001,lora_r=16,lora_alpha=32,lora_dropout=.1,negative='C: own HN plus in-batch positives',training_seed=17,training_batch=4,temperature=.05,models=['baseline','selected_C6000_epoch1',NAME],reference_C_learning_rate=.0002,reference_C_train_n=6000,reference_confounds='C6000 differs in learning rate and training size/update count; not a size-only causal comparison.',archive=info(archive),training_manifest=info(PACKAGE/'manifest.json'),train_data=info(PACKAGE/'train.jsonl'),validation_data=info(PACKAGE/'validation600.jsonl'),adapter_files={p.name:info(p) for p in adapter.iterdir()},test_data=old['test_data'],klue_data=old['klue_data'],baseline_weights=old['baseline_weights'],model_metadata_files={p.relative_to(ROOT/'models/bge-m3').as_posix():info(p) for p in [ROOT/'models/bge-m3/config.json',ROOT/'models/bge-m3/tokenizer_config.json',ROOT/'models/bge-m3/1_Pooling/config.json']},selected_result=old['selected_result'],original_final_protocol=info(OLD/'protocol.json'),original_final_manifest=info(OLD/'manifest.json'),original_final_summary=old_manifest['files']['summary.json'],reference_sts_scores=old_manifest['files']['KLUE_STS519_문항별점수.csv'],reference_files=reference_files,sentence_index=info(OUT/'sentence_index.json'),script=info(Path(__file__)),separation=info(OUT/'separation_check.json'),sentence_count=len(sentences),max_actual_tokens=max(lengths),fresh_sentence_count=len(sentences),versions={p:metadata.version(p) for p in old['versions']},no_training=True,no_model_reselection=True,no_dataset_edits=True,all_derivatives_permanent_evaluation_only=True)
    assert protocol['versions']==old['versions']
    save(OUT/'protocol.json',protocol)
    title='## Train1200 epoch1 추가 Test 평가 승인 (2026-10-18)'
    block=f'\n\n{title}\n\n- 사용자 최신 지시 “1200개 epoch 1로 test해줘”에 따라 LR0.0001·seed17·r16/alpha32/dropout0.1·C의 완료된 epoch1만 고정 Test600 및 KLUE-STS519에 추가 평가한다. 권위는 evaluation_only/test1200_epoch1_20261018/protocol.json이다. 기존 학습 단계의 Test 미사용 지시는 당시 실행 범위이며 이번 명시 평가 요청을 별도 등록한다.\n- 기존 CPU FP32·CLS/L2·FP64 cosine·epsilon1e-6과 동일한 계산 설정 및 KLUE baseline 대비 상대10% 허용 기준을 유지한다. Baseline과 원래 C6000(LR0.0002) 결과는 해시·전수 점수 검산 후 비교한다. 기존 최종 선정 및 평가를 대체하지 않는다.\n- 후속 개발 이후 이미 노출된 Test/KLUE에 대한 반복 평가임을 보고하며 독립 확증으로 해석하지 않는다. 학습·다른 checkpoint 평가·점수에 따른 문장 변경·GitHub push를 하지 않는다. 새 점수·벡터·통계·보고서도 영구 평가 파생으로 격리한다.\n'
    for p in [ROOT/'AGENTS.md',ROOT/'evaluation_only/AGENTS.md']:
        s=p.read_text(encoding='utf-8-sig')
        if title not in s:p.write_text(s.rstrip()+block,encoding='utf-8')
    save(OUT/'progress.json',{'status':'PREPARED','sentences':len(sentences),'updated_at_utc':now()})
    print(json.dumps({'status':'PREPARED','sentences':len(sentences),'adapter_sha256':sha(adapter/'adapter_model.safetensors')}),flush=True)

def run():
    p=read(OUT/'protocol.json')
    for k in ['archive','train_data','validation_data','baseline_weights','test_data','klue_data','selected_result','original_final_protocol','original_final_manifest','original_final_summary','sentence_index','script','separation']:verified(p[k])
    for ref in p['adapter_files'].values():verified(ref)
    for ref in p['model_metadata_files'].values():verified(ref)
    import numpy as np
    import torch
    from torch.nn.functional import normalize
    from transformers import AutoTokenizer,AutoModel
    from peft import PeftModel
    import psutil
    torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(17)
    index=read(OUT/'sentence_index.json');s=index['sentences'];n=len(s)
    vectors=np.zeros((n,1024),dtype=np.float32);done=np.zeros(n,dtype=bool)
    checkpoint=OUT/'inference_checkpoint.npz';final=OUT/(NAME+'_embeddings.npy');record=OUT/(NAME+'_inference.json')
    if final.exists() and record.exists():verified(read(record)['embeddings']);print('ALREADY_COMPLETE');return
    if checkpoint.exists():
        with np.load(checkpoint) as c:
            assert c['protocol_sha256'].item()==sha(OUT/'protocol.json');vectors=c['vectors'];done=c['done']
    order=[i for i in index['inference_order'] if not done[i]];start=time.perf_counter()
    print('Loading frozen CPU FP32 BGE-M3 and Train1200 epoch1 adapter',flush=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-m3'),local_files_only=True)
    base=AutoModel.from_pretrained(str(ROOT/'models/bge-m3'),local_files_only=True,use_safetensors=False,torch_dtype=torch.float32,low_cpu_mem_usage=True).to('cpu')
    model=PeftModel.from_pretrained(base,str(OUT/'adapter_epoch1'),adapter_name='frozen1200_epoch1',is_trainable=False).to('cpu').eval()
    assert model.active_adapters==['frozen1200_epoch1']
    for param in model.parameters():param.requires_grad_(False)
    with torch.inference_mode():
        for off in range(0,len(order),12):
            inds=order[off:off+12];batch=tok([s[i] for i in inds],padding=True,truncation=False,return_tensors='pt')
            v=normalize(model(**batch).last_hidden_state[:,0].float(),p=2,dim=1)
            assert torch.isfinite(v).all() and torch.allclose(v.norm(dim=1),torch.ones(len(inds)),atol=2e-6)
            vectors[inds]=v.numpy();done[inds]=True
            if off%120==0 or off+12>=len(order):
                temp=checkpoint.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez(f,vectors=vectors,done=done,protocol_sha256=np.array(sha(OUT/'protocol.json')))
                temp.replace(checkpoint)
                progress={'status':'RUNNING','completed':int(done.sum()),'total':n,'seconds':time.perf_counter()-start,'rss_gb':psutil.Process().memory_info().rss/1e9,'updated_at_utc':now()}
                save(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert done.all() and np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6)
    np.save(final,vectors)
    save(record,{'model':NAME,'seconds':time.perf_counter()-start,'fresh_sentences':n,'reused_sentences':0,'embeddings':info(final),'protocol_sha256':sha(OUT/'protocol.json'),'finished_at_utc':now()})
    for k in ['test_data','klue_data','selected_result']:verified(p[k])
    save(OUT/'inference_completion.json',{'status':'COMPLETED','finished_at_utc':now(),'no_training':True,'protocol_sha256':sha(OUT/'protocol.json')})
    save(OUT/'progress.json',{'status':'INFERENCE_COMPLETE','updated_at_utc':now()})
    print('INFERENCE_COMPLETED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:run()
