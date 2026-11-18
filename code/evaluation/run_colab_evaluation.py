"""Fresh GPU-only inference, frozen references checked before public model runs."""
import argparse,csv,gc,hashlib,importlib.metadata as md,json,os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,data):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('Output not empty; keep prior attempts separately')
    a.output.mkdir(parents=True,exist_ok=True);started=time.time()
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    from scipy.stats import spearmanr,pearsonr
    from torch.nn.attention import sdpa_kernel,SDPBackend
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; CPU inference is deliberately disabled')
    torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    protocol=read(a.bundle/'protocol_v2.json')
    for rel,want in read(a.bundle/'bundle_manifest.json').items():
        if sha(a.bundle/rel)!=want:raise RuntimeError('Bundle hash mismatch: '+rel)
    for rel,item in protocol['input_files'].items():
        if sha(a.bundle/rel)!=item['sha256']:raise RuntimeError('Frozen input mismatch: '+rel)
    test=rows(a.bundle/'test600.csv');sts=read(a.bundle/'klue519.json')
    assert len(test)==600 and len(sts)==519 and len({r['id'] for r in test})==600
    assert set(Counter((r['domain'],r['grouped_type']) for r in test).values())=={12}
    assert Counter(r['positive_form'] for r in test)=={'diverse':419,'simple':181}
    texts=list(dict.fromkeys([r[k] for r in test for k in ('anchor','positive','hard_negative')]+[r[k] for r in sts for k in ('sentence1','sentence2')]))
    assert len(texts)==2838
    save(a.output/'sentence_index.json',texts);save(a.output/'protocol.json',protocol)
    env={'utc_start':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
      'packages':{k:md.version(k) for k in ['torch','transformers','peft','numpy','scipy','huggingface_hub','safetensors','accelerate']},
      'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
      'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
      'command':sys.argv,'batch_size':12,'precision':'FP32 no autocast; TF32 off; FP64 cosine','seed':17}
    save(a.output/'environment.json',env)
    summary={'status':'IN_PROGRESS','fresh_inference':True,'models':{},'protocol_sha256':sha(a.bundle/'protocol_v2.json')}
    save(a.output/'summary.json',summary)
    ref_test={k:{r['id']:r for r in rows(a.bundle/'reference'/file)} for k,file in [('baseline','baseline_test600_scores.csv'),('l6000','L6000_test600_scores.csv')]}
    ref_sts={r['guid']:r for r in rows(a.bundle/'reference/klue_sts519_scores.csv')}
    lookup={s:i for i,s in enumerate(texts)}
    gold=np.asarray([float(r['labels']['real-label']) for r in sts])
    try:
        for key in protocol['execution_order']:
            t0=time.time();spec=protocol['models'][key];print('START '+key,flush=True)
            allow=['config.json','tokenizer*','special_tokens_map.json','sentencepiece*','*.model','modules.json','1_Pooling/config.json','README.md','config_sentence_transformers.json','sentence_bert_config.json']
            allow+=['pytorch_model.bin'] if key in ['baseline','l6000'] else ['model.safetensors','pytorch_model.bin']
            snap=Path(snapshot_download(spec['repo'],revision=spec['revision'],allow_patterns=allow))
            wf=snap/('pytorch_model.bin' if key in ['baseline','l6000'] or not (snap/'model.safetensors').exists() else 'model.safetensors')
            weight_hash=sha(wf)
            if key in ['baseline','l6000']:assert weight_hash=='b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'
            for rel,item in spec['metadata'].items():assert sha(snap/rel)==item['sha256'],rel
            tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
            lengths=[len(x) for x in tok(texts,truncation=False)['input_ids']]
            assert max(lengths)<=160,('Refuse truncation',key,max(lengths))
            order=sorted(range(len(texts)),key=lambda i:(lengths[i],i))
            model=AutoModel.from_pretrained(snap,local_files_only=True,trust_remote_code=False,
                use_safetensors=wf.suffix=='.safetensors',torch_dtype=torch.float32,attn_implementation='sdpa',low_cpu_mem_usage=True)
            if key=='l6000':model=PeftModel.from_pretrained(model,a.bundle/'adapter',is_trainable=False)
            model=model.to('cuda').eval();torch.cuda.reset_peak_memory_stats()
            vectors=np.empty((len(texts),1024),dtype=np.float32)
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                for off in range(0,len(order),12):
                    ix=order[off:off+12];batch=tok([texts[i] for i in ix],padding=True,truncation=False,return_tensors='pt').to('cuda')
                    hidden=model(**batch).last_hidden_state.float()
                    if spec['pooling']=='cls':emb=hidden[:,0]
                    else:
                        mask=batch['attention_mask'].unsqueeze(-1).float();emb=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
                    emb=torch.nn.functional.normalize(emb,p=2,dim=1)
                    assert torch.isfinite(emb).all();vectors[ix]=emb.cpu().numpy()
                    if off%240==0:
                        state={'phase':'INFERENCE','model':key,'completed':off+len(ix),'total':len(texts),'elapsed_seconds':time.time()-t0}
                        save(a.output/'progress.json',state);print(json.dumps(state),flush=True)
            np.save(a.output/(key+'_embeddings.npy'),vectors)
            v=vectors.astype(np.float64);norm=np.linalg.norm(v,axis=1,keepdims=True);assert (norm>0).all();v/=norm
            def cos(x,y):return float(v[lookup[x]]@v[lookup[y]])
            scored=[]
            for r in test:
                p=cos(r['anchor'],r['positive']);h=cos(r['anchor'],r['hard_negative']);d=p-h
                scored.append({'id':r['id'],'domain':r['domain'],'grouped_type':r['grouped_type'],'positive_form':r['positive_form'],
                  'cos_ap':p,'cos_ahn':h,'delta':d,'correct':int(d>1e-6),'tie':int(abs(d)<=1e-6),'hn_above_positive':int(d < -1e-6)})
            pred=np.array([cos(r['sentence1'],r['sentence2']) for r in sts])
            sr=[{'guid':r['guid'],'gold':float(g),'cosine':float(p)} for r,g,p in zip(sts,gold,pred)]
            csvout(a.output/(key+'_test600.csv'),scored);csvout(a.output/(key+'_klue_sts519.csv'),sr)
            breakdown={}
            for field in ['domain','grouped_type','positive_form']:
                breakdown[field]={}
                for value in sorted({r[field] for r in scored}):
                    subset=[r for r in scored if r[field]==value]
                    breakdown[field][value]={'n':len(subset),'correct':sum(r['correct'] for r in subset),'accuracy':float(np.mean([r['correct'] for r in subset])),'mean_delta':float(np.mean([r['delta'] for r in subset]))}
            metric={'n':600,'correct':sum(r['correct'] for r in scored),'accuracy':float(np.mean([r['correct'] for r in scored])),
              'macro_accuracy_10':float(np.mean([r['accuracy'] for r in breakdown['grouped_type'].values()])),
              'ties':sum(r['tie'] for r in scored),'hn_confusion_count':sum(r['hn_above_positive'] for r in scored),
              'mean_delta':float(np.mean([r['delta'] for r in scored])),'klue_spearman':float(spearmanr(gold,pred).statistic),
              'klue_pearson':float(pearsonr(gold,pred).statistic),'breakdown':breakdown,'specification':spec,
              'weight_file':wf.name,'weight_sha256':weight_hash,'max_observed_tokens':max(lengths),'inference_seconds':time.time()-t0,
              'tokenizer_class':type(tok).__name__,'local_model_files':{p.relative_to(snap).as_posix():sha(p) for p in snap.rglob('*') if p.is_file() and p.suffix in ['.json','.model','.txt']},
              'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'fresh_inference':True}
            if key in ref_test:
                rt=ref_test[key];column='baseline_cosine' if key=='baseline' else 'L6000_cosine'
                assert set(rt)=={r['id'] for r in scored} and set(ref_sts)=={r['guid'] for r in sr}
                disagreements=sum(r['correct'] != int(rt[r['id']]['correct'].lower()=='true') for r in scored)
                delta=max(abs(r[k]-float(rt[r['id']][old])) for r in scored for k,old in [('cos_ap','cosine_anchor_positive'),('cos_ahn','cosine_anchor_hn')])
                oldpred=np.array([float(ref_sts[r['guid']][column]) for r in sr]);sts_delta=float(np.max(np.abs(pred-oldpred)))
                rho_delta=metric['klue_spearman']-float(spearmanr(gold,oldpred).statistic)
                gate={'prediction_disagreements':disagreements,'max_test_cosine_delta':delta,'max_sts_cosine_delta':sts_delta,'spearman_delta':rho_delta,
                  'passed':bool(disagreements==0 and max(delta,sts_delta)<=1e-4 and abs(rho_delta)<=1e-4)}
                metric['reproduction_gate']=gate
            summary['models'][key]=metric;save(a.output/'summary.json',summary)
            del model,vectors,v;gc.collect();torch.cuda.empty_cache()
            if key in ref_test and not metric['reproduction_gate']['passed']:raise RuntimeError('Reproduction mismatch; public evaluation stopped: '+key)
            print('COMPLETE '+key+' '+str(metric['correct'])+'/600',flush=True)
        for m in summary['models'].values():
            m['sts_retention_percent']=100*m['klue_spearman']/summary['models']['baseline']['klue_spearman']
            m['sts_relative_drop_percent']=100-m['sts_retention_percent']
        summary.update(status='INFERENCE_COMPLETED_PENDING_STATISTICS',elapsed_seconds=time.time()-started)
        save(a.output/'summary.json',summary)
        # Frozen inputs are hashed again; any modification invalidates this run.
        assert all(sha(a.bundle/rel)==item['sha256'] for rel,item in protocol['input_files'].items())
        save(a.output/'input_immutability.json',{'passed':True,'files':protocol['input_files']})
        subprocess.run([sys.executable,str(a.bundle/'analyze_results.py'),'--results',str(a.output)],check=True)
    except BaseException as e:
        summary.update(status='FAILED_OR_INTERRUPTED',error=str(e));save(a.output/'summary.json',summary)
        (a.output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
