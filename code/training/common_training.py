"""Diverse-Positive comparison: fresh BGE-M3 + paired Model D LoRA, arms A/B/C.

Inputs are an immutable, explicitly packaged Train600/1200 and Validation600.
No Test loader exists here. All three epochs are retained. KorSTS dev retention
requires at least 90% of the fresh baseline Spearman score, fixed before scores.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

import torch
from torch.nn import functional as F


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8-sig').splitlines() if l.strip()]


def logistic_loss(a, p, n, temperature):
    return F.softplus(((a*n).sum(-1)-(a*p).sum(-1))/temperature).mean()



def negative_loss(a,p,n,temperature,arm,positions):
    """One positive versus permitted other positives; C adds ONLY own HN."""
    batch=a.shape[0]
    if arm not in ('A','B','C') or temperature<=0 or len(positions)!=batch:
        raise ValueError('Invalid negative configuration')
    mask=torch.zeros((batch,batch),dtype=torch.bool,device=a.device)
    for i,js in enumerate(positions):
        if not js or len(set(js))!=len(js) or i in js or any(j<0 or j>=batch for j in js):
            raise ValueError('Invalid/empty permitted negatives')
        mask[i,js]=True
    positive=(a*p).sum(-1,keepdim=True)
    negatives=(a@p.T).masked_fill(~mask,float('-inf'))
    columns=[positive,negatives]
    if arm=='A':
        if n is None:raise ValueError('A requires own HN')
        return F.softplus(((a*n).sum(-1)-(a*p).sum(-1))/temperature).mean()
    if arm=='C':
        if n is None:raise ValueError('C requires own HN')
        columns.append((a*n).sum(-1,keepdim=True))
    logits=torch.cat(columns,dim=1)/temperature
    return F.cross_entropy(logits,torch.zeros(batch,dtype=torch.long,device=a.device))

def encode_training(model,tokenizer,batch,max_length,arm):
    if arm in ('A','C'):return encode(model,tokenizer,batch,max_length)
    texts=[r[k] for r in batch for k in ('anchor','positive')]
    tokens=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
    assert tokens['input_ids'].shape[1]<=max_length
    tokens={k:v.to('cuda') for k,v in tokens.items()}
    with torch.autocast('cuda',dtype=torch.float16):
        dense=model(**tokens).last_hidden_state[:,0]
        vectors=F.normalize(dense.float(),p=2,dim=1)
    return vectors[0::2],vectors[1::2],None

def sts_eligible(value, baseline, max_relative_decline=0.10):
    assert math.isfinite(value) and math.isfinite(baseline) and baseline > 0
    return value >= baseline*(1-max_relative_decline)


def summarize(scores, epsilon):
    assert scores
    def group_summary(group):
        deltas = [r['margin'] for r in group]
        return {'count': len(group), 'correct': sum(d > epsilon for d in deltas),
            'ties': sum(abs(d) <= epsilon for d in deltas),
            'accuracy': sum(d > epsilon for d in deltas)/len(group),
            'mean_margin': sum(deltas)/len(group),
            'mean_loss': sum(r['loss'] for r in group)/len(group)}
    result = group_summary(scores)
    for field in ('grouped_type', 'domain', 'subtype'):
        groups = collections.defaultdict(list)
        for row in scores:
            groups[row[field]].append(row)
        result['by_'+field] = {k: group_summary(v) for k,v in sorted(groups.items())}
    types = result['by_grouped_type']
    assert set(types) == {f'T{i}' for i in range(1,11)}
    result['macro_accuracy_10'] = sum(r['accuracy'] for r in types.values())/10
    result['accuracy_strict_gt_zero_reference'] = sum(r['margin'] > 0 for r in scores)/len(scores)
    result['tie_epsilon'] = epsilon
    return result


def load_inputs(package):
    manifest = read(package/'manifest.json')
    assert manifest['run_id'] in ('small600_lr1em4_C_seed17_20261018', 'small1200_lr1em4_C_seed17_20261018')
    assert manifest['training_authorized'] and not manifest['test_included']
    for rel, info in manifest['files'].items():
        p = (package/rel).resolve()
        assert p.is_relative_to(package.resolve()) and p.is_file(), rel
        assert sha(p) == info['sha256'] and p.stat().st_size == info['size'], rel
    cfg = read(package/'run_config.json')
    train, validation = rows(package/'train.jsonl'), rows(package/'validation600.jsonl')
    schedule, stage = rows(package/'batches.jsonl'), read(package/'stage.json')
    n=cfg['common']['train']['rows']
    assert n in (600,1200)
    assert len(train)==n and len(validation)==600 and len(schedule)==n*3//4
    assert [r['id'] for r in train] == stage['ids']
    assert len(set(stage['row_indices'])) == n and len(set(stage['ids'])) == n
    by_master_index = dict(zip(stage['row_indices'], train))
    assert all(r['approved_for_training'] and r['training_input_allowed'] and r['split']=='Train' for r in train)
    assert all(not r['approved_for_training'] and r['split']=='Validation' and r['human_review']=='complete' for r in validation)
    for field in ('id','group_id','anchor','article_key'):
        assert not ({r[field] for r in train if r[field]} & {r[field] for r in validation if r[field]}), field
    assert set(collections.Counter((r['domain'],r['grouped_type']) for r in train).values()) == {n//50}
    for epoch in range(1,4):
        batches = [b for b in schedule if b['epoch']==epoch]
        assert [b['step'] for b in batches] == list(range(1,n//4+1))
        flat = [i for b in batches for i in b['row_indices']]
        assert sorted(flat) == sorted(stage['row_indices'])
        for b in batches:
            assert len(b['row_indices']) == 4
            assert len({by_master_index[i]['group_id'] for i in b['row_indices']}) == 4
            assert b['negative_positive_positions']==[[j for j in range(4) if j!=i] for i in range(4)]
    assert cfg['common']['epochs']==3 and cfg['common']['physical_triplet_batch_size']==4
    assert cfg['common']['gradient_accumulation_steps']==1 and cfg['arm'] in ('A','B','C')
    assert read(package/'official_guard.json')['result']=='STRUCTURAL_CHECK_PASSED'
    external=rows(package/'korsts_dev.jsonl')
    assert len(external)==1376
    assert cfg['sts']['max_relative_decline']==0.10 and cfg['sts']['metric']=='spearman'
    return cfg, train, validation, schedule, by_master_index


def seed_all(seed):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(model, tokenizer, batch, max_length):
    texts = [r[k] for r in batch for k in ('anchor','positive','hard_negative')]
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
    assert tokens['input_ids'].shape[1] <= max_length
    tokens = {k:v.to('cuda') for k,v in tokens.items()}
    with torch.autocast('cuda', dtype=torch.float16):
        dense = model(**tokens).last_hidden_state[:,0]
        vectors = F.normalize(dense.float(),p=2,dim=1)
    return vectors[0::3],vectors[1::3],vectors[2::3]


@torch.inference_mode()
def evaluate(model, tokenizer, data, common):
    model.eval()
    scores = []
    for start in range(0,len(data),4):
        batch = data[start:start+4]
        a,p,n = encode(model,tokenizer,batch,common['max_length'])
        sp,sn = (a*p).sum(-1),(a*n).sum(-1)
        deltas = sp-sn
        losses = F.softplus(-deltas/common['temperature'])
        assert torch.isfinite(deltas).all() and torch.isfinite(losses).all()
        for r,pp,nn,dd,ll in zip(batch,sp.tolist(),sn.tolist(),deltas.tolist(),losses.tolist()):
            scores.append({'id':r['id'],'domain':r['domain'],'subtype':r['subtype'],
                'grouped_type':r['grouped_type'],'group_id':r['group_id'],
                'cosine_positive':pp,'cosine_hard_negative':nn,'margin':dd,
                'loss':ll,'correct':dd>common['tie_epsilon']})
    return summarize(scores,common['tie_epsilon']),scores


def save_scores(path, scores):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]))
        writer.writeheader();writer.writerows(scores)


@torch.inference_mode()
def evaluate_sts(model, tokenizer, data, max_length):
    from scipy.stats import spearmanr, pearsonr
    model.eval();scores=[]
    for start in range(0,len(data),8):
        batch=data[start:start+8]
        texts=[r[k] for r in batch for k in ('sentence1','sentence2')]
        tokens=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        assert tokens['input_ids'].shape[1]<=max_length
        tokens={k:v.to('cuda') for k,v in tokens.items()}
        with torch.autocast('cuda',dtype=torch.float16):
            dense=model(**tokens).last_hidden_state[:,0]
        vectors=F.normalize(dense.float(),p=2,dim=1)
        cosine=(vectors[0::2]*vectors[1::2]).sum(-1)
        assert torch.isfinite(cosine).all()
        for r,s in zip(batch,cosine.tolist()):
            scores.append({'id':r['id'],'human_score':r['score'],'cosine':s,'genre':r['genre']})
    labels=[r['human_score'] for r in scores];pred=[r['cosine'] for r in scores]
    metrics={'count':len(scores),'spearman':float(spearmanr(labels,pred).statistic),
        'pearson':float(pearsonr(labels,pred).statistic)}
    assert math.isfinite(metrics['spearman']) and math.isfinite(metrics['pearson'])
    return metrics,scores


def cuda_memory():
    return {'max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'max_reserved_bytes':torch.cuda.max_memory_reserved()}


def run(package, output):
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    cfg,train,validation,schedule,by_index=load_inputs(package)
    assert cfg['common']['lora']=={'rank':16,'alpha':32,'dropout':0.1,'target_modules':['query','value'],'bias':'none'}
    assert cfg['common']['learning_rate']==0.0001 and cfg['arm']=='C'
    external=rows(package/'korsts_dev.jsonl')
    c=cfg['common']
    assert torch.cuda.is_available(), 'CUDA GPU required; no CPU training fallback.'
    assert 'T4' in torch.cuda.get_device_name(0), 'Use the prespecified T4 for this first run.'
    assert torch.__version__=='2.11.0+cu128', torch.__version__
    assert importlib.metadata.version('transformers')=='4.57.6'
    assert importlib.metadata.version('peft')=='0.20.0'
    assert not output.exists(), 'Preserve previous results; choose a new run directory for a rerun.'
    output.mkdir(parents=True)
    started=time.perf_counter()
    started_at_utc=datetime.now(timezone.utc).isoformat()
    runtime={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'packages':{p:importlib.metadata.version(p) for p in ('transformers','peft','accelerate','numpy','safetensors','scipy')},
        'precision':'fp32 base weights + fp16 autocast + fp32 normalized cosine/loss',
        'deterministic_algorithms':True,'seed':cfg['seed']}
    dump(output/'runtime.json',runtime)
    dump(output/'run_config.json',cfg)
    shutil.copy2(package/'manifest.json',output/'input_manifest.json')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    seed_all(cfg['seed'])
    torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'])
    texts=[r[k] for r in train+validation for k in ('anchor','positive','hard_negative')]
    texts += [r[k] for r in external for k in ('sentence1','sentence2')]
    lengths=[len(v) for v in tokenizer(texts,truncation=False,add_special_tokens=True)['input_ids']]
    assert max(lengths)<=c['max_length'], 'Token truncation forbidden.'
    dump(output/'token_check.json',{'sentences':len(texts),'max_tokens':max(lengths),'truncated':0})
    base=AutoModel.from_pretrained(c['model_id'],revision=c['model_revision']).to('cuda')
    assert base.config._commit_hash==c['model_revision']
    runtime['attention_implementation']=base.config._attn_implementation
    runtime['base_parameter_dtype']=str(next(base.parameters()).dtype)
    dump(output/'runtime.json',runtime)
    t=time.perf_counter()
    baseline,baseline_scores=evaluate(base,tokenizer,validation,c)
    assert baseline['correct']==cfg['reference_baseline']['correct']
    assert abs(baseline['macro_accuracy_10']-cfg['reference_baseline']['macro_accuracy_10'])<1e-10
    baseline['evaluation_seconds']=time.perf_counter()-t
    dump(output/'baseline_validation.json',baseline)
    save_scores(output/'baseline_validation_scores.csv',baseline_scores)
    print('baseline_validation',json.dumps(baseline,ensure_ascii=False),flush=True)
    baseline_sts,sts_scores=evaluate_sts(base,tokenizer,external,c['max_length'])
    assert baseline_sts['spearman']>0
    assert abs(baseline_sts['spearman']-cfg['reference_baseline']['sts_spearman'])<1e-9, 'Baseline numerical mismatch'
    baseline_sts['minimum_allowed_spearman']=0.90*baseline_sts['spearman']
    dump(output/'baseline_sts.json',baseline_sts)
    save_scores(output/'baseline_sts_scores.csv',sts_scores)
    print('baseline_sts',json.dumps(baseline_sts),flush=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    seed_all(cfg['seed'])
    l=c['lora']
    model=get_peft_model(base,LoraConfig(r=l['rank'],lora_alpha=l['alpha'],
        lora_dropout=l['dropout'],target_modules=l['target_modules'],bias=l['bias']))
    trainable=[p for p in model.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in trainable)
    total_count=sum(p.numel() for p in model.parameters())
    assert trainable_count==1572864, trainable_count
    init_hash=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            init_hash.update(name.encode());init_hash.update(p.detach().cpu().numpy().tobytes())
    params={'trainable_parameters':trainable_count,'total_parameters':total_count,
        'trainable_percent':100*trainable_count/total_count,'initial_adapter_sha256':init_hash.hexdigest()}
    assert params['initial_adapter_sha256']==cfg['expected_initial_adapter_sha256'], 'Paired initialization mismatch'
    dump(output/'parameters.json',params)
    tokenizer.save_pretrained(output/'tokenizer')
    opt=c['optimizer']
    optimizer=torch.optim.AdamW(trainable,lr=c['learning_rate'],weight_decay=opt['weight_decay'],
        betas=tuple(opt['betas']),eps=opt['eps'])
    scaler=torch.amp.GradScaler('cuda')
    history=[];global_step=0;optimizer_updates=0;skipped=0
    training_started=time.perf_counter()
    with (output/'steps.jsonl').open('w',encoding='utf-8',buffering=1) as step_log:
        for epoch in range(1,4):
            epoch_start=time.perf_counter();train_seconds=0.;losses=[]
            model.train()
            batches=[b for b in schedule if b['epoch']==epoch]
            for b in batches:
                t=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                a,p,n=encode_training(model,tokenizer,[by_index[i] for i in b['row_indices']],c['max_length'],cfg['arm'])
                loss=negative_loss(a,p,n,c['temperature'],cfg['arm'],b['negative_positive_positions'])
                assert torch.isfinite(loss), 'Non-finite loss'
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.clip_grad_norm_(trainable,c['gradient_clip_norm'])
                previous_scale=scaler.get_scale()
                scaler.step(optimizer);scaler.update()
                step_skipped=scaler.get_scale()<previous_scale
                skipped+=int(step_skipped);optimizer_updates+=int(not step_skipped)
                loss_value=float(loss.detach().cpu());losses.append(loss_value)
                global_step+=1;train_seconds+=time.perf_counter()-t
                record={'epoch':epoch,'step':b['step'],'global_step':global_step,'loss':loss_value,
                    'grad_norm':float(grad_norm) if torch.isfinite(grad_norm) else None,
                    'gradient_scale':scaler.get_scale(),'optimizer_step_skipped':step_skipped,
                    'elapsed_training_seconds':time.perf_counter()-training_started}
                step_log.write(json.dumps(record,allow_nan=False)+'\n')
                if b['step'] in (1,10,50) or b['step']%100==0:
                    print('progress',json.dumps(record),flush=True)
                    dump(output/'progress.json',record)
            torch.cuda.synchronize()
            t=time.perf_counter()
            metrics,scores=evaluate(model,tokenizer,validation,c)
            sts,sts_scores=evaluate_sts(model,tokenizer,external,c['max_length'])
            sts['relative_change']=(sts['spearman']-baseline_sts['spearman'])/baseline_sts['spearman']
            sts['eligible']=sts_eligible(sts['spearman'],baseline_sts['spearman'])
            save_scores(output/f'epoch{epoch}_sts_scores.csv',sts_scores)
            validation_seconds=time.perf_counter()-t
            save_scores(output/f'epoch{epoch}_validation_scores.csv',scores)
            model.save_pretrained(output/f'adapter_epoch{epoch}',safe_serialization=True)
            record={'epoch':epoch,'global_step':global_step,'train_loss':sum(losses)/len(losses),
                'train_seconds':train_seconds,'validation_seconds':validation_seconds,
                'epoch_seconds':time.perf_counter()-epoch_start,'optimizer_updates_total':optimizer_updates,
                'skipped_optimizer_steps_total':skipped,'gpu_peak':cuda_memory(),'sts':sts,**metrics}
            history.append(record)
            dump(output/'history.json',history)
            dump(output/f'epoch{epoch}_validation.json',record)
            print('epoch',json.dumps(record,ensure_ascii=False),flush=True)
    best=max(history,key=lambda h:(h['macro_accuracy_10'],-h['epoch']))
    eligible=[h for h in history if h['sts']['eligible']]
    selected=max(eligible,key=lambda h:(h['macro_accuracy_10'],h['sts']['spearman'],-h['epoch'])) if eligible else None
    report={'status':'TRAINING_COMPLETED_VALIDATION_ONLY','run_id':cfg['run_id'],'arm':cfg['arm'],
        'started_at_utc':started_at_utc,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'training_wall_seconds':time.perf_counter()-training_started,
        'total_execution_seconds':time.perf_counter()-started,
        'train_count':len(train),'validation_count':len(validation),'epochs':len(history),
        'scheduled_steps':len(schedule),'optimizer_updates':optimizer_updates,'skipped_steps':skipped,
        'gpu_peak':cuda_memory(),**params,'baseline_validation':baseline,'baseline_sts':baseline_sts,'history':history,
        'validation_leading_epoch':best['epoch'],'stage_selected_epoch':selected['epoch'] if selected else None,
        'selection_rule':'Keep epochs with Spearman >= 0.90 * fresh baseline; maximize Validation macro_accuracy_10, tie development KorSTS Spearman then earlier epoch.',
        'final_model_selected':False,'external_sts_dev_evaluated':True,'external_sts_final_evaluated':False,
        'selection_limitation':'This is the first size/seed run. Independent final STS confirmation and Test evaluation are not performed.',
        'test_read':False,'test_evaluated':False,'runtime':runtime,'config':cfg}
    dump(output/'training_report.json',report)
    print('training_complete',json.dumps({k:report[k] for k in ('run_id','epochs','optimizer_updates','skipped_steps','validation_leading_epoch','stage_selected_epoch','training_wall_seconds','gpu_peak')}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.preflight_only:
        cfg,train,validation,schedule,_=load_inputs(args.package)
        print(json.dumps({'result':'PASS','train':len(train),'validation':600,'batches':len(schedule),'test_included':False}))
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    existed=args.output.exists()
    try:
        run(args.package,args.output)
    except BaseException as exc:
        if not existed and args.output.is_dir():
            dump(args.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if not existed and args.output.is_dir():
            shutil.make_archive(str(args.output)+'_results','zip',root_dir=args.output)


if __name__=='__main__':
    main()
