"""Nested C2000/4000/6000 + frozen-teacher cosine preservation, one epoch, four checkpoints."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,hashlib,importlib.metadata,json,math,os,shutil,sys,time,traceback
import torch
from torch.nn import functional as F
import common_training as old
read,rows,sha,dump=old.read,old.rows,old.sha,old.dump

def retention_loss(u,v,target):
    return (((u*v).sum(-1)-target)**2).mean()

def numerical_preflight():
    torch.manual_seed(17)
    u=torch.randn(4,9,dtype=torch.float64,requires_grad=True);v=torch.randn(4,9,dtype=torch.float64,requires_grad=True);target=torch.randn(4,dtype=torch.float64)
    loss=retention_loss(u,v,target);errors=(u*v).sum(-1)-target
    expected=sum((sum(u[i,j]*v[i,j] for j in range(9))-target[i])**2 for i in range(4))/4
    assert torch.allclose(loss,expected,atol=1e-12)
    gu,gv=torch.autograd.grad(loss,(u,v));assert torch.allclose(gu,2*errors[:,None]*v/4,atol=1e-12);assert torch.allclose(gv,2*errors[:,None]*u/4,atol=1e-12)
    assert retention_loss(u,v,(u*v).sum(-1).detach()).item()==0
    # The C objective remains exactly the existing tested implementation.
    a,p,n=[F.normalize(torch.randn(4,9,dtype=torch.float64),dim=1).requires_grad_() for _ in range(3)]
    positions=[[j for j in range(4) if j!=i] for i in range(4)]
    c=old.negative_loss(a,p,n,.05,'C',positions)
    manual=[]
    for i in range(4):
        positive=(a[i]*p[i]).sum()/.05
        logits=torch.stack([positive,(a[i]*n[i]).sum()/.05]+[(a[i]*p[j]).sum()/.05 for j in positions[i]])
        manual.append(torch.logsumexp(logits,0)-positive)
    assert torch.allclose(c,torch.stack(manual).mean(),atol=1e-12)
    return {'result':'PASS','retention_MSE_value_and_analytic_gradients':True,'zero_deviation_zero_loss':True,'C_objective_unchanged':True}

def load_inputs(package):
    m=read(package/'manifest.json');assert m['training_authorized'] and not m['test_included'] and not m['final_external_sts_included']
    for rel,ref in m['files'].items():
        p=(package/rel).resolve();assert p.is_relative_to(package.resolve()) and p.is_file() and sha(p)==ref['sha256'] and p.stat().st_size==ref['size'],rel
    cfg=read(package/'run_config.json');assert cfg['run_id']==m['run_id'] and cfg['lambda_retention'] in (0,1,10)
    train=rows(package/'train.jsonl');val=rows(package/'validation600.jsonl');sts=rows(package/'korsts_dev.jsonl');general=rows(package/'retention_pairs2000.jsonl')
    schedule=rows(package/'batches.jsonl');gs=rows(package/'retention_batches.jsonl');stage=read(package/'stage.json')
    n=cfg['common']['train']['rows'];steps=n//4
    assert n in (2000,4000,6000)
    assert (len(train),len(val),len(sts),len(general),len(schedule),len(gs))==(n,600,1376,2000,steps,steps)
    assert [r['id'] for r in train]==stage['ids'];by_index=dict(zip(stage['row_indices'],train))
    assert all(r['approved_for_training'] and r['training_input_allowed'] and r['split']=='Train' for r in train)
    assert all(r['human_review']=='complete' and r['split']=='Validation' and not r['approved_for_training'] for r in val)
    assert set(collections.Counter((r['domain'],r['grouped_type']) for r in train).values())=={n//50}
    assert [b['step'] for b in schedule]==list(range(1,steps+1)) and all(b['epoch']==1 for b in schedule)
    assert sorted(i for b in schedule for i in b['row_indices'])==sorted(stage['row_indices'])
    for b in schedule:
        assert len({by_index[i]['group_id'] for i in b['row_indices']})==4
        assert b['negative_positive_positions']==[[j for j in range(4) if j!=i] for i in range(4)]
    teacher_batches=rows(package/'teacher_batches.jsonl')
    assert sorted(i for b in teacher_batches for i in b['pair_indices'])==list(range(2000))
    assert len(teacher_batches)==300
    assert [b['step'] for b in gs]==list(range(1,steps+1))
    assert all(b['pair_indices']==teacher_batches[i%300]['pair_indices'] for i,b in enumerate(gs))
    assert sum(len(b['pair_indices']) for b in gs)==cfg['retention_pair_exposures']
    assert {len(b['pair_indices']) for b in gs}=={6,7}
    for field in ('id','group_id','anchor','article_key'):
        assert not ({r[field] for r in train if r[field]}&{r[field] for r in val if r[field]})
    assert read(package/'official_guard.json')['result']=='STRUCTURAL_CHECK_PASSED'
    assert read(package/'official_retention_guard.json')['result']=='STRUCTURAL_CHECK_PASSED'
    assert cfg['common']['epochs']==1 and cfg['checkpoints']==[steps//4,steps//2,3*steps//4,steps]
    assert cfg['common']['learning_rate']==.0001 and cfg['seed']==17 and cfg['arm']=='C'
    return cfg,train,val,sts,general,schedule,gs,by_index

def pair_vectors(model,tok,batch,maxlen):
    texts=[r[k] for r in batch for k in ('sentence1','sentence2')]
    tokens=tok(texts,padding=True,truncation=False,return_tensors='pt');assert tokens['input_ids'].shape[1]<=maxlen
    with torch.autocast('cuda',dtype=torch.float16):
        hidden=model(**{k:v.to('cuda') for k,v in tokens.items()}).last_hidden_state[:,0]
    v=F.normalize(hidden.float(),p=2,dim=1)
    return v[0::2],v[1::2]

def run(package,out):
    from transformers import AutoModel,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    cfg,train,val,sts,general,schedule,gs,by_index=load_inputs(package);c=cfg['common'];lam=cfg['lambda_retention']
    n=len(train);steps=n//4
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    assert torch.__version__=='2.11.0+cu128' and importlib.metadata.version('transformers')=='4.57.6' and importlib.metadata.version('peft')=='0.20.0'
    assert not out.exists(),'Do not overwrite earlier attempts.'
    out.mkdir();start=time.perf_counter();dump(out/'numerical_preflight.json',numerical_preflight())
    dump(out/'run_config.json',cfg);shutil.copy2(package/'manifest.json',out/'input_manifest.json')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;old.seed_all(17)
    tok=AutoTokenizer.from_pretrained(c['model_id'],revision=c['model_revision'])
    texts=[r[k] for r in train+val for k in ('anchor','positive','hard_negative')]+[r[k] for r in sts+general for k in ('sentence1','sentence2')]
    lens=[len(v) for v in tok(texts,truncation=False)['input_ids']];assert max(lens)<=160
    dump(out/'token_check.json',{'sentences':len(texts),'max_tokens':max(lens),'truncated':0})
    base=AutoModel.from_pretrained(c['model_id'],revision=c['model_revision']).to('cuda').eval();assert base.config._commit_hash==c['model_revision']
    runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'packages':{p:importlib.metadata.version(p) for p in ('transformers','peft','numpy','scipy','safetensors')},'precision':'FP32 base, FP16 autocast, FP32 normalized cosine/MSE','attention':base.config._attn_implementation,'seed':17}
    dump(out/'runtime.json',runtime)
    baseline,vs=old.evaluate(base,tok,val,c);bs,ss=old.evaluate_sts(base,tok,sts,160)
    assert baseline['correct']==589 and abs(bs['spearman']-.8369954004811585)<1e-9
    dump(out/'baseline_validation.json',baseline);dump(out/'baseline_sts.json',bs);old.save_scores(out/'baseline_validation_scores.csv',vs);old.save_scores(out/'baseline_sts_scores.csv',ss)
    # Cache teacher scores before creating/optimizing LoRA. No teacher gradients.
    targets=torch.empty(2000,dtype=torch.float32,device='cuda');teacher_start=time.perf_counter()
    with torch.inference_mode():
        for batch in rows(package/'teacher_batches.jsonl'):
            ii=batch['pair_indices'];u,v=pair_vectors(base,tok,[general[i] for i in ii],160);targets[ii]=(u*v).sum(-1)
    assert torch.isfinite(targets).all()
    targetrows=[{'id':r['id'],'teacher_cosine':float(targets[i]),'selection_band':r['selection_band']} for i,r in enumerate(general)]
    old.save_scores(out/'teacher_scores.csv',targetrows)
    dump(out/'teacher_manifest.json',{'model_revision':c['model_revision'],'source_pairs_sha256':sha(package/'retention_pairs2000.jsonl'),'batch_schedule_sha256':sha(package/'teacher_batches.jsonl'),'training_retention_schedule_sha256':sha(package/'retention_batches.jsonl'),'teacher_scores_sha256':sha(out/'teacher_scores.csv'),'pairs':2000,'seconds':time.perf_counter()-teacher_start,'frozen_teacher':True,'dropout':False,'general_targets_not_human_labels':True})
    base.gradient_checkpointing_enable();base.enable_input_require_grads();old.seed_all(17);l=c['lora']
    model=get_peft_model(base,LoraConfig(r=l['rank'],lora_alpha=l['alpha'],lora_dropout=l['dropout'],target_modules=l['target_modules'],bias='none'))
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];parameters=[p for _,p in named]
    count=sum(p.numel() for p in parameters);assert count==1572864
    h=hashlib.sha256()
    for name,p in named:h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    assert h.hexdigest()==cfg['expected_initial_adapter_sha256']
    params={'trainable_parameters':count,'total_parameters':sum(p.numel() for p in model.parameters()),'initial_adapter_sha256':h.hexdigest()};dump(out/'parameters.json',params);tok.save_pretrained(out/'tokenizer')
    opt=c['optimizer'];optimizer=torch.optim.AdamW(parameters,lr=.0001,weight_decay=opt['weight_decay'],betas=tuple(opt['betas']),eps=opt['eps']);scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats();history=[];updates=0;skips=0;trainsecs=0.;allsteps=[]
    with (out/'steps.jsonl').open('w',encoding='utf8',buffering=1) as f:
        for b,g in zip(schedule,gs):
            t=time.perf_counter();step=b['step'];optimizer.zero_grad(set_to_none=True);model.train()
            a,p,negative=old.encode_training(model,tok,[by_index[i] for i in b['row_indices']],160,'C')
            particle=old.negative_loss(a,p,negative,.05,'C',b['negative_positive_positions']);assert torch.isfinite(particle)
            scaler.scale(particle).backward()
            diagnostics=step==1 or step in cfg['checkpoints']
            gp=[x.grad.detach().clone() if x.grad is not None else torch.zeros_like(x) for x in parameters] if diagnostics else None
            # eval() disables dropout but gradients remain enabled. Sequential
            # backward limits memory and preserves the particle dropout RNG stream.
            model.eval();ii=g['pair_indices'];u,v=pair_vectors(model,tok,[general[i] for i in ii],160)
            keep=retention_loss(u,v,targets[ii]);assert torch.isfinite(keep)
            # Compute the same branch at lambda0 too; no extra RNG or data-order confound.
            scaler.scale(lam*keep).backward();model.train()
            grad_diag={}
            if diagnostics:
                scale=scaler.get_scale();partnorm=torch.sqrt(sum((x.float()/scale).square().sum() for x in gp))
                keepnorm=torch.sqrt(sum(((p.grad.float()-x.float())/scale).square().sum() for p,x in zip(parameters,gp)))
                grad_diag={'particle_grad_norm':float(partnorm) if torch.isfinite(partnorm) else None,'weighted_retention_grad_norm':float(keepnorm) if torch.isfinite(keepnorm) else None}
                del gp
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(parameters,1.0);scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skip=scaler.get_scale()<scale
            skips+=int(skip);updates+=int(not skip);torch.cuda.synchronize();trainsecs+=time.perf_counter()-t
            record={'step':step,'epoch_fraction':step/steps,'particle_loss':float(particle.detach()),'retention_mse':float(keep.detach()),'weighted_retention_loss':float(lam*keep.detach()),'total_loss':float((particle+lam*keep).detach()),'retention_pairs':len(ii),'lambda_retention':lam,'grad_norm':float(norm) if torch.isfinite(norm) else None,'optimizer_step_skipped':skip,'gradient_scale':scaler.get_scale(),**grad_diag}
            f.write(json.dumps(record,allow_nan=False)+'\n');allsteps.append(record)
            if step==1 or step%25==0:dump(out/'progress.json',record);print('progress '+json.dumps(record),flush=True)
            if step in cfg['checkpoints']:
                # Evaluate without changing training RNG; preserve every prespecified checkpoint.
                t=time.perf_counter()
                with torch.random.fork_rng(devices=[0]):
                    metrics,vs=old.evaluate(model,tok,val,c);sm,ss=old.evaluate_sts(model,tok,sts,160)
                sm['relative_decline_percent']=100*(bs['spearman']-sm['spearman'])/bs['spearman'];sm['eligible']=sm['spearman']>=.9*bs['spearman']
                old.save_scores(out/f'step{step:03d}_validation_scores.csv',vs);old.save_scores(out/f'step{step:03d}_sts_scores.csv',ss)
                model.save_pretrained(out/f'adapter_step{step:03d}',safe_serialization=True)
                item={'step':step,'epoch_fraction':step/steps,**metrics,'sts':sm,'evaluation_seconds':time.perf_counter()-t,'training_seconds_cumulative':trainsecs,'optimizer_updates':updates,'skipped_steps':skips,'gpu_peak':old.cuda_memory()}
                history.append(item);dump(out/'history.json',history);print('checkpoint '+json.dumps(item,ensure_ascii=False),flush=True)
                model.train()
    eligible=[x for x in history if x['sts']['eligible']]
    best=max(eligible,key=lambda x:(x['macro_accuracy_10'],x['sts']['spearman'],-x['step'])) if eligible else None
    report={'status':'COMPLETED_DEVELOPMENT_ONLY','run_id':cfg['run_id'],'lambda_retention':lam,'train_count':n,'retention_pairs':2000,'retention_pair_exposures':sum(x['retention_pairs'] for x in allsteps),'scheduled_steps':steps,'epochs':1,'checkpoints':len(history),'optimizer_updates':updates,'skipped_steps':skips,**params,'training_seconds':trainsecs,'total_execution_seconds':time.perf_counter()-start,'gpu_peak':old.cuda_memory(),'baseline_validation':baseline,'baseline_sts':bs,'history':history,'selected_step':best['step'] if best else None,'teacher_scores_sha256':sha(out/'teacher_scores.csv'),'test_evaluated':False,'klue_sts_evaluated':False,'test_read':False,'final_model_selected':False,'human_review_changed':False,'selection_rule':'STS >=0.9 baseline required; highest Validation macro, then STS, then earlier step','scope':'Exploratory development screening using existing approved Validation600 and KorSTS1376. Diverse-Positive validation and new independent external confirmation remain next steps.'}
    dump(out/'training_report.json',report);print('complete '+json.dumps({'run_id':cfg['run_id'],'lambda':lam,'selected_step':report['selected_step'],'seconds':report['total_execution_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.preflight_only:
        load_inputs(a.package);print(json.dumps(numerical_preflight()));sys.exit(0)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    try:run(a.package,a.output)
    except BaseException as e:
        if a.output and a.output.exists():dump(a.output/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
