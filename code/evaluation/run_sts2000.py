"""GPU-only read-only evaluation of five frozen encoders; no training or selection."""
import argparse, csv, gc, hashlib, importlib.metadata as md, json, os
import platform, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    p = Path(p)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def csvout(p, data):
    with Path(p).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def analyze(output, protocol):
    import numpy as np
    from scipy.stats import rankdata, spearmanr, pearsonr
    keys = protocol['execution_order']
    scores = {}
    for key in keys:
        with (output / (key + '_klue_sts2000.csv')).open(encoding='utf-8-sig') as f:
            scores[key] = list(csv.DictReader(f))
    ids = [r['guid'] for r in scores[keys[0]]]
    assert len(ids) == 2000 and len(set(ids)) == 2000
    gold = np.array([float(r['gold']) for r in scores[keys[0]]])
    preds = []
    for key in keys:
        assert [r['guid'] for r in scores[key]] == ids
        assert np.array_equal(gold, [float(r['gold']) for r in scores[key]])
        preds.append(np.array([float(r['cosine']) for r in scores[key]]))
    values = np.stack(preds)
    point = np.array([spearmanr(gold, p).statistic for p in values])
    bootstrap = np.empty((10000, len(keys)))
    rng = np.random.default_rng(17)
    for off in range(0, 10000, 20):
        ix = rng.integers(0, len(gold), size=(min(20, 10000-off), len(gold)))
        g = rankdata(gold[ix], axis=1)
        g -= g.mean(axis=1, keepdims=True)
        for j, pred in enumerate(values):
            p = rankdata(pred[ix], axis=1)
            p -= p.mean(axis=1, keepdims=True)
            bootstrap[off:off+len(ix), j] = (g*p).sum(axis=1) / np.sqrt((g*g).sum(axis=1)*(p*p).sum(axis=1))
    assert np.isfinite(bootstrap).all()
    np.save(output / 'paired_bootstrap_spearman.npy', bootstrap)
    baseline = keys.index('baseline')
    result = read(output / 'summary.json')
    rows = []
    for j, key in enumerate(keys):
        ci = np.quantile(bootstrap[:, j], [.025, .975]).tolist()
        dci = np.quantile(bootstrap[:, j] - bootstrap[:, baseline], [.025, .975]).tolist()
        retention = 100 * point[j] / point[baseline]
        rci = np.quantile(100 * bootstrap[:, j] / bootstrap[:, baseline], [.025, .975]).tolist()
        result['models'][key].update(klue_spearman=float(point[j]), klue_pearson=float(pearsonr(gold, values[j]).statistic),
            spearman_ci95=ci, spearman_difference_from_baseline=float(point[j]-point[baseline]),
            difference_ci95=dci, sts_retention_percent=float(retention), retention_ci95=rci,
            sts_relative_drop_percent=float(100-retention))
        rows.append({'model':key, 'n':2000, 'spearman':float(point[j]), 'spearman_ci95_low':ci[0],
            'spearman_ci95_high':ci[1], 'pearson':result['models'][key]['klue_pearson'],
            'retention_percent':float(retention), 'relative_drop_percent':float(100-retention),
            'difference_from_baseline':float(point[j]-point[baseline]), 'difference_ci95_low':dci[0],
            'difference_ci95_high':dci[1]})
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append({'model_a':keys[i], 'model_b':keys[j], 'difference_a_minus_b':float(point[i]-point[j]),
                'paired_percentile_ci95':np.quantile(bootstrap[:,i]-bootstrap[:,j],[.025,.975]).tolist()})
    save(output / 'paired_statistics.json', {'replicates':10000, 'seed':17, 'resampling_unit':'paired sentence-pair row',
        'method':'Recompute average ranks inside each paired bootstrap resample; percentile intervals.',
        'intervals_are_pointwise_not_simultaneous':True, 'p_values_or_noninferiority_not_claimed':True, 'comparisons':pairs})
    csvout(output / 'paper_comparison.csv', rows)
    lines = ['# KLUE-STS 보류 2,000쌍의 고정 모델 비교', '',
        '| 모델 | Spearman ρ | Pearson r | BGE-M3 대비 유지율(%) | 상대 하락률(%) | ρ 차이의 95% CI |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['spearman']:.4f} | {r['pearson']:.4f} | {r['retention_percent']:.2f} | {r['relative_drop_percent']:.2f} | {r['difference_ci95_low']:+.4f} ~ {r['difference_ci95_high']:+.4f} |")
    lines += ['', '공개 학습 분할에서 보류한 자료의 반복 비교이며 공식 시험 점수나 독립 확증이 아니다. 본 연구의 학습·보존 손실·모델 선정에는 사용하지 않는다. 공개 모델의 사전학습 중복은 보장하지 않는다. 각 신뢰구간은 개별 비교용이며 동시 신뢰구간이나 비열등성 검정이 아니다.', '',
        '기존 Test 정확도는 이번 실험에서 재평가하지 않았다. 519쌍은 추론 재현 검사에만 사용하고 2,000쌍 결과와 합산하지 않는다.']
    (output / 'paper_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    result['status'] = 'COMPLETED'
    save(output / 'summary.json', result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise RuntimeError('Output must be new; historical results are immutable.')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = read(a.bundle / 'protocol.json')
    for rel, want in read(a.bundle / 'bundle_manifest.json').items():
        assert sha(a.bundle / rel) == want, ('Bundle hash mismatch', rel)
    assert read(a.bundle / 'preflight.json')['passed']
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    from scipy.stats import spearmanr
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; no silent CPU fallback.')
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    save(a.output / 'protocol.json', protocol)
    save(a.output / 'environment.json', {'utc_start':datetime.now(timezone.utc).isoformat(), 'python':sys.version,
        'platform':platform.platform(), 'packages':{k:md.version(k) for k in ['torch','transformers','peft','numpy','scipy','huggingface_hub','accelerate','safetensors']},
        'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0), 'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'cudnn':torch.backends.cudnn.version(), 'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True),
        'command':sys.argv, 'precision':'FP32 no autocast, TF32 off; FP64 cosine', 'batch_size':12})
    sts = read(a.bundle / 'klue2000.json')
    ref = read(a.bundle / 'klue519.json')
    assert len(sts) == 2000 and len(ref) == 519
    texts = list(dict.fromkeys(r[k] for r in ref+sts for k in ('sentence1','sentence2')))
    save(a.output / 'sentence_index.json', texts)
    lookup = {s:i for i,s in enumerate(texts)}
    summary = {'status':'IN_PROGRESS', 'fresh_inference':True, 'models':{}, 'n':2000,
        'protocol_sha256':sha(a.bundle / 'protocol.json'), 'interpretation':protocol['interpretation']}
    save(a.output / 'summary.json', summary)
    started = time.time()
    try:
        for key in protocol['execution_order']:
            spec = protocol['models'][key]
            t0 = time.time()
            print('START', key, flush=True)
            allow = ['config.json','tokenizer*','special_tokens_map.json','sentencepiece*','*.model',
                'modules.json','1_Pooling/config.json','README.md','config_sentence_transformers.json','sentence_bert_config.json']
            allow += ['pytorch_model.bin'] if key in ('baseline','l6000') else ['model.safetensors','pytorch_model.bin']
            snap = Path(snapshot_download(spec['repo'], revision=spec['revision'], allow_patterns=allow))
            wf = snap / spec['weight_file']
            assert sha(wf) == spec['weight_sha256'], ('Weight hash mismatch', key)
            for rel,item in spec['metadata'].items():
                assert sha(snap/rel) == item['sha256'], (key, rel)
            tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=False)
            lengths = [len(x) for x in tok(texts,truncation=False)['input_ids']]
            assert max(lengths) <= 160, ('Refuse truncation', key, max(lengths))
            order = sorted(range(len(texts)), key=lambda i:(lengths[i],i))
            model = AutoModel.from_pretrained(snap, local_files_only=True, trust_remote_code=False,
                use_safetensors=wf.suffix=='.safetensors', torch_dtype=torch.float32,
                attn_implementation='sdpa', low_cpu_mem_usage=True)
            if key == 'l6000':
                model = PeftModel.from_pretrained(model, a.bundle/'adapter', is_trainable=False)
            model = model.to('cuda').eval()
            torch.cuda.reset_peak_memory_stats()
            vectors = np.empty((len(texts),1024),dtype=np.float32)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for off in range(0,len(order),12):
                    ix = order[off:off+12]
                    batch = tok([texts[i] for i in ix],padding=True,truncation=False,return_tensors='pt').to('cuda')
                    hidden = model(**batch).last_hidden_state.float()
                    if spec['pooling']=='cls':
                        emb = hidden[:,0]
                    else:
                        mask = batch['attention_mask'].unsqueeze(-1).float()
                        emb = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
                    emb = torch.nn.functional.normalize(emb,p=2,dim=1)
                    assert torch.isfinite(emb).all()
                    vectors[ix] = emb.cpu().numpy()
                    if off % 240 == 0:
                        state={'model':key,'encoded':off+len(ix),'unique_sentences':len(texts),'elapsed_seconds':time.time()-t0}
                        save(a.output/'progress.json',state)
                        print(json.dumps(state),flush=True)
            np.save(a.output/(key+'_embeddings.npy'),vectors)
            v = vectors.astype(np.float64)
            norm = np.linalg.norm(v,axis=1,keepdims=True)
            assert (norm>0).all()
            v /= norm
            def score(data):
                return [{'guid':r['guid'],'gold':float(r['labels']['real-label']),
                    'cosine':float(v[lookup[r['sentence1']]]@v[lookup[r['sentence2']]])} for r in data]
            new_scores, ref_scores = score(sts), score(ref)
            csvout(a.output/(key+'_klue_sts2000.csv'),new_scores)
            csvout(a.output/(key+'_reference519.csv'),ref_scores)
            with (a.bundle/'reference'/f'{key}_klue_sts519.csv').open(encoding='utf-8-sig') as f:
                old = {r['guid']:r for r in csv.DictReader(f)}
            assert set(old)=={r['guid'] for r in ref_scores}
            delta=max(abs(r['cosine']-float(old[r['guid']]['cosine'])) for r in ref_scores)
            gold=[r['gold'] for r in ref_scores]
            rho_delta=float(spearmanr(gold,[r['cosine'] for r in ref_scores]).statistic)-float(spearmanr(gold,[float(old[r['guid']]['cosine']) for r in ref_scores]).statistic)
            gate={'max_cosine_difference':delta,'spearman_difference':rho_delta,'passed':bool(delta<=1e-4 and abs(rho_delta)<=1e-4)}
            summary['models'][key]={'specification':spec,'weight_sha256':sha(wf), 'fresh_inference':True,
                'reference519_reproduction_gate':gate,'max_observed_tokens':max(lengths),
                'inference_seconds':time.time()-t0, 'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
            save(a.output/'summary.json',summary)
            if not gate['passed']:
                raise RuntimeError('Reproduction gate failed; stop before interpreting STS2000: '+key)
            print('COMPLETE',key,flush=True)
            del model, vectors, v, hidden, emb, batch
            gc.collect()
            torch.cuda.empty_cache()
        assert all(sha(a.bundle/rel)==want for rel,want in read(a.bundle/'bundle_manifest.json').items())
        save(a.output/'input_immutability.json',{'passed':True,'files':read(a.bundle/'bundle_manifest.json')})
        summary.update(status='INFERENCE_COMPLETED_PENDING_STATISTICS',elapsed_seconds=time.time()-started)
        save(a.output/'summary.json',summary)
        analyze(a.output,protocol)
        print((a.output/'paper_comparison.md').read_text(encoding='utf-8'),flush=True)
    except BaseException as exc:
        summary.update(status='FAILED_OR_INTERRUPTED',error=str(exc))
        save(a.output/'summary.json',summary)
        (a.output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
