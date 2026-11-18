"""Paired statistics from completed fresh inference; no model selection."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,rankdata

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,data):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def ci(v):return np.quantile(v,[.025,.975]).tolist()
def correlate_ranks(x,y):
    x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True)
    return (x*y).sum(1)/np.sqrt((x*x).sum(1)*(y*y).sum(1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);a=ap.parse_args();out=a.results
    s=read(out/'summary.json');assert s['status']=='INFERENCE_COMPLETED_PENDING_STATISTICS'
    names=list(s['models']);assert len(names)==5
    assert all(s['models'][k]['reproduction_gate']['passed'] for k in ['baseline','l6000'])
    ts={k:rows(out/(k+'_test600.csv')) for k in names};ss={k:rows(out/(k+'_klue_sts519.csv')) for k in names}
    ids=[r['id'] for r in ts[names[0]]];guids=[r['guid'] for r in ss[names[0]]]
    assert all([r['id'] for r in ts[k]]==ids and [r['guid'] for r in ss[k]]==guids for k in names)
    groups={}
    for i,r in enumerate(ts[names[0]]):groups.setdefault((r['domain'],r['grouped_type']),[]).append(i)
    assert len(groups)==50 and all(len(v)==12 for v in groups.values())
    correct=np.array([[int(r['correct']) for r in ts[k]] for k in names])
    gold=np.array([float(r['gold']) for r in ss[names[0]]]);pred=np.array([[float(r['cosine']) for r in ss[k]] for k in names])
    rng=np.random.default_rng(17);boots_acc=np.empty((5,10000));boots_rho=np.empty((5,10000))
    for start in range(0,10000,100):
        ix=np.concatenate([rng.choice(g,size=(100,len(g)),replace=True) for g in groups.values()],axis=1)
        for j in range(5):boots_acc[j,start:start+100]=correct[j,ix].mean(1)
    rng=np.random.default_rng(17)
    for start in range(0,10000,100):
        ix=rng.integers(0,len(gold),size=(100,len(gold)));gr=rankdata(gold[ix],axis=1)
        for j in range(5):boots_rho[j,start:start+100]=correlate_ranks(gr,rankdata(pred[j,ix],axis=1))
    assert np.isfinite(boots_rho).all()
    pairs=[]
    for i,j in itertools.combinations(range(5),2):
        gain=int(((correct[j]==1)&(correct[i]==0)).sum());loss=int(((correct[j]==0)&(correct[i]==1)).sum())
        p=float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.
        pairs.append({'reference':names[i],'comparison':names[j],'new_only_correct':gain,'reference_only_correct':loss,
            'accuracy_difference_pp':float(100*(correct[j].mean()-correct[i].mean())),
            'accuracy_difference_ci95_pp':ci(100*(boots_acc[j]-boots_acc[i])),
            'exact_mcnemar_p':p,'spearman_difference':s['models'][names[j]]['klue_spearman']-s['models'][names[i]]['klue_spearman'],
            'spearman_difference_ci95':ci(boots_rho[j]-boots_rho[i])})
    order=sorted(range(len(pairs)),key=lambda i:pairs[i]['exact_mcnemar_p']);last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,(len(pairs)-rank)*pairs[i]['exact_mcnemar_p']));pairs[i]['holm_p']=last
    save(out/'paired_statistics.json',{'replicates':10000,'seed':17,'intervals':'individual percentile95%; not simultaneous or post-selection adjusted','holm_family':'all ten Test model pairs','pairs':pairs})
    table=[]
    labels={'baseline':'BGE-M3','l6000':'L6000 (6,000건, λ=10)','kure-v1':'nlpai-lab/KURE-v1','bge-m3-ko':'dragonkue/BGE-m3-ko','bge-m3-korean':'upskyy/bge-m3-korean'}
    for j,key in enumerate(names):
        m=s['models'][key];m['accuracy_ci95']=ci(boots_acc[j]);m['spearman_ci95']=ci(boots_rho[j])
        table.append({'model':labels[key],'pooling':m['specification']['pooling'],'correct':m['correct'],'n':600,
          'accuracy_percent':100*m['accuracy'],'macro_percent':100*m['macro_accuracy_10'],'mean_delta':m['mean_delta'],
          'ties':m['ties'],'hn_confusion_count':m['hn_confusion_count'],'spearman':m['klue_spearman'],'pearson':m['klue_pearson'],
          'sts_retention_percent':m['sts_retention_percent'],'sts_relative_drop_percent':m['sts_relative_drop_percent']})
    csvout(out/'paper_comparison.csv',table)
    lines=['# 한국어 BGE-M3 파생 모델 비교 — 새 GPU 추론 결과','',
      '| 모델 | Pooling | Test 정답 | 정확도 / macro (%) | KLUE-STS Spearman | STS 유지율 (%) |',
      '|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['model']} | {r['pooling']} | {r['correct']}/600 | {r['accuracy_percent']:.2f} | {r['spearman']:.6f} | {r['sts_retention_percent']:.2f} |")
    lines+=['','유지율은 같은 실행의 BGE-M3 Spearman 대비 비율이다. 100% 초과는 개선, 100% 미만은 하락을 뜻한다. Test는 유형별60건으로 전체 정확도와 macro가 같다.',
      '', '## L6000과 공개 한국어 모델의 대응 비교']
    for p in pairs:
        if p['reference']=='l6000':
            # Reverse orientation for a clear L6000-minus-Korean comparison.
            ac=p['accuracy_difference_ci95_pp'];sc=p['spearman_difference_ci95']
            lines.append(f"- L6000 − {labels[p['comparison']]}: 정확도 {-p['accuracy_difference_pp']:+.2f}%p (95% CI {-ac[1]:+.2f}~{-ac[0]:+.2f}), McNemar p={p['exact_mcnemar_p']:.6g}, Holm p={p['holm_p']:.6g}; Spearman 차이 {-p['spearman_difference']:+.6f} (95% CI {-sc[1]:+.6f}~{-sc[0]:+.6f}).")
    lines+=['','## 해석 범위','- 기존 Test600 및 KLUE-STS519를 반복 사용한 탐색적 외부 모델 비교이며 독립 확증이 아니다.',
      '- 각 공개 모델의 native pooling을 적용한 전체 시스템 비교다. 가중치나 한국어 추가 학습 한 요인의 인과 효과로 해석하지 않는다.',
      '- 공개 모델의 학습 자료에 해당 벤치마크가 포함되지 않았다는 보장은 없다.',
      '- 원래 L6000 선택은 유지하며 이 결과로 재선정하지 않는다. 신뢰구간은 개별95%이며 기존 반복 선택을 보정하지 않는다.',
      '- 원본 평가 자료와 체크포인트는 변경하지 않았다. 이 결과 및 파생 벡터는 영구 평가 전용이다.']
    (out/'paper_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    s.update(status='COMPLETED',statistics_file='paired_statistics.json');save(out/'summary.json',s)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',manifest);print('ALL_EVALUATION_AND_STATISTICS_COMPLETED',flush=True)
if __name__=='__main__':main()
