"""Recompute frozen selected-model scores, paired statistics and independent checks."""
from common import *
from collections import defaultdict
import csv,math
import numpy as np
from scipy.stats import spearmanr,pearsonr,rankdata,binomtest
def csvwrite(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def ci(a):return [float(v) for v in np.percentile(a,[2.5,97.5])]
def corrrows(a,b):
 a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
 return (a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
def manual_cos(a,b):
 return math.fsum(float(x)*float(y) for x,y in zip(a,b))/math.sqrt(math.fsum(float(x)**2 for x in a)*math.fsum(float(y)**2 for y in b))
def manual_rank(a):
 order=sorted(range(len(a)),key=lambda i:a[i]);r=[0.]*len(a);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and a[order[j]]==a[order[i]]:j+=1
  for k in order[i:j]:r[k]=(i+1+j)/2
  i=j
 return r
def manual_corr(a,b):
 ma=math.fsum(a)/len(a);mb=math.fsum(b)/len(b)
 return math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(math.fsum((x-ma)**2 for x in a)*math.fsum((y-mb)**2 for y in b))
def summarize(rs):
 return {'n':len(rs),'correct':sum(r['correct'] for r in rs),'ties':sum(r['tie'] for r in rs),'accuracy':float(np.mean([r['correct'] for r in rs])),'mean_cosine_margin':float(np.mean([r['cosine_margin'] for r in rs])),'mean_anchor_positive':float(np.mean([r['cosine_anchor_positive'] for r in rs])),'mean_anchor_hn':float(np.mean([r['cosine_anchor_hn'] for r in rs]))}
def main():
 p=read(H/'protocol.json');verified(p['selection']);verified(p['selection_rule'])
 test=rows(verified(p['test_data']));sts=read(verified(p['klue_data']));index=read(verified(p['sentence_index']))
 sentences=index['sentences'];lookup={s:i for i,s in enumerate(sentences)}
 assert len(test)==600 and len(sts)==519 and len(sentences)==2838
 gold=np.array([r['labels']['real-label'] for r in sts]);rounded=np.array([r['labels']['label'] for r in sts])
 stats={};scores={};preds={};proofs={};allrows=[];grouprows=[];sources={}
 old=read(verified(p['prior_test_summary']))
 for name in p['models']:
  refs=p['reference_files'].get(name)
  if refs:
   raw=np.load(verified(refs['_embeddings.npy']));previous=read(verified(refs['_test_scores.json']))
   rec=read(verified(refs['_inference.json']));sources[name]=refs['_embeddings.npy'];oldname=rec['model']
  else:
   meta=p['model_info'][name];dest=R/meta['directory']
   assert read(dest/'inference_completion.json')['status']=='COMPLETED'
   rec=read(dest/(name+'_inference.json'));assert rec['protocol_sha256']==sha(verified(meta['protocol']))
   raw=np.load(verified(rec['embeddings']));sources[name]=rec['embeddings'];previous=None
  assert raw.shape==(2838,1024) and np.isfinite(raw).all() and np.allclose(np.linalg.norm(raw,axis=1),1,atol=2e-6)
  vectors=raw.astype(np.float64);vectors/=np.linalg.norm(vectors,axis=1)[:,None]
  ss=[];errors=[]
  for i,r in enumerate(test):
   ids=[lookup[r[k]] for k in ['anchor','positive','hard_negative']];a,b,c=[vectors[j] for j in ids]
   ap=float(a@b);an=float(a@c);d=ap-an
   errors.extend([abs(ap-manual_cos(raw[ids[0]],raw[ids[1]])),abs(an-manual_cos(raw[ids[0]],raw[ids[2]]))])
   v={'id':r['id'],'domain':r['domain'],'type':r['grouped_type'],'subtype':r['subtype'],'positive_form':r['positive_form'],'anchor':r['anchor'],'positive':r['positive'],'hard_negative':r['hard_negative'],'article_url':r['article_link'],'cosine_anchor_positive':ap,'cosine_anchor_hn':an,'cosine_margin':d,'correct':bool(d>p['tie_epsilon']),'tie':bool(abs(d)<=p['tie_epsilon'])}
   if previous:
    assert all(v[k]==previous[i][k] for k in ['id','anchor','positive','hard_negative','correct','tie'])
    errors.extend(abs(v[k]-previous[i][k]) for k in ['cosine_anchor_positive','cosine_anchor_hn','cosine_margin'])
   ss.append(v);allrows.append({'model':name,**v})
  pred=np.array([vectors[lookup[r['sentence1']]]@vectors[lookup[r['sentence2']]] for r in sts])
  errors.extend(abs(pred[i]-manual_cos(raw[lookup[r['sentence1']]],raw[lookup[r['sentence2']]])) for i,r in enumerate(sts))
  assert max(errors)<1e-12
  rho=float(spearmanr(gold,pred).statistic);pearson=float(pearsonr(gold,pred).statistic)
  mr=manual_corr(manual_rank(gold.tolist()),manual_rank(pred.tolist()));mp=manual_corr(gold.tolist(),pred.tolist())
  assert abs(mr-rho)<1e-12 and abs(mp-pearson)<1e-12
  groups={axis:{g:summarize([v for v in ss if v[axis]==g]) for g in dict.fromkeys(v[axis] for v in ss)} for axis in ['domain','type','subtype','positive_form']}
  total=summarize(ss);total['macro_accuracy_10']=float(np.mean([v['accuracy'] for v in groups['type'].values()]))
  assert abs(total['accuracy']-total['macro_accuracy_10'])<1e-12
  stats[name]={'test':total,'groups':groups,'klue_sts':{'n':519,'label_field':'labels.real-label','spearman':rho,'pearson':pearson,'rounded_label_sensitivity_spearman':float(spearmanr(rounded,pred).statistic),'rounded_label_sensitivity_pearson':float(pearsonr(rounded,pred).statistic)}}
  if previous:
   assert total['correct']==old['models'][oldname]['test']['correct']
   assert abs(rho-old['models'][oldname]['klue_sts']['spearman'])<1e-12
  else:save(R/p['model_info'][name]['directory']/(name+'_test_scores.json'),ss)
  scores[name]=ss;preds[name]=pred
  proofs[name]={'max_cosine_reconstruction_error':float(max(errors)),'manual_spearman_error':abs(mr-rho),'manual_pearson_error':abs(mp-pearson),'reused_identical_vectors':bool(refs),'test_rows':600,'sts_pairs':519,'embeddings':sources[name]}
  for axis,gg in groups.items():
   for group,rr in gg.items():grouprows.append({'model':name,'axis':axis,'group':group,**rr})
 strata=defaultdict(list)
 for i,r in enumerate(test):strata[r['domain'],r['grouped_type']].append(i)
 assert len(strata)==50 and {len(v) for v in strata.values()}=={12}
 names=p['models'];correct={n:np.array([r['correct'] for r in scores[n]],dtype=np.int8) for n in names}
 acc_boot={n:[] for n in names};rng=np.random.default_rng(17)
 for _ in range(20):
  ix=np.concatenate([rng.choice(ids,(500,len(ids)),replace=True) for ids in strata.values()],axis=1)
  for n in names:acc_boot[n].extend(correct[n][ix].mean(1))
 acc_boot={n:np.array(v) for n,v in acc_boot.items()}
 sts_boot={n:[] for n in names};rng=np.random.default_rng(17)
 for _ in range(40):
  ix=rng.integers(0,519,(250,519));rg=rankdata(gold[ix],axis=1)
  for n in names:sts_boot[n].extend(corrrows(rg,rankdata(preds[n][ix],axis=1)))
 sts_boot={n:np.array(v) for n,v in sts_boot.items()}
 for n in names:
  assert np.isfinite(sts_boot[n]).all()
  st=stats[n]['klue_sts'];base=stats['baseline']['klue_sts']['spearman']
  st.update(baseline_relative_decline_percent=(1-st['spearman']/base)*100,minimum_spearman_at_relative10pct=.9*base,passes_relative10pct_point_estimate=st['spearman']>=.9*base,spearman_95ci=ci(sts_boot[n]),minus_0_9baseline_95ci=ci(sts_boot[n]-.9*sts_boot['baseline']))
  stats[n]['test']['accuracy_95ci']=ci(acc_boot[n])
 chosen=[n for n in names if n!='baseline']
 contrasts=[('baseline',f'C{n}_retention_lambda1_step{n//4}','lambda1_vs_baseline') for n in [2000,4000,6000]]+[(f'C{n}_retention_lambda10_step{n//4}',f'C{n}_retention_lambda1_step{n//4}','matched_size_lambda1_vs_lambda10') for n in [2000,4000,6000]]+[(f'C{x}_retention_lambda1_step{x//4}',f'C{y}_retention_lambda1_step{y//4}','adjacent_lambda1_size') for x,y in [(2000,4000),(4000,6000)]]
 pairs=[]
 for reference,new,kind in contrasts:
  a=correct[reference];b=correct[new];gain=int(((a==0)&(b==1)).sum());loss=int(((a==1)&(b==0)).sum());n=gain+loss
  prob=float(binomtest(gain,n,.5).pvalue) if n else 1.
  exact=min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.
  assert abs(exact-prob)<=max(1e-300,abs(prob)*1e-12)
  pairs.append({'reference':reference,'new_model':new,'contrast_family':kind,'test_new_gain':gain,'test_new_loss':loss,'test_accuracy_difference_pp':100*(b.mean()-a.mean()),'test_difference_95ci_pp':[100*v for v in ci(acc_boot[new]-acc_boot[reference])],'test_exact_mcnemar_two_sided_p':prob,'klue_spearman_difference':stats[new]['klue_sts']['spearman']-stats[reference]['klue_sts']['spearman'],'klue_difference_95ci':ci(sts_boot[new]-sts_boot[reference]),'scope':'Exploratory repeated Test follow-up; CIs unadjusted.'})
 prev=0.
 for rank,idx in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]['test_exact_mcnemar_two_sided_p'])):
  prev=max(prev,min(1.,(len(pairs)-rank)*pairs[idx]['test_exact_mcnemar_two_sided_p']))
  pairs[idx]['test_mcnemar_holm_p']=prev
 summary={'status':'COMPLETED','finished_at_utc':now(),'protocol_sha256':sha(H/'protocol.json'),'selection_sha256':sha(H/'selection.json'),'model_info':p['model_info'],'models':stats,'paired_comparisons':pairs,'test_rows':600,'klue_sts_rows':519,'test_positive_forms':dict((k,v['n']) for k,v in stats['baseline']['groups']['positive_form'].items()),'repeated_test_followup':True,'new_independent_test_evaluated':False,'new_additional_sts_evaluated':False,'test_based_model_reselection':False,'limitations':read(H/'selection_rule.json')['interpretation']}
 save(H/'summary.json',summary)
 save(H/'verification.json',{'status':'PASS','checked_at_utc':now(),'models':proofs,'original_final_selection_unchanged':sha(verified(p['original_final_selection']))==p['original_final_selection']['sha256'],'selection_before_inference':True,'exact_mcnemar_independent_binomial_sum_verified':True,'holm_family_size':len(pairs),'bootstrap_replicates':10000,'bootstrap_seed':17})
 csvwrite(H/'Test600_전체문항점수.csv',allrows);csvwrite(H/'도메인_유형_Positive별_결과.csv',grouprows);csvwrite(H/'Test600_대응비교.csv',pairs)
 csvwrite(H/'KLUE_STS519_문항별점수.csv',[{'guid':r['guid'],'sentence1':r['sentence1'],'sentence2':r['sentence2'],'human_score':gold[i],**{n:float(preds[n][i]) for n in names}} for i,r in enumerate(sts)])
 csvwrite(H/'평가_종합비교.csv',[{'model':n,'train_rows':p['model_info'].get(n,{}).get('train_rows',0),'lambda':p['model_info'].get(n,{}).get('lambda',''),'correct':stats[n]['test']['correct'],'accuracy_percent':100*stats[n]['test']['accuracy'],'macro_accuracy_percent':100*stats[n]['test']['macro_accuracy_10'],'KLUE_Spearman':stats[n]['klue_sts']['spearman'],'KLUE_relative_drop_percent':stats[n]['klue_sts']['baseline_relative_decline_percent'],'KLUE_pass_relative10':stats[n]['klue_sts']['passes_relative10pct_point_estimate']} for n in names])
 np.savez_compressed(H/'paired_bootstrap_distributions.npz',**{'accuracy_'+n:acc_boot[n] for n in names},**{'spearman_'+n:sts_boot[n] for n in names})
 print(json.dumps({'status':'ANALYZED_VERIFIED','models':{n:{'correct':stats[n]['test']['correct'],'sts':stats[n]['klue_sts']['spearman']} for n in names}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
