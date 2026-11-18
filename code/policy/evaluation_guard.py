"""Fail-closed preflight for explicit training inputs. No training occurs here."""
import argparse,csv,hashlib,json,re,sys,zipfile
from collections import Counter,defaultdict
from project_paths import Path
from prepare_final_evaluation import ROOT,canonical,norm
from xlsx_to_csv_stdlib import read_sheet,shared_strings,sheet_paths
from external_sts_guard import load_protection as external_sts_protection, blocked_marker

PROTECTED=(ROOT/'evaluation_only').resolve()
ACTIVE_REGISTRY=PROTECTED/'active_registry.json'
def grams(t):
    t=norm(t);return {t[i:i+4] for i in range(max(0,len(t)-3))}
def dice(a,b):return 2*len(a&b)/(len(a)+len(b)) if a and b else 0
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)
def protected_path(p):
    path=Path(p).resolve()
    roots=[PROTECTED,ROOT/'데이터셋_모음/02_평가후보',ROOT/'데이터셋_모음/03_최종평가전용_승인대기',ROOT/'데이터셋_모음/04_평가분리_2026-10-02']
    return any(path.is_relative_to(root.resolve()) for root in roots)
def load_records(p):
    if p.suffix.lower()=='.csv':return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
    if p.suffix.lower()=='.json':return json.loads(p.read_text(encoding='utf-8-sig'))
    if p.suffix.lower()=='.jsonl':return [json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if p.suffix.lower()=='.xlsx':
        rows=[]
        with zipfile.ZipFile(p) as z:
            for _,path,_ in sheet_paths(z):
                v=read_sheet(z,path,shared_strings(z))
                if v:rows.extend(dict(zip(v[0],r)) for r in v[1:])
        return rows
    raise ValueError('Unsupported training format; refuse unchecked input: '+str(p))
def registry(registry_path=None):
    """Load explicit, hash-pinned reservations; retired permanent versions stay listed.

    The user explicitly released the two legacy 600 versions on 2026-10-02.
    Their archival location does not add their text to the current reservation.
    Raw archival files remain blocked as direct training inputs.
    """
    config_path=Path(registry_path) if registry_path else ACTIVE_REGISTRY
    if not config_path.is_file():raise ValueError('Evaluation exclusion registry is missing; training not authorized')
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    permanent=config.get('permanent_datasets')
    candidates=config.get('evaluation_candidate_datasets')
    if config.get('schema_version')!=1 or not isinstance(permanent,list) or not permanent or not isinstance(candidates,list):
        raise ValueError('Incomplete evaluation exclusion registry; training not authorized')
    versions={entry.get('version') for entry in permanent}
    # A permanent version cannot disappear merely because another version is active.
    for manifest in PROTECTED.glob('*/manifest.json'):
        meta=json.loads(manifest.read_text(encoding='utf-8-sig'))
        if meta.get('permanent_training_prohibition') is True and meta.get('version') not in versions:
            raise ValueError('Permanent evaluation version omitted from registry: '+str(meta.get('version')))
    urls=set();texts=set()
    for entry in permanent+candidates:
        p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to(PROTECTED) or not p.is_file():raise ValueError('Invalid evaluation registry path: '+str(p))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry.get('sha256'):raise ValueError('Evaluation registry hash mismatch: '+str(p))
        obj=json.loads(p.read_text(encoding='utf-8-sig'));records=obj.get('records')
        if not isinstance(records,list) or not records or len(records)!=entry.get('row_count'):
            raise ValueError('Evaluation registry row count mismatch: '+str(p))
        if entry in permanent and obj.get('version')!=entry.get('version'):
            raise ValueError('Evaluation registry version mismatch: '+str(p))
        ids=set()
        for r in records:
            if any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('id','anchor','positive','hard_negative','article_link')):
                raise ValueError('Incomplete evaluation record: '+str(p))
            if r['id'] in ids:raise ValueError('Duplicate evaluation record ID: '+r['id'])
            ids.add(r['id']);url=canonical(r['article_link'])
            if not url:raise ValueError('Invalid evaluation article URL: '+r['id'])
            urls.add(url)
            texts.update(r[k] for k in ('anchor','positive','hard_negative'))
    if not urls or not texts:raise ValueError('Evaluation exclusion registry is empty; training not authorized')
    texts.update(external_sts_protection())
    return urls,texts
def check_values(values,urls,texts):
    protected_norm={norm(t) for t in texts}
    protected_grams=[grams(t) for t in texts]
    # Inverted 4-gram index computes the same set intersections and Dice score
    # as the former all-pairs comparison. Zero-overlap pairs cannot reach .65.
    gram_index=defaultdict(list)
    for i,pg in enumerate(protected_grams):
        for gram in pg:gram_index[gram].append(i)
    issues=[]
    checked={}
    for v in values:
        if v in checked:issues.extend(checked[v]);continue
        found=[]
        if blocked_marker(v):found.append('reserved final external STS dataset/policy')
        if v.strip().casefold() in ('test_only','evaluation_only','final_test','permanent_test_only','evaluation_candidate_only'):found.append('evaluation use-policy marker')
        if v.startswith(('http://','https://')) and canonical(v) in urls:found.append('reserved evaluation article')
        if norm(v) in protected_norm:found.append('exact normalized evaluation sentence')
        elif len(v)>=20:
            g=grams(v)
            intersections=Counter(i for gram in g for i in gram_index.get(gram,()))
            if g and any(2*n/(len(g)+len(protected_grams[i]))>=.65 for i,n in intersections.items()):found.append('near-duplicate evaluation sentence: manual review required')
        checked[v]=found;issues.extend(found)
    return sorted(set(issues))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training-file',type=Path,action='append');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        a='연구기관은 협력 기업에 새로운 기술을 이전하는 계약을 체결했다.'
        assert protected_path(PROTECTED/'x.csv')
        assert not protected_path(ROOT/'작업 데이터셋/train.csv')
        assert 'exact normalized evaluation sentence' in check_values([a],set(),{a})
        assert check_values([a.replace('새로운','혁신적인')],set(),{a})
        assert check_values(['test_only'],set(),set())
        assert check_values(['https://mobile.newsis.com/view/NISX20260715_0003710686'],{'newsis:NISX20260715_0003710686'},set())
        assert not check_values(['평가 문장과 전혀 관련이 없는 독립적인 학습 입력 예시입니다.'],set(),{a})
        print('6 guard tests passed');return
    if not args.training_file:parser.error('Explicit --training-file is mandatory. Recursive file discovery is forbidden.')
    urls,texts=registry();report=[]
    for p in args.training_file:
        p=p.resolve()
        if protected_path(p):report.append({'file':str(p),'issues':['evaluation-only path']});continue
        if not p.is_file():raise ValueError('Input file missing: '+str(p))
        values=list(strings(load_records(p)))
        report.append({'file':str(p),'issues':check_values(values,urls,texts)})
    print(json.dumps({'result':'BLOCKED' if any(r['issues'] for r in report) else 'STRUCTURAL_CHECK_PASSED','files':report,'protected_articles':len(urls),'protected_texts':len(texts),'limitation':'Requires separate event-level review and mandatory training-loader integration; does not prove pretraining absence.'},ensure_ascii=False))
    sys.exit(2 if any(r['issues'] for r in report) else 0)
if __name__=='__main__':main()
