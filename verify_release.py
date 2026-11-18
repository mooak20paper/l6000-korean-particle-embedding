"""Offline integrity check. Does not load a model, train, or run inference."""
from pathlib import Path
import ast, csv, hashlib, json, re

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    allowed = manifest['files']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts}
    expected = set(allowed) | {'MANIFEST.json'}
    if actual != expected:
        raise RuntimeError(f'Unexpected or missing files: {sorted(actual ^ expected)}')
    for rel, spec in allowed.items():
        p = (ROOT / rel).resolve()
        assert p.is_relative_to(ROOT) and not p.is_symlink(), rel
        b = p.read_bytes()
        assert len(b) == spec['bytes'] and hashlib.sha256(b).hexdigest() == spec['sha256'], rel
        assert not any(s in p.parts for s in ('.git', '.aws', '.ssh', '__pycache__')), rel
        if p.suffix in ('.py', '.json', '.csv', '.txt', '.md'):
            text = b.decode('utf-8-sig')
            assert not re.search(r'(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}', text), rel
            assert not re.search(r'[A-Za-z]:[\\/]Users[\\/][^\s]+', text), rel
        if p.suffix == '.py':
            ast.parse(b.decode('utf-8-sig'))
    for model, expected_correct in [('baseline', 261), ('l6000', 570)]:
        p = ROOT / 'results/test519' / (model + '_test600.csv')
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 600 and len({r['id'] for r in rows}) == 600
        assert sum(int(r['correct']) for r in rows) == expected_correct
        assert all(int(r['correct']) == int(float(r['delta']) > 1e-6) for r in rows)
        assert not {'anchor', 'positive', 'hard_negative', 'human_reviewer', 'review_note'} & set(rows[0])
    adapter = ROOT / 'model/adapter_model.safetensors'
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == 'ac321bf43f009a5789f266ecc6be5caf8d1129a9a07eea7f9ff066e34f9f9ee1'
    return {'status': 'PASS', 'files': len(actual), 'bytes': sum((ROOT/p).stat().st_size for p in actual),
            'raw_sentence_data_included': False, 'models_loaded': 0, 'new_experiments': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
