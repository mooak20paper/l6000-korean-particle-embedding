"""Copy only the verified allowlist to a new, empty directory. No Git/network actions."""
from pathlib import Path
import argparse, json, shutil, sys
sys.dont_write_bytecode = True
from verify_release import verify, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dest', type=Path, required=True)
    args = p.parse_args()
    result = verify()
    dest = args.dest.resolve()
    if dest.exists():
        raise RuntimeError('Destination must not exist; preserve prior exports.')
    if dest == ROOT or dest.is_relative_to(ROOT):
        raise RuntimeError('Export outside the source folder.')
    dest.mkdir(parents=True)
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for rel in list(manifest['files']) + ['MANIFEST.json']:
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
    assert not (dest/'.git').exists()
    print(json.dumps({'status':'EXPORTED_NOT_PUBLISHED', 'destination':str(dest), **result}))


if __name__ == '__main__':
    main()
