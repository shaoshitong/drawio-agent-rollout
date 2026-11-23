"""Upload artifacts to the HF backup dataset through hf-mirror (H20 cannot reach huggingface.co)."""
import os, sys, time
from pathlib import Path

REPO = 'Sutoonq/drawio-sft-20261123'


def token():
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN']
    p = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121/secrets/hf.token')
    return p.read_text().strip()


def main():
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ['HF_TOKEN'] = token()
    from huggingface_hub import HfApi
    api = HfApi()
    items = sys.argv[1:]
    if not items:
        print('usage: backup_hf.py <prefix=path> ...')
        return 1
    for item in items:
        prefix, _, path = item.partition('=')
        p = Path(path)
        t = time.time()
        if p.is_dir():
            api.upload_folder(folder_path=str(p), path_in_repo=prefix, repo_id=REPO, repo_type='dataset')
            print('dir ', prefix, 'from', p, 'in', round(time.time() - t, 1), 's', flush=True)
        elif p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=prefix, repo_id=REPO, repo_type='dataset')
            print('file', prefix, p.stat().st_size, 'bytes in', round(time.time() - t, 1), 's', flush=True)
        else:
            print('skip missing', p, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
