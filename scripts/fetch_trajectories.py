"""Resolve private HF metadata locally; download and verify bulk data remotely."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urljoin, urlsplit

import requests


def resolve(args):
    token = sys.stdin.readline().strip()
    session = requests.Session()
    session.trust_env = False
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else {}
    parts = json.loads(args.plan.read_text())["parts"]
    for part in parts:
        url = f"https://huggingface.co/datasets/{args.repository}/resolve/{args.revision}/{part['path']}"
        while urlsplit(url).hostname == "huggingface.co":
            response = session.head(url, headers={"Authorization": f"Bearer {token}"},
                                    proxies=proxies, allow_redirects=False, timeout=(15, 90))
            if response.status_code not in (301, 302, 303, 307, 308):
                raise RuntimeError(f"HF resolve returned HTTP {response.status_code}")
            url = urljoin(url, response.headers["Location"])
        host = urlsplit(url).hostname or ""
        if not url.startswith("https://") or not host.endswith((".hf.co", ".xethub.hf.co", ".amazonaws.com")):
            raise ValueError("Refusing an unexpected bulk storage host")
        part["download_url"] = url
    print(json.dumps({"revision": args.revision, "parts": parts}))


def download(part, root):
    destination = root / "archives" / (part["extract_dir"] + ".tar.zst")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    if not destination.exists():
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(part["download_url"], headers=headers, stream=True,
                         allow_redirects=False, timeout=(20, 180)) as response:
            if response.status_code not in (200, 206):
                raise RuntimeError(f"Storage download returned HTTP {response.status_code}")
            if offset and response.status_code == 200:
                offset = 0
            reported = offset
            with temporary.open("ab" if offset else "wb") as stream:
                for block in response.iter_content(4 * 1024**2):
                    stream.write(block)
                    offset += len(block)
                    if offset - reported >= 64 * 1024**2:
                        print(json.dumps({"part": part["extract_dir"], "downloaded": offset,
                                          "total": part["bytes"], "at": time.time()}), flush=True)
                        reported = offset
        if temporary.stat().st_size != part["bytes"]:
            raise ValueError("Downloaded archive size mismatch")
        temporary.replace(destination)
    digest = sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != part["sha256"]:
        raise ValueError("Downloaded archive SHA256 mismatch")
    target = root / part["extract_dir"]
    receipt = target / "training-transfer-receipt.json"
    if not receipt.exists():
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(destination),
                        "-C", str(target)], check=True)
        receipt.write_text(json.dumps({"path": part["path"], "sha256": part["sha256"],
                                       "bytes": part["bytes"], "verified_at": time.time()}, indent=2) + "\n")
    print(json.dumps({"part": part["extract_dir"], "verified": True, "extracted": str(target)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("resolve", "download"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repository", default="sst12345/png-to-editable-drawio-20261120")
    parser.add_argument("--revision", default="346e2b7299a91e66c3a8896c712e49dacdaffca7")
    parser.add_argument("--proxy", default="http://127.0.0.1:7896")
    args = parser.parse_args()
    if args.mode == "resolve":
        resolve(args)
    else:
        parts = json.loads(sys.stdin.readline())["parts"]
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda part: download(part, args.root), parts))
        except requests.RequestException as error:
            raise RuntimeError(f"Storage transfer failed: {type(error).__name__}") from None


if __name__ == "__main__":
    main()
