"""Transfer a small checkpoint bundle through a private HF repo using an ephemeral token."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


def metadata_transport():
    import base64
    import httpx
    from hf_metadata_relay import PREFIX, is_metadata

    class MetadataTransport(httpx.HTTPTransport):
        def handle_request(self, request):
            if not is_metadata(request.method, str(request.url)):
                return super().handle_request(request)
            print(PREFIX + json.dumps({"method": request.method, "url": str(request.url),
                                       "headers": dict(request.headers),
                                       "body": base64.b64encode(request.read()).decode()}), flush=True)
            reply = json.loads(sys.stdin.readline())
            headers = {key: value for key, value in reply["headers"].items()
                       if key.lower() not in ("content-encoding", "content-length", "transfer-encoding")}
            return httpx.Response(reply["status"], headers=headers,
                                  content=base64.b64decode(reply["body"]), request=request)

    return httpx.Client(transport=MetadataTransport(), timeout=httpx.Timeout(60, connect=15), follow_redirects=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def parallel_download(args, url, temporary, record):
    command = ["aria2c", "--input-file=-", "--continue=true", "--allow-overwrite=true",
               "--auto-file-renaming=false", "--file-allocation=none", "--enable-color=false",
               "--human-readable=false", "--summary-interval=5", "--connect-timeout=15",
               "--timeout=60", "--max-tries=4", "--retry-wait=2", "--min-split-size=8M",
               "--split=" + str(args.connections), "--max-connection-per-server=" + str(args.connections)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as child:
        record["downloader_pid"] = child.pid
        args.state.write_text(json.dumps(record) + "\n")
        child.stdin.write(url + "\n  dir=" + str(temporary.parent) + "\n  out=" + temporary.name + "\n")
        child.stdin.close()
        for line in child.stdout:
            progress = re.search(r"\[#[0-9a-f]+\s+(\d+)(?:B)?/(\d+)(?:B)?", line)
            if progress:
                record.update(bytes=int(progress[1]), total_bytes=int(progress[2]))
                args.state.write_text(json.dumps(record) + "\n")
        status = child.wait()
    if status:
        record.update(phase="failed", downloader_exit_code=status)
        args.state.write_text(json.dumps(record) + "\n")
        raise RuntimeError("Parallel download failed")


def signed_download(args):
    import requests

    session = requests.Session()
    session.trust_env = False
    url = sys.stdin.readline().strip()
    args.file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.file.with_suffix(args.file.suffix + ".part")
    record = {"phase": "downloading", "bytes": 0, "started_at": time.time()}
    if args.connections > 1:
        record.update(downloader="aria2", connections=args.connections,
                      partial_file_bytes_at_start=temporary.stat().st_size if temporary.exists() else 0)
        args.state.write_text(json.dumps(record) + "\n")
        parallel_download(args, url, temporary, record)
    else:
        with session.get(url, stream=True, timeout=(15, 60)) as response, temporary.open("wb") as stream:
            response.raise_for_status()
            for block in response.iter_content(8 * 1024**2):
                stream.write(block)
                record["bytes"] += len(block)
                args.state.write_text(json.dumps(record) + "\n")
    if digest(temporary) != args.sha256:
        raise ValueError("Signed download checksum differs")
    temporary.replace(args.file)
    record.update(phase="verified", bytes=args.file.stat().st_size,
                  sha256=args.sha256, completed_at=time.time())
    args.state.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def resolve_download(args):
    import requests

    token = sys.stdin.readline().strip()
    session = requests.Session()
    session.trust_env = False
    url = f"https://huggingface.co/datasets/{args.repo}/resolve/{args.revision}/{args.path}"
    response = session.head(url, headers={"Authorization": "Bearer " + token},
                            proxies={"https": args.metadata_proxy}, allow_redirects=False, timeout=(15, 60))
    response.raise_for_status()
    location = response.headers["Location"]
    command = [args.remote_python, args.remote_script, "download-url", "--repo", args.repo,
               "--path", args.path, "--file", str(args.file), "--state", str(args.state), "--sha256", args.sha256,
               "--connections", str(args.connections)]
    result = subprocess.run(["ssh", args.ssh_host, shlex.join(command)], input=location + "\n", text=True)
    if result.returncode:
        raise RuntimeError("Remote signed download failed")


def transfer(args):
    if args.mode == "download-url":
        return signed_download(args)
    if args.mode == "resolve":
        return resolve_download(args)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download

    token = sys.stdin.readline().strip()
    if args.metadata_rpc:
        from huggingface_hub import set_client_factory

        set_client_factory(metadata_transport)
    api = HfApi(endpoint="https://huggingface.co", token=token)
    if not api.repo_info(args.repo, repo_type="dataset").private:
        raise ValueError("Checkpoint destination must be private")
    record = {"phase": args.mode, "started_at": time.time(), "repo": args.repo, "path": args.path}
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(record, indent=2) + "\n")
    if args.mode == "upload":
        if digest(args.file) != args.sha256:
            raise ValueError("Source bundle checksum differs")
        commit = api.upload_file(path_or_fileobj=args.file, path_in_repo=args.path,
                                 repo_id=args.repo, repo_type="dataset",
                                 commit_message="Publish verified SFT artifact " + args.file.name)
        info = api.repo_info(args.repo, repo_type="dataset", revision=commit.oid, files_metadata=True)
        remote = next(item for item in info.siblings if item.rfilename == args.path)
        if not remote.lfs or remote.lfs.sha256 != args.sha256 or remote.size != args.file.stat().st_size:
            raise ValueError("HF checkpoint checksum or size differs")
        record.update(revision=commit.oid, bytes=remote.size)
    else:
        downloaded = Path(hf_hub_download(repo_id=args.repo, filename=args.path, repo_type="dataset",
                                         revision=args.revision, token=token,
                                         local_dir=args.file.parent, endpoint="https://huggingface.co"))
        if digest(downloaded) != args.sha256:
            raise ValueError("Downloaded checkpoint checksum differs")
        if downloaded != args.file:
            downloaded.replace(args.file)
        record.update(revision=args.revision, bytes=args.file.stat().st_size)
    record.update(phase="verified", sha256=args.sha256, completed_at=time.time())
    args.state.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload", "download", "resolve", "download-url"))
    for name in ("repo", "path", "sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("file", "state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--metadata-rpc", action="store_true")
    parser.add_argument("--metadata-proxy", default="http://127.0.0.1:7896")
    parser.add_argument("--ssh-host")
    parser.add_argument("--remote-python")
    parser.add_argument("--remote-script")
    parser.add_argument("--connections", type=int, default=1)
    args = parser.parse_args()
    try:
        transfer(args)
    except Exception as error:
        print(json.dumps({"phase": "failed", "error_type": type(error).__name__,
                          "http_status": getattr(getattr(error, "response", None), "status_code", None)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
