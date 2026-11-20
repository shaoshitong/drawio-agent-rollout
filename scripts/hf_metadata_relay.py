"""Relay HF metadata over an SSH process while storage uploads remain remote."""

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

import requests


PREFIX = "HF_METADATA_RPC "
upstream_send = None


def is_metadata(method, url):
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "huggingface.co"
        and method in ("GET", "HEAD", "POST")
        and (parsed.path.startswith("/api/") or parsed.path.endswith("/info/lfs/objects/batch"))
    )


def rpc_send(self, request, **kwargs):
    if not is_metadata(request.method, request.url):
        return upstream_send(self, request, **kwargs)
    body = request.body or b""
    if isinstance(body, str):
        body = body.encode()
    message = {
        "method": request.method,
        "url": request.url,
        "headers": dict(request.headers),
        "body": base64.b64encode(body).decode(),
    }
    print(PREFIX + json.dumps(message), flush=True)
    reply = json.loads(sys.stdin.readline())
    response = requests.Response()
    response.status_code = reply["status"]
    response.headers.update(reply["headers"])
    response._content = base64.b64decode(reply["body"])
    response._content_consumed = True
    response.request, response.url = request, request.url
    response.encoding = requests.utils.get_encoding_from_headers(response.headers)
    return response


def remote(args):
    global upstream_send
    specification = importlib.util.spec_from_file_location("publisher", args.publisher)
    publisher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(publisher)
    upstream_send = publisher.UploadSession.send
    publisher.UploadSession.send = rpc_send
    if args.check_transport:
        from threading import Lock

        session = publisher.UploadSession(None, False, {}, Lock(), Path("unused"))
        response = session.get("https://huggingface.co/api/models/Qwen/Qwen3.5-4B")
        response.raise_for_status()
        print(json.dumps({"metadata_relay_check": response.json()["id"] == "Qwen/Qwen3.5-4B"}))
        return 0
    sys.argv = [str(args.publisher), *args.command]
    publisher.main()
    return 0


def relay(args):
    token = "" if args.check_transport else sys.stdin.readline()
    session = requests.Session()
    session.trust_env = False
    proxies = {"http": args.metadata_proxy, "https": args.metadata_proxy}
    with subprocess.Popen(args.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        if token:
            child.stdin.write(token)
            child.stdin.flush()
            token = ""
        for line in child.stdout:
            if not line.startswith(PREFIX):
                print(line, end="", flush=True)
                continue
            message = json.loads(line[len(PREFIX) :])
            if not is_metadata(message["method"], message["url"]):
                raise ValueError("Refusing metadata relay outside the HF API")
            response = session.request(
                message["method"],
                message["url"],
                headers=message["headers"],
                data=base64.b64decode(message["body"]),
                proxies=proxies,
                timeout=(20, 180),
                allow_redirects=False,
            )
            reply = {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": base64.b64encode(response.content).decode(),
            }
            child.stdin.write(json.dumps(reply) + "\n")
            child.stdin.flush()
        return child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--publisher", type=Path, default=Path(__file__).with_name("publish_hf.py"))
    parser.add_argument("--metadata-proxy", default="http://127.0.0.1:7896")
    parser.add_argument("--check-transport", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return remote(args) if args.remote else relay(args)


if __name__ == "__main__":
    raise SystemExit(main())
