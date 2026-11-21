"""Copy one verified artifact directly to a peer, receiving its password on stdin."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def quoted_path(path):
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    password = sys.stdin.readline().rstrip("\r\n")
    digest = hashlib.sha256()
    with args.source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != args.source_sha256:
        raise ValueError("Source archive SHA256 differs from its frozen manifest")
    askpass = args.output / "askpass.py"
    askpass.write_text("#!/usr/bin/env python3\nimport os\nprint(os.environ['DRAWIO_SFTP_PASSWORD'])\n")
    askpass.chmod(0o700)
    command = [
        "sftp", "-oBatchMode=no", "-oLogLevel=VERBOSE",
        "-oPreferredAuthentications=password", "-oPubkeyAuthentication=no",
        "-oNumberOfPasswordPrompts=1", "-oConnectTimeout=15",
        "-oServerAliveInterval=15", "-oServerAliveCountMax=3",
        "-oProxyCommand=none", "-oProxyJump=none", "-oForwardAgent=no",
        "-oStrictHostKeyChecking=yes", "-oUserKnownHostsFile=" + str(args.known_hosts), args.host,
    ]
    env = dict(os.environ, DRAWIO_SFTP_PASSWORD=password, SSH_ASKPASS=str(askpass),
               SSH_ASKPASS_REQUIRE="force", DISPLAY=":0")
    password = None
    batch = "put " + quoted_path(args.source) + " " + quoted_path(args.target) + "\nbye\n"
    with (args.output / "transfer.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        env.pop("DRAWIO_SFTP_PASSWORD")
        launch = {"at": time.time(), "pid": process.pid,
                  "start_ticks": Path("/proc", str(process.pid), "stat").read_text().split(") ", 1)[1].split()[19],
                  "source": str(args.source), "source_bytes": args.source.stat().st_size,
                  "source_sha256": digest.hexdigest(), "host": args.host, "target": str(args.target),
                  "credentials": "stdin and process RAM only", "command": command}
        (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        print(json.dumps({key: value for key, value in launch.items() if key != "command"}), flush=True)
        process.communicate(batch.encode())
    (args.output / "exit-code").write_text(str(process.returncode) + "\n")
    result = {"at": time.time(), "transport_exit_code": process.returncode,
              "target_content_verification_required": True}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
