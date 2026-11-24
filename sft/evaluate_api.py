"""Use the same reconstruction workflow with an external DeepSeek generator."""

import argparse
import os
from pathlib import Path
import secrets
import sys

from .evaluate_checkpoint import Provider, Store, gateway, reconstruct, save


def main():
    from setproctitle import setproctitle

    setproctitle("python")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    key = sys.stdin.readline().strip()
    provider = Provider(key)
    provider.balance()
    token = secrets.token_urlsafe(32)
    auth = args.out / "generator-auth.json"
    with os.fdopen(os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        import json

        json.dump({"headers": {"Authorization": "Bearer " + token}}, stream)
    store = Store(args.out / "generator-archive")
    service = gateway(store, provider, args.out.name, "generator", args.out / "case", token, 12, 12)
    args.generator_model, args.port = "deepseek-flash", service.server_port
    args.generator_auth_file, args.generator_service = auth, service
    args.model_path = Path("api/deepseek-flash")
    args.reasoning_effort, args.sampling_preset = "max", "model"
    args.deployment = "External deepseek-flash API; same local DSH/skill/reference/reviewer/budgets"
    save(args.out / "launch.json", {"pid": os.getpid(), "source_commit": args.source_commit,
                                   "generator_model": args.generator_model})
    try:
        reconstruct(args, key)
    finally:
        service.shutdown()
        service.server_close()
        auth.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
