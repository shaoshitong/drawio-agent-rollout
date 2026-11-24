"""Host-side MCP checker. Gold stays in this process, outside agent sandboxes."""

import argparse
import os
from pathlib import Path

from drawio.compiler import compile_bytes
from .mcp_bridge import serve
from .verifier import Verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, required=True, help="Directory containing target.png and candidates"
    )
    parser.add_argument("--gold", type=Path, required=True, help="Host-private reference XML")
    args = parser.parse_args()
    os.umask(0o077)
    verifier = Verifier(args.case.resolve(), compile_bytes(args.gold.read_bytes()), reveal_structure=True)
    serve(
        verifier.check,
        "Check only this case. Gold XML is host-private. Returns candidate render and structural diagnostics.",
    )


if __name__ == "__main__":
    main()
