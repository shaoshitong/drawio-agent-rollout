"""Temporarily serve one artifact through an unguessable download path."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
from threading import Timer


class ArtifactHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != self.server.download_path:
            self.send_error(404)
            return
        with self.server.artifact.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(self.server.artifact.stat().st_size))
            self.end_headers()
            shutil.copyfileobj(stream, self.wfile, 1024**2)

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), ArtifactHandler)
    server.artifact = args.file.resolve()
    server.download_path = "/" + secrets.token_urlsafe(32)
    print(json.dumps({"port": server.server_port, "path": server.download_path}), flush=True)
    timer = Timer(args.timeout, server.shutdown)
    timer.daemon = True
    timer.start()
    try:
        server.serve_forever()
    finally:
        timer.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
