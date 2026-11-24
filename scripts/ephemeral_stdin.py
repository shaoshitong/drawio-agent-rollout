"""Receive RSA-encrypted stdin, then pass plaintext only to a child process pipe."""

import base64
import json
import subprocess
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def main():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.DER,
                                           serialization.PublicFormat.SubjectPublicKeyInfo)
    print(json.dumps({"public_key_spki_base64": base64.b64encode(public).decode()}), flush=True)
    encrypted = base64.b64decode(sys.stdin.readline().strip(), validate=True)
    secret = key.decrypt(encrypted, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                algorithm=hashes.SHA256(), label=None))
    return subprocess.run(sys.argv[1:], input=secret.rstrip(b"\r\n") + b"\n").returncode


if __name__ == "__main__":
    raise SystemExit(main())
