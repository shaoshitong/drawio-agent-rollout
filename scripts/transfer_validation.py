"""Stream a resumable, hash-verified bundle between SSH hosts without local data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess
import time
from tempfile import TemporaryFile


def command(ssh, program, arguments=()):
    return shlex.split(ssh) + [shlex.join(["python3", "-c", program, *map(str, arguments)])]


def remote(ssh, program, arguments=()):
    return subprocess.check_output(command(ssh, program, arguments), timeout=60)


READ = """from pathlib import Path
import sys,time
p,r=map(Path,sys.argv[1:3]);index,step=map(int,sys.argv[3:5])
with p.open('rb') as source:
 while True:
  source.seek(index*2*1024*1024)
  data=source.read(2*1024*1024)
  if len(data)==2*1024*1024:
   sys.stdout.buffer.write(data);sys.stdout.buffer.flush();index+=step
  elif r.exists():
   if data:sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
   print(r.read_text(),file=sys.stderr,flush=True);break
  else:time.sleep(1)
"""

WRITE = """from pathlib import Path
from hashlib import sha256
import json,sys,time
p=Path(sys.argv[1]);index,step=map(int,sys.argv[2:4]);done=0
while True:
 data=sys.stdin.buffer.read(2*1024*1024)
 if not data:break
 if len(data)<2*1024*1024:
  (p/f'{index:08d}.tail').write_bytes(data);break
 digest=sha256(data).hexdigest();path=p/f'{index:08d}.{digest}.part'
 prior=list(p.glob(f'{index:08d}.*.part'))
 if prior and prior!=[path]:raise ValueError('Source prefix changed')
 temporary=path.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(path)
 index+=step;done+=1
 if done%32==0:print(json.dumps({'stream_fragment':index-step,'at':time.time()}),flush=True)
"""


def relay_once(args, parts, index):
    with TemporaryFile() as errors:
        source = subprocess.Popen(
            command(args.source_ssh, READ, [args.source, args.receipt, index, args.streams]),
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        target = None
        try:
            target = subprocess.Popen(
                command(args.target_ssh, WRITE, [parts, index, args.streams]), stdin=source.stdout
            )
            source.stdout.close()
            target_exit = target.wait()
            if target_exit:
                raise subprocess.CalledProcessError(target_exit, target.args)
            if source.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace")[-2000:])
            errors.seek(0)
            diagnostic = errors.read().decode()
            return json.loads(diagnostic[diagnostic.rfind("{") :])
        finally:
            for process in (source, target):
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)


def relay(args, parts, index):
    pending = """from pathlib import Path
import sys
p=Path(sys.argv[1]);index,step=map(int,sys.argv[2:4])
while any(p.glob(f'{index:08d}.*.part')):index+=step
print(index)
"""
    failures, last_index = 0, index
    while failures < 3:
        try:
            start = int(remote(args.target_ssh, pending, [parts, index, args.streams]))
            if start > last_index:
                failures = 0
            last_index = start
            return relay_once(args, parts, start)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            failures += 1
            print(
                json.dumps(
                    {
                        "stream": index,
                        "consecutive_failures": failures,
                        "error": type(error).__name__,
                        "at": time.time(),
                    }
                ),
                flush=True,
            )
            if failures == 3:
                raise
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-ssh", "target-ssh", "source", "receipt", "target"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--streams", type=int, default=3, choices=range(1, 9))
    args = parser.parse_args()
    parts = args.target + ".parts"
    inventory = """from pathlib import Path
import sys
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);index=0
while True:
 files=list(p.glob(f'{index:08d}.*.part'))
 if not files or files[0].stat().st_size!=2*1024*1024:break
 index+=1
print(index)
"""
    index = int(remote(args.target_ssh, inventory, [parts]))
    with ThreadPoolExecutor(max_workers=args.streams) as pool:
        metadata = list(pool.map(lambda start: relay(args, parts, start), range(index, index + args.streams)))
    if any(item != metadata[0] for item in metadata[1:]):
        raise ValueError("Source receipt changed between streams")
    metadata = metadata[0]
    assemble = """from pathlib import Path
from hashlib import sha256
import sys,json
p,destination=map(Path,sys.argv[1:3]);expected=json.loads(sys.argv[3])
index=expected['bytes']//(2*1024*1024);tail=p/f'{index:08d}.tail'
if expected['bytes']%(2*1024*1024) and tail.exists():
 data=tail.read_bytes();assert len(data)==expected['bytes']%(2*1024*1024)
 tail.replace(p/f'{index:08d}.{sha256(data).hexdigest()}.part')
h=sha256();size=0
with destination.with_suffix('.assembled').open('wb') as out:
 for index,path in enumerate(sorted(p.glob('*.part'))):
  assert int(path.name.split('.')[0])==index
  data=path.read_bytes();assert sha256(data).hexdigest()==path.name.split('.')[1]
  out.write(data);h.update(data);size+=len(data)
assert size==expected['bytes'] and h.hexdigest()==expected['sha256']
destination.with_suffix('.assembled').replace(destination)
destination.with_suffix('.receipt.json').write_text(json.dumps(expected,indent=2)+chr(10))
print(json.dumps({'verified_bytes':size,'sha256':h.hexdigest()}))
"""
    print(remote(args.target_ssh, assemble, [parts, args.target, json.dumps(metadata)]).decode(), flush=True)


if __name__ == "__main__":
    main()
