"""Refresh the small manifest index without downloading archived trajectories."""
import json
from pathlib import Path
from server import remote_query

if __name__ == '__main__':
    result = remote_query({'kind': 'manifest'})
    target = Path(__file__).with_name('current-manifest-index.json')
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False))
    temporary.replace(target)
    print(f"Indexed {result['total_episodes']} episodes / {result['total_decisions']} decisions")
