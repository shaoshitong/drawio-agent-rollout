"""Aggregate acceptance metrics for one model over the frozen 20-case set."""
import argparse, json, subprocess, sys
from pathlib import Path

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
Q = Path('/DATA/disk1/zbw_tmp4/drawio-agent-qwen38')
ACC = B / 'checks/dsh-acceptance-20261123'
REL = B / 'releases/28d8a383ebf711beb256cd1a6ede8d0f8be2dc08'
THRESH = {"structure_f1": 0.8, "edge_f1": 0.8, "position_within_5pct": 0.8, "color_within_de10": 0.8}


def compare(a, b, out):
    if out.exists():
        return
    subprocess.run([sys.executable, '-c', 'import sys;from drawio.compare.cli import main;sys.exit(main())',
                    str(a), str(b), '--out', str(out), '--mode', 'reference'],
                   cwd=str(ACC / 'work'), check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='model run name under runs/<name>/<episode>/case/roundN.drawio')
    p.add_argument('--round', default='round1')
    p.add_argument('--only', type=int, default=0)
    a = p.parse_args()
    cases = json.loads((ACC / 'acceptance-set.json').read_text())['cases']
    if a.only:
        cases = cases[:a.only]
    work = ACC / 'scoreboard' / a.model
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        ep = case['episode']
        model_xml = ACC / 'runs' / a.model / ep / 'case' / f'{a.round}.drawio'
        if not model_xml.exists():
            rows.append({"episode": ep, "status": "missing_output"})
            continue
        pair = work / ep / 'pair'
        mself = work / ep / 'model-self'
        compare(Path(case['teacher_final_xml']), model_xml, pair)
        compare(model_xml, model_xml, mself)
        out = subprocess.run([sys.executable, str(ACC / 'acceptance_score.py'), str(pair),
                              case['teacher_final_xml'], str(model_xml),
                              str(ACC / 'teacher-self' / ep / 'diff.json'), str(mself / 'diff.json')],
                             check=True, capture_output=True, text=True)
        m = json.loads(out.stdout)
        m.update(episode=ep, status="scored",
                 passed=all(m[k] >= v for k, v in THRESH.items()))
        rows.append(m)
    scored = [r for r in rows if r['status'] == 'scored']
    summary = {"model": a.model, "round": a.round, "cases": len(rows), "scored": len(scored),
               "missing": sum(r['status'] == 'missing_output' for r in rows),
               "case_pass": sum(r['passed'] for r in scored),
               "case_pass_rate": (sum(r['passed'] for r in scored) / len(scored)) if scored else 0,
               "mean_structure_f1": (sum(r['structure_f1'] for r in scored) / len(scored)) if scored else 0,
               "thresholds": THRESH, "rows": rows}
    (work / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
