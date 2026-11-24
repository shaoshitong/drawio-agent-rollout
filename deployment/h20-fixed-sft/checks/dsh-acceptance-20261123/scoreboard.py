"""Score saved artifacts from one model over the frozen acceptance set."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

B = Path('/DATA/disk1/zbw_tmp4/drawio-agent-sft-20261121')
ACC = B / 'checks/dsh-acceptance-20261123'
THRESH = {"structure_f1": 0.8, "edge_f1": 0.8, "position_within_5pct": 0.8, "color_within_de10": 0.8}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare(a, b, out):
    if (out / 'diff.json').is_file():
        return
    subprocess.run([sys.executable, '-c', 'import sys;from drawio.compare.cli import main;sys.exit(main())',
                    str(a), str(b), '--out', str(out), '--mode', 'reference'],
                   cwd=str(ACC / 'work'), check=True, capture_output=True)


def verify_render(folder, stage, xml_sha):
    png = folder / 'case' / f'round{stage}.png'
    receipt = folder / f'render{stage}.json'
    if not png.is_file() or not receipt.is_file():
        return False
    render = json.loads(receipt.read_text())
    return (render['renderer'].startswith('official_drawio') and not render.get('unresolved')
            and render['source_sha256'] == xml_sha and render['png_sha256'] == digest(png))


def verified_review(folder):
    files = [folder / 'completed.json', folder / 'case/review1.json', folder / 'case/round1.drawio']
    if not all(path.is_file() for path in files):
        return False
    completed, review = (json.loads(path.read_text()) for path in files[:2])
    return (bool(completed.get('completed') and completed.get('reviewer_session'))
            and review.get('status') in ('PASS', 'FIX', 'UNCERTAIN') and isinstance(review.get('issues'), list)
            and verify_render(folder, 1, digest(files[2])))


def score_case(case, folder, stage, work):
    model = folder / 'case' / f'round{stage}.drawio'
    row = {'episode': case['episode'], 'status': 'missing_output',
           'loop_detected': (folder / 'host-loop-outcome.json').is_file(),
           'has_xml': model.is_file(), 'rendered': False, 'passed': False}
    if not row['has_xml']:
        return row
    reference = folder / 'case/reference.png'
    teacher = Path(case['teacher_final_xml'])
    if not reference.is_file() or digest(reference) != case['reference_sha256']:
        return {**row, 'status': 'reference_mismatch'}
    expected, actual = digest(teacher), digest(model)
    if expected != case['teacher_final_xml_sha256']:
        raise ValueError('Frozen teacher XML changed: ' + case['episode'])
    if not verify_render(folder, stage, actual):
        return {**row, 'status': 'unverified_render'}
    reviewed = stage == 2 and verified_review(folder)
    pair = work / f'pair-{expected}-{actual}'
    model_self = work / f'model-self-{actual}'
    row.update(rendered=True, review_verified=reviewed)
    try:
        compare(teacher, model, pair)
        compare(model, model, model_self)
        result = subprocess.run([sys.executable, str(ACC / 'acceptance_score.py'), str(pair),
                                 str(teacher), str(model), str(ACC / 'teacher-self' / case['episode'] / 'diff.json'),
                                 str(model_self / 'diff.json')], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        return {**row, 'status': 'score_failed', 'error': str(error)}
    metrics = json.loads(result.stdout)
    passed = all(metrics[key] >= limit for key, limit in THRESH.items())
    return {**row, **metrics, 'status': 'scored',
            'passed': passed and (stage == 1 or reviewed) and not row['loop_detected']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='model run name under runs/<name>/<episode>/case/roundN.drawio')
    p.add_argument('--round', default='round1', choices=('round1', 'round2'))
    p.add_argument('--evaluation-root', type=Path, help='Existing evaluate_live evaluation/ directory')
    p.add_argument('--only', type=int, default=0)
    a = p.parse_args()
    frozen = ACC / 'acceptance-set.json'
    cases = json.loads(frozen.read_text())['cases']
    configured_cases = len(cases)
    if a.only:
        cases = cases[:a.only]
    if not cases:
        p.error('The selected acceptance set is empty')
    work = ACC / 'scoreboard' / a.model
    work.mkdir(parents=True, exist_ok=True)
    root = a.evaluation_root or ACC / 'runs' / a.model
    rows = [score_case(case, root / case['episode'], int(a.round[-1]), work / a.round / case['episode'])
            for case in cases]
    scored = [r for r in rows if r['status'] == 'scored']
    count = len(rows)
    passed = sum(r['passed'] for r in rows)
    rendered = sum(r['rendered'] for r in rows)
    loops = sum(r['loop_detected'] for r in rows)
    missing = sum(not r['has_xml'] for r in rows)
    summary = {"model": a.model, "round": a.round, "cases": len(rows), "scored": len(scored),
               "fixed_set_sha256": digest(frozen), "whole_fixed_set": count == configured_cases,
               "missing": missing, "rendered": rendered, "detected_loops": loops,
               "render_rate": rendered / count, "missing_xml_rate": missing / count,
               "detected_loop_rate": loops / count, "case_pass": passed, "case_pass_rate": passed / count,
               "mean_structure_f1": sum(r['structure_f1'] for r in scored) / count,
               "thresholds": THRESH, "rows": rows}
    (work / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
