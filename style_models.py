"""Local registry and bounded subprocess interface for learned speaking styles.

No tensor libraries are imported into the Flask process. Existing checkpoints
and source recordings are never destinations of this interface.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'research/finnish_style_tts/step12_style_flow/runs'
REGISTRY = ROOT / 'local_state/style_models.json'
SCHEMA = 'zundanen-paired-style-flow-v1'


def registry():
    try:
        rows = json.loads(REGISTRY.read_text(encoding='utf-8'))
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def models():
    result = []
    for row in registry():
        path = (ROOT / row['checkpoint']).resolve()
        if path.is_relative_to(RUNS.resolve()) and path.is_file() and row.get('eligible'):
            result.append({k: row[k] for k in ('id', 'name', 'styles', 'status', 'metrics')})
    return result


def register(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(RUNS.resolve()):
        raise ValueError('Only independent Step12 runs can be registered')
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('schema') not in (SCHEMA,'zundanen-wave-prosody-v1') or not meta.get('eligible'):
        return None
    filename = meta.get('checkpoint','adapter.pt')
    if Path(filename).name != filename:
        raise ValueError('Checkpoint must be a file within the independent run')
    checkpoint = folder / filename
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    decisions = meta.get('style_decisions', {})
    available = [s for s in meta['styles'] if decisions.get(s, {}).get('eligible', True)]
    row = dict(id=digest[:16], name=folder.name, styles=available,
               style_decisions=decisions,
               engine=meta.get('engine','style_flow'),
               checkpoint=checkpoint.relative_to(ROOT).as_posix(), sha256=digest,
               eligible=True, status='experimental_unheard', metrics=meta['metrics'])
    rows = [r for r in registry() if r['id'] != row['id']] + [row]
    previous = next((r for r in registry() if r['id'] == row['id']), None)
    if previous:
        row.update({k: previous[k] for k in ('eligible', 'status', 'listening_note') if k in previous})
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    temp = REGISTRY.with_suffix('.tmp')
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(REGISTRY)
    return row


def record_listening(report, verdict, note):
    if not report.startswith('step12/'):
        return
    folder = (RUNS / report.removeprefix('step12/')).resolve()
    rows = registry()
    changed = False
    for row in rows:
        run = (ROOT / row['checkpoint']).resolve().parent
        if folder == run or folder.is_relative_to(run):
            # Reject removes the whole checkpoint from Voice Generation. A later
            # review cannot silently re-enable it without a positive audition.
            if verdict == 'reject':
                row.update(eligible=False, status='rejected_by_listening')
            elif verdict == 'promising':
                row.update(eligible=True, status='promising_by_listening')
            row['listening_note'] = note
            changed = True
    if changed:
        temp = REGISTRY.with_suffix('.tmp')
        temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(REGISTRY)


def resolve(model_id, style, strength):
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError('Style strength must be finite and between 0 and 1.5')
    style = str(style or 'neutral').lower()
    if style == 'neutral' or strength == 0:
        return None
    row = next((r for r in registry() if r['id'] == model_id and r.get('eligible')), None)
    if row is None or style not in row['styles']:
        raise ValueError('Select an available trained style model and style')
    path = (ROOT / row['checkpoint']).resolve()
    if not path.is_relative_to(RUNS.resolve()) or not path.is_file():
        raise ValueError('Style checkpoint missing')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Style checkpoint changed since registration')
    return row


def apply(source, destination, model_id='', style='neutral', strength=1.):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    row = resolve(model_id, style, strength)
    allowed = (ROOT / 'temp', ROOT / 'outputs', RUNS)
    if not any(destination.is_relative_to(p.resolve()) for p in allowed) or source == destination or destination.exists():
        raise ValueError('Style output must be a separate generated file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if row is None:
        shutil.copyfile(source, destination)
        return dict(style='neutral', bypass=True, samples_unchanged=True)
    python = ROOT / 'research/finnish_style_tts/step10_xespeech/.venv/Scripts/python.exe'
    worker = ROOT / 'research/finnish_style_tts/step12_style_flow' / ('wave_prosody.py' if row.get('engine')=='wave_prosody' else 'infer.py')
    log_path = ROOT / 'temp' / ('style_' + uuid.uuid4().hex + '.log')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(ROOT / 'temp/matplotlib'), MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    cmd = [str(python), '-B', str(worker), '--checkpoint', str(ROOT / row['checkpoint']),
           '--input', str(source), '--output', str(destination), '--style', style,
           '--strength', str(float(strength))]
    with log_path.open('w', encoding='utf-8') as log:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=180, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Style conversion stopped at its 180 second limit. See ' + str(log_path)) from exc
    if proc.returncode or not destination.is_file():
        detail = log_path.read_text(encoding='utf-8', errors='replace')[-3000:]
        raise RuntimeError('Style conversion failed; no styled result was accepted.\n' + detail)
    stats = json.loads(destination.with_suffix('.style.json').read_text(encoding='utf-8'))
    return dict(model=row['id'], style=style, strength=float(strength), status=row['status'], **stats)
