"""Zundanen Voice Conversion tab: offline RVC, recorded-microphone RVC and in-app realtime RVC.

No model download and no modifications to upstream RVC code.
v2.0.0: WAV conversion accepts a single WAV or a folder (every audio file in it
is converted by one RVC CLI run); a browser recording (WAV uploaded by the page)
is converted with the same settings and can be saved with Save As.
Realtime conversion runs rvc_realtime_worker.py (official RVC realtime pipeline,
no extra window) on the default microphone and speaker.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac'}


def create_voice_conversion_blueprint(*, app_dir: Path, load_config, scan_voices,
                                     run_rvc_on_audio, generation_lock, rvc_env,
                                     kill_process_tree, choose_save_path=None):
    bp = Blueprint('voice_conversion', __name__)
    output_dir = app_dir / 'outputs' / 'voice_conversion'
    output_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = output_dir / 'recordings'
    job_guard = threading.Lock()
    state_guard = threading.Lock()
    job = {'status': 'idle', 'job_id': None, 'error': '', 'output': '', 'audio_url': '', 'kind': '', 'count': 0}
    realtime_guard = threading.Lock()
    rt_process = {'proc': None, 'log': None, 'error': ''}

    def job_snapshot():
        with state_guard:
            return dict(job)

    def set_job(**changes):
        with state_guard:
            job.update(changes)

    def get_voices():
        cfg = load_config()
        voices = scan_voices(cfg)
        # Every voice is loaded from the same configured RVC exports folder.
        return cfg, voices

    def safe(text, n=45):
        return ''.join(c if c.isalnum() or c in '-_' else '_' for c in text)[:n] or 'input'

    def read_settings(data):
        cfg, voices = get_voices()
        name = str(data.get('model', '')).strip()
        if name not in voices:
            raise ValueError('Select an available RVC model.')
        pitch = int(float(data.get('pitch', 0)))
        index_rate = float(data.get('index_rate', 0.1))
        protect = float(data.get('protect', 0.15))
        if not (-36 <= pitch <= 36 and 0 <= index_rate <= 1 and 0 <= protect <= 0.5):
            raise ValueError('Pitch, index rate, or protect is outside the supported range.')
        return cfg, name, pitch, index_rate, protect

    def start_job(kind, source, dest, settings, count):
        """Run one RVC CLI conversion (file -> file or folder -> folder) in the background."""
        cfg, name, pitch, index_rate, protect = settings
        identifier = uuid.uuid4().hex
        set_job(status='running', job_id=identifier, error='', output='', audio_url='', kind=kind,
                count=count, source=str(source), model=name)

        def worker():
            acquired = generation_lock.acquire(blocking=False)
            if not acquired:
                set_job(status='error', error='Another Zundanen generation or validation job is using the audio engine.')
                job_guard.release()
                return
            try:
                run_rvc_on_audio(input_wav=source, output_wav=dest, voice=name, pitch=pitch,
                                 index_rate=index_rate, protect=protect, cfg=cfg)
                if dest.suffix.lower() == '.wav':
                    if not dest.is_file() or dest.stat().st_size < 44:
                        raise RuntimeError('RVC did not create a valid output file.')
                    set_job(status='done', output=str(dest), audio_url='/api/vc/audio/' + dest.name)
                else:
                    made = sum(1 for p in dest.iterdir() if p.is_file()) if dest.is_dir() else 0
                    if made == 0:
                        raise RuntimeError('RVC did not create any output file.')
                    set_job(status='done', output=str(dest), count=made)
            except Exception as exc:
                set_job(status='error', error=str(exc)[-7000:])
                if dest.suffix.lower() == '.wav':
                    try:
                        dest.unlink(missing_ok=True)
                    except OSError:
                        pass
            finally:
                generation_lock.release()
                job_guard.release()

        threading.Thread(target=worker, daemon=True, name='zundanen-vc-' + kind).start()
        return identifier

    def guard_realtime():
        with realtime_guard:
            if rt_status()['running']:
                raise ValueError('Stop realtime RVC before running file conversion to avoid GPU conflicts.')

    @bp.get('/api/vc/models')
    def models():
        try:
            _, voices = get_voices()
            return jsonify(ok=True, models={name: {'pitch': item.get('pitch', 0),
                'index_rate': item.get('index_rate', 0.55),
                'protect': item.get('protect', 0.15)} for name, item in voices.items()})
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 500

    @bp.post('/api/vc/convert')
    def convert():
        """input_wav: a WAV file, or a folder (every audio file in it is converted)."""
        data = request.get_json(silent=True) or {}
        if not job_guard.acquire(blocking=False):
            return jsonify(ok=False, error='An RVC conversion is already running.'), 409
        try:
            guard_realtime()
            settings = read_settings(data)
            source = Path(str(data.get('input_wav', '')).strip().strip('"')).expanduser()
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            token = uuid.uuid4().hex[:8]
            if source.is_dir():
                files = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
                if not files:
                    raise ValueError('The folder contains no WAV / audio files.')
                source = source.resolve()
                # Output always stays under Zundanen's own output folder (originals are never touched).
                dest = output_dir / f'{stamp}_{safe(source.name)}_{safe(settings[1], 30)}'
                kind, count = 'folder', len(files)
            else:
                if not source.is_file() or source.suffix.lower() != '.wav':
                    raise ValueError('Select an existing WAV file or a folder.')
                if source.stat().st_size > 600 * 1024 * 1024:
                    raise ValueError('WAV exceeds the 600 MB single-file limit.')
                source = source.resolve()
                dest = output_dir / f'{stamp}_{safe(source.stem)}_{token}.wav'
                kind, count = 'file', 1
            identifier = start_job(kind, source, dest, settings, count)
            return jsonify(ok=True, job_id=identifier, kind=kind, count=count), 202
        except (ValueError, TypeError, OSError) as exc:
            job_guard.release()
            return jsonify(ok=False, error=str(exc)), 400
        except Exception as exc:
            job_guard.release()
            return jsonify(ok=False, error=str(exc)), 500

    @bp.post('/api/vc/convert-recording')
    def convert_recording():
        """A microphone recording (WAV made by the page) -> RVC with the selected settings."""
        if not job_guard.acquire(blocking=False):
            return jsonify(ok=False, error='An RVC conversion is already running.'), 409
        try:
            guard_realtime()
            settings = read_settings(request.form)
            upload = request.files.get('audio')
            if upload is None:
                raise ValueError('No recording was received.')
            rec_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            source = rec_dir / f'{stamp}_mic.wav'
            upload.save(str(source))
            if source.stat().st_size < 1000:
                raise ValueError('The recording is empty.')
            dest = output_dir / f'{stamp}_mic_{safe(settings[1], 30)}.wav'
            identifier = start_job('recording', source, dest, settings, 1)
            return jsonify(ok=True, job_id=identifier, kind='recording'), 202
        except (ValueError, TypeError, OSError) as exc:
            job_guard.release()
            return jsonify(ok=False, error=str(exc)), 400
        except Exception as exc:
            job_guard.release()
            return jsonify(ok=False, error=str(exc)), 500

    @bp.post('/api/vc/save-as')
    def save_as():
        data = request.get_json(silent=True) or {}
        filename = str(data.get('filename', ''))
        src = output_dir / Path(filename).name
        if filename != Path(filename).name or not src.is_file() or src.suffix.lower() != '.wav':
            return jsonify(ok=False, error='Converted WAV was not found.'), 404
        if choose_save_path is None:
            return jsonify(ok=False, error='Save dialog is not available.'), 400
        try:
            target = choose_save_path(str(data.get('default_name') or src.name))
            if target is None:
                return jsonify(ok=True, cancelled=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            return jsonify(ok=True, path=str(target))
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 500

    @bp.get('/api/vc/status')
    def status():
        return jsonify(ok=True, **job_snapshot())

    @bp.get('/api/vc/audio/<path:filename>')
    def audio(filename):
        # Do not expose unrelated files from outputs or the user's filesystem.
        if filename != Path(filename).name or not filename.lower().endswith('.wav'):
            return jsonify(ok=False, error='Invalid audio filename'), 400
        return send_from_directory(output_dir, filename, mimetype='audio/wav', as_attachment=False)

    @bp.post('/api/vc/open-output')
    def open_output():
        try:
            target = output_dir
            last = job_snapshot().get('output') or ''
            if last and Path(last).is_dir():
                target = Path(last)
            if sys.platform.startswith('win'):
                os.startfile(str(target))
            else:
                subprocess.Popen(['xdg-open', str(target)])
            return jsonify(ok=True)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 500

    def rt_status():
        proc = rt_process['proc']
        running = bool(proc is not None and proc.poll() is None)
        error, ready, info = rt_process['error'], False, {}
        log_path = rt_process['log']
        text = ''
        if log_path and log_path.exists():
            try:
                text = log_path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                text = ''
        for line in text.splitlines():
            if line.startswith('ZUNDANEN_RT_READY '):
                ready = True
                try:
                    info = json.loads(line[len('ZUNDANEN_RT_READY '):])
                except ValueError:
                    info = {}
            elif line.startswith('ZUNDANEN_RT_ERROR '):
                error = line[len('ZUNDANEN_RT_ERROR '):][-1500:]
        if proc is not None and proc.poll() is not None and proc.returncode not in (0, None) and not rt_process.get('stopped'):
            error = error or ('Realtime RVC exited (code %s):\n%s' % (proc.returncode, text[-2500:]))
        return {'running': running, 'ready': running and ready, 'info': info, 'error': error,
                'model': rt_process.get('model', ''), 'log': str(log_path) if log_path else ''}

    @bp.get('/api/vc/realtime/status')
    def realtime_status():
        with realtime_guard:
            return jsonify(ok=True, **rt_status())

    @bp.get('/api/vc/realtime/devices')
    def realtime_devices():
        """Audio devices as seen by the RVC Python (PortAudio), plus the automatic choice."""
        try:
            cfg, _ = get_voices()
            python_path = Path(cfg['rvc_python']).resolve()
            env = rvc_env(Path(cfg['rvc_repo']).resolve())
            env['PYTHONIOENCODING'] = 'utf-8'
            res = subprocess.run([str(python_path), str(app_dir / 'rvc_realtime_worker.py'), '--list-devices'],
                                 capture_output=True, env=env, timeout=60,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if res.returncode != 0:
                raise RuntimeError(res.stderr.decode('utf-8', 'replace')[-1500:])
            return jsonify(ok=True, **json.loads(res.stdout.decode('utf-8')))
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 500

    @bp.post('/api/vc/realtime/start')
    def realtime_start():
        """In-app realtime RVC (rvc_realtime_worker.py): default microphone -> selected model -> default speaker."""
        data = request.get_json(silent=True) or {}
        with realtime_guard:
            if rt_status()['running']:
                return jsonify(ok=True, **rt_status())
            if job_snapshot()['status'] == 'running':
                return jsonify(ok=False, error='Wait for the WAV conversion to finish before starting realtime RVC.'), 409
            try:
                cfg, voices = get_voices()
                name = str(data.get('model', '')).strip()
                if name not in voices:
                    raise ValueError('Select an available RVC model.')
                pitch = int(float(data.get('pitch', 0)))
                index_rate = float(data.get('index_rate', 0.1))
                repo = Path(cfg['rvc_repo']).resolve()
                python_path = Path(cfg['rvc_python']).resolve()
                worker = app_dir / 'rvc_realtime_worker.py'
                if not python_path.is_file():
                    raise RuntimeError('RVC Python environment was not found: ' + str(python_path))
                env = rvc_env(repo)
                def dev(key):
                    v = data.get(key)
                    return None if v in (None, '', 'auto') else int(v)
                env['ZUNDANEN_RT_JSON'] = json.dumps({'rvc_repo': str(repo), 'pth': str(voices[name]['pth']),
                                                      'index': str(voices[name].get('index') or ''),
                                                      'pitch': pitch, 'index_rate': index_rate,
                                                      'input_device': dev('input_device'),
                                                      'output_device': dev('output_device'),
                                                      'mute': bool(data.get('mute', False))})  # mute: test mode
                env['PYTHONIOENCODING'] = 'utf-8'
                rt_log = output_dir / 'realtime_latest.log'
                log_handle = rt_log.open('w', encoding='utf-8')
                try:
                    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform.startswith('win') else 0
                    proc = subprocess.Popen([str(python_path), '-u', str(worker)],
                        cwd=str(repo), env=env, stdin=subprocess.DEVNULL,
                        stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags)
                finally:
                    log_handle.close()
                rt_process.update(proc=proc, log=rt_log, error='', model=name, stopped=False)
                return jsonify(ok=True, **rt_status())
            except Exception as exc:
                rt_process['error'] = str(exc)
                return jsonify(ok=False, error=str(exc)), 400

    @bp.post('/api/vc/realtime/stop')
    def realtime_stop():
        with realtime_guard:
            proc = rt_process['proc']
            if proc and proc.poll() is None:
                try:
                    kill_process_tree(proc)
                except Exception as exc:
                    return jsonify(ok=False, error=str(exc)), 500
            rt_process['proc'] = None
            rt_process['stopped'] = True
            return jsonify(ok=True, running=False)

    return bp
