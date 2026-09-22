from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ALIGNER_IMAGE = "lingsoft/aalto-kaldi-align:5.1.1-elg"
ALIGNER_LANGUAGE = "fi"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class AlignmentError(RuntimeError):
    def __init__(self, message: str, code: str = "alignment_failed"):
        super().__init__(message)
        self.code = code


def _creationflags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def find_docker() -> str | None:
    found = shutil.which("docker")
    if found:
        return found

    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates = [
            program_files / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
            program_files / "Docker" / "Docker" / "resources" / "docker.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def _docker(
    docker: str,
    args: list[str],
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [docker, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_creationflags(),
    )


def ensure_docker_ready() -> str:
    docker = find_docker()
    if not docker:
        raise AlignmentError(
            "Docker Desktop is required for Aalto Forced Alignment. "
            "Install Docker Desktop, start it, and try again.",
            "docker_missing",
        )

    try:
        info = _docker(
            docker,
            ["info", "--format", "{{.ServerVersion}}"],
            timeout=20,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise AlignmentError(
            f"Docker could not be checked: {exc}",
            "docker_not_running",
        ) from exc

    if info.returncode != 0:
        detail = (info.stderr or info.stdout or "").strip()
        raise AlignmentError(
            "Docker is installed, but the Docker engine is not running. "
            "Start Docker Desktop and try again."
            + (f"\n{detail[-1200:]}" if detail else ""),
            "docker_not_running",
        )
    return docker


def ensure_image(docker: str) -> bool:
    inspect = _docker(
        docker,
        ["image", "inspect", ALIGNER_IMAGE],
        timeout=30,
    )
    if inspect.returncode == 0:
        return False

    try:
        pull = _docker(
            docker,
            ["pull", ALIGNER_IMAGE],
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise AlignmentError(
            "Downloading the Aalto alignment Docker image timed out.",
            "image_pull_timeout",
        ) from exc

    if pull.returncode != 0:
        detail = (pull.stderr or pull.stdout or "Unknown Docker pull error.").strip()
        raise AlignmentError(
            "Could not download the Aalto alignment Docker image:\n"
            + detail[-4000:],
            "image_pull_failed",
        )
    return True


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(port: int, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except urllib.error.HTTPError:
            # A 404/405 is enough to show that the web service is ready.
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)

    raise AlignmentError(
        "The Aalto alignment service did not become ready in time.",
        "service_start_timeout",
    )


def build_multipart(wav_path: Path, transcript: str) -> tuple[bytes, str]:
    boundary = "----ZundanenBoundary" + uuid.uuid4().hex
    request_json = json.dumps(
        {
            "type": "audio",
            "format": "LINEAR16",
            "sampleRate": 16000,
            "params": {"transcript": transcript},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    audio = wav_path.read_bytes()

    chunks: list[bytes] = []

    def add(value: str | bytes) -> None:
        chunks.append(value.encode("utf-8") if isinstance(value, str) else value)

    add(f"--{boundary}\r\n")
    add('Content-Disposition: form-data; name="request"\r\n')
    add("Content-Type: application/json; charset=utf-8\r\n\r\n")
    add(request_json)
    add("\r\n")

    add(f"--{boundary}\r\n")
    add(
        'Content-Disposition: form-data; name="content"; '
        f'filename="{wav_path.name}"\r\n'
    )
    add("Content-Type: audio/x-wav\r\n\r\n")
    add(audio)
    add("\r\n")
    add(f"--{boundary}--\r\n")

    return b"".join(chunks), boundary


def _format_elg_failure(payload: dict) -> str | None:
    """Return a human-readable ELG failure message when present."""
    failure = payload.get("failure")
    if not isinstance(failure, dict):
        return None

    errors = failure.get("errors")
    if not isinstance(errors, list) or not errors:
        return "The Aalto alignment service reported a failure."

    messages: list[str] = []
    for item in errors:
        if not isinstance(item, dict):
            messages.append(str(item))
            continue

        code = str(item.get("code") or "").strip()
        template = str(item.get("text") or "").strip()
        params = item.get("params")
        if not isinstance(params, list):
            params = []

        rendered = template
        for idx, value in enumerate(params):
            rendered = rendered.replace("{" + str(idx) + "}", str(value))

        if code and rendered:
            messages.append(f"{rendered} [{code}]")
        elif rendered:
            messages.append(rendered)
        elif code:
            messages.append(code)
        else:
            messages.append(json.dumps(item, ensure_ascii=False))

    return "Aalto alignment failed: " + " | ".join(messages)


def _base_phone_symbol(phone: str) -> str:
    """Remove Kaldi word-position suffixes such as _B/_I/_E/_S."""
    return re.sub(r"_[BIES]$", "", phone.strip())


# Aalto's Finnish acoustic model uses a compact phone alphabet rather than
# ordinary Finnish spelling.  These mappings are only for UI/syllable labels;
# the original phone symbol is kept in the phone-level result.
_PHONE_TO_FINNISH = {
    "A": "a",
    "I": "i",
    "U": "u",
    "{": "ä",
    "2": "ö",
    "N": "ŋ",
}
_VOWEL_PHONES = {"A", "I", "U", "e", "o", "y", "{", "2"}
_FINNISH_DIPHTHONGS = {
    "ai", "ei", "oi", "ui", "yi", "äi", "öi",
    "au", "eu", "iu", "ou",
    "ey", "iy", "äy", "öy",
    "ie", "uo", "yö",
}
_FINNISH_VOWELS = set("aeiouyäöAEIOUYÄÖ")


def _source_word_tokens(transcript: str) -> list[str]:
    """Extract source-side word spellings while preserving case/diacritics."""
    # [^\W_] is a Unicode-aware letter/digit class in Python's re module.
    # Keep simple internal hyphens/apostrophes with the word when present.
    return re.findall(r"[^\W_]+(?:[-’'][^\W_]+)*", transcript, flags=re.UNICODE)


def _attach_source_spellings(words: list[dict], transcript: str) -> None:
    """Best-effort mapping from aligner words to the original transcript tokens."""
    source_tokens = _source_word_tokens(transcript)
    if not source_tokens or not words:
        return

    # In the normal case the aligner preserves word count even if G2P changes
    # pronunciation (e.g. Zundamon -> /tsundamon/). Position is then the most
    # reliable way to preserve the user's exact spelling and capitalization.
    if len(source_tokens) == len(words):
        for word, token in zip(words, source_tokens):
            word["display_word"] = token
        return

    # If token counts differ, keep a conservative sequential matcher and leave
    # unmatched items on the aligner's spelling rather than guessing wildly.
    src_i = 0
    for word in words:
        aligned = str(word.get("word", ""))
        aligned_norm = re.sub(r"[^\wäöå]+", "", aligned.casefold(), flags=re.UNICODE)
        match = None
        for lookahead in range(src_i, min(len(source_tokens), src_i + 4)):
            token = source_tokens[lookahead]
            token_norm = re.sub(r"[^\wäöå]+", "", token.casefold(), flags=re.UNICODE)
            if token_norm == aligned_norm:
                match = (lookahead, token)
                break
        if match is not None:
            src_i = match[0] + 1
            word["display_word"] = match[1]
        else:
            word["display_word"] = aligned


def _orthographic_syllables(word: str) -> list[str]:
    """Syllabify a Finnish spelling while preserving the original characters."""
    if not word:
        return []

    chars = list(word)
    nuclei: list[tuple[int, int]] = []
    i = 0
    while i < len(chars):
        if chars[i] not in _FINNISH_VOWELS:
            i += 1
            continue
        start = i
        end = i
        if i + 1 < len(chars) and chars[i + 1] in _FINNISH_VOWELS:
            pair = (chars[i] + chars[i + 1]).casefold()
            if chars[i].casefold() == chars[i + 1].casefold() or pair in _FINNISH_DIPHTHONGS:
                end = i + 1
        nuclei.append((start, end))
        i = end + 1

    if not nuclei:
        return [word]

    starts = [0]
    for nucleus_index in range(1, len(nuclei)):
        prev_end = nuclei[nucleus_index - 1][1]
        next_start = nuclei[nucleus_index][0]
        consonants_between = next_start - prev_end - 1
        if consonants_between <= 0:
            starts.append(next_start)
        else:
            starts.append(next_start - 1)

    result: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(chars)
        piece = "".join(chars[start:end])
        if piece:
            result.append(piece)
    return result


def _syllabify_source_token_for_display(token: str) -> str:
    """Insert visual syllable bars without changing the user's spelling."""
    # Syllabify alphabetic chunks independently so punctuation inside a token
    # (most notably hyphens/apostrophes) is preserved byte-for-byte.
    parts = re.split(r"([-’'])", token)
    rendered: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in {"-", "’", "'"}:
            rendered.append(part)
            continue
        rendered.append(" | ".join(_orthographic_syllables(part)))
    return "".join(rendered)


def syllabify_transcript_for_display(transcript: str) -> str:
    """Return the original transcript with only syllable separators inserted."""
    pattern = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", flags=re.UNICODE)
    return pattern.sub(lambda m: _syllabify_source_token_for_display(m.group(0)), transcript)


def _phone_display(phone: str) -> str:
    return _PHONE_TO_FINNISH.get(phone, phone)


def _phones_share_nucleus(left: str, right: str) -> bool:
    """Return True when two adjacent vowel phones belong to one nucleus."""
    a = _phone_display(left)
    b = _phone_display(right)
    return a == b or (a + b) in _FINNISH_DIPHTHONGS


def _syllabify_phone_items(items: list[dict]) -> list[list[dict]]:
    """Split one word's non-silence phones into Finnish syllable-like groups.

    This is intentionally phonological rather than spelling-length based:
    long vowels and Finnish diphthongs form one nucleus, hiatus starts a new
    nucleus, and between two nuclei the final consonant starts the next
    syllable (e.g. ka-la, kart-ta).
    """
    if not items:
        return []

    nuclei: list[tuple[int, int]] = []
    i = 0
    while i < len(items):
        phone = str(items[i].get("phone", ""))
        if phone not in _VOWEL_PHONES:
            i += 1
            continue

        start = i
        end = i
        # A nucleus may contain a long vowel or one Finnish diphthong.  Do not
        # greedily absorb a third vowel: kauan must be kau-an, not *kauan.
        if i + 1 < len(items):
            nxt = str(items[i + 1].get("phone", ""))
            if nxt in _VOWEL_PHONES and _phones_share_nucleus(phone, nxt):
                end = i + 1
        nuclei.append((start, end))
        i = end + 1

    # Rare acronyms/noise-like tokens may have no vowel in the model.  Keep
    # them visible as one unit rather than dropping timing information.
    if not nuclei:
        return [items]

    starts = [0]
    for nucleus_index in range(1, len(nuclei)):
        prev_end = nuclei[nucleus_index - 1][1]
        next_start = nuclei[nucleus_index][0]
        consonants_between = next_start - prev_end - 1
        if consonants_between <= 0:
            # Hiatus: V.V
            starts.append(next_start)
        else:
            # One consonant -> V.CV.  A cluster -> VC...CV, with the final
            # consonant serving as the onset of the following syllable.
            starts.append(next_start - 1)

    groups: list[list[dict]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(items)
        group = items[start:end]
        if group:
            groups.append(group)
    return groups


def build_syllable_alignment(words: list[dict], phones: list[dict]) -> list[dict]:
    """Attach non-silence phones to aligned words and derive syllable times."""
    if not words or not phones:
        return []

    buckets: list[list[dict]] = [[] for _ in words]
    for phone in phones:
        if phone.get("is_silence"):
            continue
        p_start = float(phone.get("start", 0.0))
        p_end = float(phone.get("end", p_start))
        p_mid = (p_start + p_end) / 2.0

        best_index = None
        best_overlap = -1.0
        best_distance = float("inf")
        for idx, word in enumerate(words):
            w_start = float(word.get("start", 0.0))
            w_end = float(word.get("end", w_start))
            overlap = max(0.0, min(p_end, w_end) - max(p_start, w_start))
            if w_start <= p_mid <= w_end:
                distance = 0.0
            else:
                distance = min(abs(p_mid - w_start), abs(p_mid - w_end))
            if overlap > best_overlap + 1e-9 or (abs(overlap - best_overlap) <= 1e-9 and distance < best_distance):
                best_index = idx
                best_overlap = overlap
                best_distance = distance

        # Because word and phone CTMs are generated from the same alignment,
        # overlap should normally be positive.  The small tolerance only
        # protects against CTM rounding at a word boundary.
        if best_index is not None and (best_overlap > 0.0 or best_distance <= 0.04):
            buckets[best_index].append(phone)

    syllables: list[dict] = []
    for word_index, (word, word_phones) in enumerate(zip(words, buckets)):
        word_phones.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        groups = _syllabify_phone_items(word_phones)
        display_word = str(word.get("display_word") or word.get("word", ""))
        spelling_syllables = _orthographic_syllables(display_word)
        use_spelling = len(spelling_syllables) == len(groups)
        for syllable_index, group in enumerate(groups):
            start = float(group[0]["start"])
            end = float(group[-1]["end"])
            phonetic_label = "".join(_phone_display(str(item.get("phone", ""))) for item in group)
            # Keep timing and grouping from the actual aligned phones, but use
            # the original transcript spelling for the user-facing label when
            # both analyses agree on the syllable count.  This preserves e.g.
            # Zun-da-mon even when Aalto's G2P aligns it as tsun-da-mon.
            label = spelling_syllables[syllable_index] if use_spelling else phonetic_label
            syllables.append({
                "syllable": label,
                "phonetic_syllable": phonetic_label,
                "word": str(word.get("word", "")),
                "display_word": display_word,
                "word_index": word_index,
                "syllable_index": syllable_index,
                "phones": [str(item.get("phone", "")) for item in group],
                "start": round(start, 4),
                "end": round(end, 4),
                "duration": round(max(0.0, end - start), 4),
            })
    return syllables


def parse_alignment_response(payload: dict, transcript: str = "") -> dict:
    if not isinstance(payload, dict):
        raise AlignmentError(
            "The alignment service returned invalid JSON data.",
            "invalid_response",
        )

    failure_message = _format_elg_failure(payload)
    if failure_message:
        raise AlignmentError(failure_message, "service_failure")

    try:
        raw = payload["response"]["annotations"]["forced_alignment"]
    except (KeyError, TypeError) as exc:
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(compact) > 3000:
            compact = compact[:3000] + "…"
        raise AlignmentError(
            "The alignment service returned an unexpected response:\n" + compact,
            "invalid_response",
        ) from exc

    if not isinstance(raw, list):
        raise AlignmentError(
            "The alignment service did not return an alignment list.",
            "invalid_response",
        )

    words: list[dict] = []
    phones: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        features = item.get("features") or {}
        aligned = str(features.get("aligned", "")).strip()
        if not aligned:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue

        interval = {
            "start": round(start, 4),
            "end": round(end, 4),
            "duration": round(max(0.0, end - start), 4),
        }
        if aligned.startswith("__PHONE__:"):
            raw_phone = aligned.split(":", 1)[1].strip()
            if not raw_phone:
                continue
            phone = _base_phone_symbol(raw_phone)
            phones.append(
                {
                    "phone": phone,
                    "raw_phone": raw_phone,
                    "is_silence": phone in {"SIL", "NSN", "SPN"},
                    **interval,
                }
            )
        else:
            words.append({"word": aligned, **interval})

    if not words:
        raise AlignmentError(
            "No aligned words were returned. Try a shorter sentence or check the transcript.",
            "empty_alignment",
        )
    if not phones:
        raise AlignmentError(
            "Word alignment succeeded, but no phone intervals were returned. "
            "The phone extraction patch may not have been installed in the Aalto container.",
            "empty_phone_alignment",
        )
    _attach_source_spellings(words, transcript)
    syllables = build_syllable_alignment(words, phones)
    return {
        "words": words,
        "phones": phones,
        "syllables": syllables,
        "syllabified_transcript": syllabify_transcript_for_display(transcript),
    }


def _normalize_for_aligner(source_wav: Path, output_wav: Path) -> None:
    """Convert arbitrary WAV input to the aligner's expected 16 kHz mono PCM16."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AlignmentError(
            "FFmpeg is required to prepare audio for Aalto Forced Alignment.",
            "ffmpeg_missing",
        )

    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_wav),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            creationflags=_creationflags(),
        )
    except subprocess.TimeoutExpired as exc:
        raise AlignmentError(
            "Audio conversion for Aalto Forced Alignment timed out.",
            "audio_conversion_timeout",
        ) from exc
    except OSError as exc:
        raise AlignmentError(
            f"Could not start FFmpeg for Aalto Forced Alignment: {exc}",
            "ffmpeg_failed",
        ) from exc

    if proc.returncode != 0 or not output_wav.exists():
        detail = (proc.stderr or proc.stdout or "Unknown FFmpeg error.").strip()
        raise AlignmentError(
            "Could not convert the preview WAV to 16 kHz / 16-bit PCM:\n"
            + detail[-4000:],
            "audio_conversion_failed",
        )


def _post_alignment(
    port: int,
    wav_path: Path,
    transcript: str,
    timeout: int,
) -> dict:
    body, boundary = build_multipart(wav_path, transcript)
    url = f"http://127.0.0.1:{port}/process/{ALIGNER_LANGUAGE}"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise AlignmentError(
            f"Aalto alignment request failed (HTTP {exc.code}):\n{detail[-4000:]}",
            "request_failed",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AlignmentError(
            f"Could not communicate with the Aalto alignment service: {exc}",
            "request_failed",
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        preview = raw.decode("utf-8", errors="replace")
        if len(preview) > 3000:
            preview = preview[:3000] + "…"
        raise AlignmentError(
            "The Aalto alignment service returned invalid JSON:\n" + preview,
            "invalid_response",
        ) from exc

    return parse_alignment_response(payload, transcript)



def _install_phone_alignment_patch(docker: str, container_name: str) -> None:
    patch_path = Path(__file__).with_name("aalto_align_with_phones.sh")
    if not patch_path.exists():
        raise AlignmentError(
            f"Phone alignment helper is missing: {patch_path.name}",
            "phone_patch_missing",
        )

    copied = _docker(
        docker,
        [
            "cp",
            str(patch_path),
            f"{container_name}:/opt/kaldi/egs/align/aligning_with_Docker/bin/align.sh",
        ],
        timeout=30,
    )
    if copied.returncode != 0:
        detail = (copied.stderr or copied.stdout or "Unknown Docker copy error.").strip()
        raise AlignmentError(
            "Could not install the phone-level Aalto alignment helper:\n"
            + detail[-3000:],
            "phone_patch_failed",
        )

def run_alignment(
    wav_path: Path,
    transcript: str,
    *,
    request_timeout: int = 330,
) -> dict:
    wav_path = Path(wav_path)
    transcript = str(transcript).strip()

    if not wav_path.exists():
        raise AlignmentError(
            "Generate a Voice Generation preview before running alignment.",
            "preview_missing",
        )
    if not transcript:
        raise AlignmentError(
            "The generated Finnish transcript is missing.",
            "transcript_missing",
        )
    if wav_path.stat().st_size > MAX_AUDIO_BYTES:
        raise AlignmentError(
            "The preview WAV is larger than the aligner's 25 MB request limit.",
            "audio_too_large",
        )

    docker = ensure_docker_ready()
    image_downloaded = ensure_image(docker)
    port = _free_local_port()
    container_name = "zundanen-aalto-" + uuid.uuid4().hex[:10]
    started = False

    try:
        run = _docker(
            docker,
            [
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "--label",
                "com.zundanen.role=aalto-alignment",
                "-p",
                f"127.0.0.1:{port}:8000",
                "--init",
                "--memory=2g",
                "--env",
                "TIMEOUT=300",
                ALIGNER_IMAGE,
            ],
            timeout=90,
        )
        if run.returncode != 0:
            detail = (run.stderr or run.stdout or "Unknown Docker error.").strip()
            raise AlignmentError(
                "Could not start the Aalto alignment container:\n"
                + detail[-4000:],
                "container_start_failed",
            )

        started = True
        _install_phone_alignment_patch(docker, container_name)
        _wait_for_http(port)

        # Chatterbox commonly outputs 24 kHz audio, while this ELG service
        # expects 16 kHz / 16-bit WAV. Normalize only the alignment copy;
        # the original Finnish TTS and final RVC audio remain untouched.
        with tempfile.TemporaryDirectory(prefix="zundanen_align_") as tmp:
            normalized_wav = Path(tmp) / "alignment_16k.wav"
            _normalize_for_aligner(wav_path, normalized_wav)
            alignment = _post_alignment(
                port,
                normalized_wav,
                transcript,
                request_timeout,
            )

        return {
            "words": alignment["words"],
            "phones": alignment["phones"],
            "syllables": alignment["syllables"],
            "syllabified_transcript": alignment["syllabified_transcript"],
            "image": ALIGNER_IMAGE,
            "image_downloaded": image_downloaded,
            "source": "Finnish TTS (before RVC; 16 kHz alignment copy)",
        }
    finally:
        if started:
            try:
                _docker(docker, ["rm", "-f", container_name], timeout=30)
            except Exception:
                pass
