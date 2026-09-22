from __future__ import annotations

import shutil
from pathlib import Path

MARKER = "ZUNDANEN_RVC_LANGUAGE_OVERRIDE"


def ensure_rvc_language_override(rvc_root: str | Path) -> bool:
    """Make upstream RVC honor RVC_LANGUAGE before the Windows system locale."""
    rvc_root = Path(rvc_root)
    target = rvc_root / "i18n" / "i18n.py"
    if not target.exists():
        raise FileNotFoundError(f"RVC i18n file not found: {target}")

    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        return False

    needle = '    def __init__(self, language=None):\n        if language in ["Auto", None]:\n'
    replacement = (
        '    def __init__(self, language=None):\n'
        '        # ZUNDANEN_RVC_LANGUAGE_OVERRIDE: allow callers to override the Windows locale.\n'
        '        env_language = os.environ.get("RVC_LANGUAGE")\n'
        '        if language in ["Auto", None] and env_language:\n'
        '            language = env_language\n'
        '        elif language in ["Auto", None]:\n'
    )
    if needle not in text:
        raise RuntimeError(
            "Unsupported RVC i18n.py layout. The upstream file changed, so the "
            "Zundanen language patch was not applied."
        )

    backup = target.with_suffix(".py.svs-original")
    if not backup.exists():
        shutil.copy2(target, backup)

    target.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return True
