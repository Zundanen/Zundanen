from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


APP_TITLE = "Zundanen"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "app.py").exists():
            return exe_dir
        if (exe_dir.parent / "app.py").exists():
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parent


ROOT = app_dir()
TEMP_DIR = ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)
SERVER_LOG = TEMP_DIR / "desktop_server.log"


def message_box(text: str, title: str = APP_TITLE, error: bool = True) -> None:
    if sys.platform.startswith("win"):
        flags = 0x10 if error else 0x40
        try:
            ctypes.windll.user32.MessageBoxW(None, text, title, flags)
            return
        except Exception:
            pass
    # Useful when launched from a debug console.
    try:
        print(f"{title}: {text}", file=sys.stderr if error else sys.stdout)
    except Exception:
        pass


def webview2_version() -> str | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg
    except ImportError:
        return None

    guid = r"{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    candidates = [
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{guid}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{guid}"),
    ]
    for hive, key_path in candidates:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, "pv")
                value = str(value).strip()
                if value and value != "0.0.0.0":
                    return value
        except OSError:
            pass
    return None


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_until_ready(url: str, process: subprocess.Popen, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/api/state", timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    return False


def log_tail(lines: int = 35) -> str:
    try:
        content = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return ""


def stop_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if sys.platform.startswith("win"):
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=4)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def main() -> int:
    app_py = ROOT / "app.py"
    runtime_python = ROOT / "runtime" / "tts" / ".venv" / "Scripts" / "python.exe"

    if not app_py.exists():
        message_box(f"app.py was not found.\n\nExpected:\n{app_py}")
        return 1
    if not runtime_python.exists():
        message_box(
            "Local runtime was not found.\n\n"
            "Run setup.bat first, then start Zundanen again."
        )
        return 1

    if sys.platform.startswith("win") and not webview2_version():
        message_box(
            "Microsoft Edge WebView2 Runtime was not detected.\n\n"
            "Run setup.bat again. The setup will install WebView2 when it is missing."
        )
        return 1

    try:
        import webview
    except Exception as exc:
        message_box(
            "The desktop UI component is not installed.\n\n"
            "Run install_desktop.bat or setup.bat first.\n\n"
            f"Details: {exc}"
        )
        return 1

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["ZUNDANEN_PORT"] = str(port)
    env["ZUNDANEN_NO_BROWSER"] = "1"
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    server_log_handle = SERVER_LOG.open("w", encoding="utf-8", errors="replace")
    server_process: subprocess.Popen | None = None
    try:
        server_process = subprocess.Popen(
            [str(runtime_python), str(app_py)],
            cwd=str(ROOT),
            env=env,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

        if not wait_until_ready(url, server_process):
            tail = log_tail()
            message_box(
                "Zundanen could not start its local server."
                + (f"\n\n{tail}" if tail else "")
            )
            return 1

        # Required for CSV template and WAV download buttons inside the desktop window.
        webview.settings["ALLOW_DOWNLOADS"] = True

        webview.create_window(
            APP_TITLE,
            url,
            width=1380,
            height=900,
            min_size=(980, 680),
            resizable=True,
            background_color="#090d18",
            text_select=True,
        )

        storage = TEMP_DIR / "webview-profile"
        storage.mkdir(exist_ok=True)
        webview.start(
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(storage),
        )
        return 0
    except Exception as exc:
        message_box(f"Desktop launcher failed.\n\n{exc}\n\n{log_tail()}")
        return 1
    finally:
        stop_process_tree(server_process)
        try:
            server_log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
