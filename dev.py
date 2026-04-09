from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

BACKEND_HOST = os.environ.get("DEEPRESEARCH_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("DEEPRESEARCH_BACKEND_PORT", "8000"))


def _npm_command() -> str:
    # Windows: npm.cmd, others: npm
    if os.name == "nt":
        return shutil.which("npm.cmd") or shutil.which("npm") or "npm.cmd"
    return shutil.which("npm") or "npm"


def _popen(
    args: list[str],
    *,
    cwd: Path,
    name: str,
) -> subprocess.Popen[bytes]:
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        # Put each child in its own process group so we can stop them together.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        start_new_session = True

    return subprocess.Popen(
        args,
        cwd=str(cwd),
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        creationflags=creationflags,
        start_new_session=start_new_session,
        text=False,
    )


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def _terminate(proc: subprocess.Popen[bytes], *, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        # Fallback
        try:
            proc.terminate()
        except Exception:
            pass


def main() -> int:
    if not FRONTEND_DIR.exists():
        print("frontend/ が見つかりません。dev.py はリポジトリ直下で実行してください。", file=sys.stderr)
        return 2

    npm = _npm_command()
    if not shutil.which(npm):
        print("npm が見つかりません。Node.js をインストールして npm に PATH を通してください。", file=sys.stderr)
        return 2

    backend_args = [sys.executable, "-m", "deepresearch.web"]
    frontend_args = [npm, "run", "dev"]

    backend: subprocess.Popen[bytes] | None = None
    if _is_port_available(BACKEND_HOST, BACKEND_PORT):
        print("Starting backend:", " ".join(backend_args))
        backend = _popen(backend_args, cwd=ROOT, name="backend")
        time.sleep(0.4)
    else:
        print(
            f"Backend port {BACKEND_HOST}:{BACKEND_PORT} is already in use; "
            "skipping backend start and assuming an existing backend is running."
        )

    print("Starting frontend:", " ".join(frontend_args))
    frontend = _popen(frontend_args, cwd=FRONTEND_DIR, name="frontend")

    try:
        while True:
            b = backend.poll() if backend else None
            f = frontend.poll()
            if b is not None:
                print(f"[backend exited] code={b}", file=sys.stderr)
                _terminate(frontend, name="frontend")
                break
            if f is not None:
                print(f"[frontend exited] code={f}", file=sys.stderr)
                if backend:
                    _terminate(backend, name="backend")
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
        _terminate(frontend, name="frontend")
        if backend:
            _terminate(backend, name="backend")
    finally:
        # Ensure both are gone.
        procs: list[tuple[subprocess.Popen[bytes], str]] = [(frontend, "frontend")]
        if backend:
            procs.append((backend, "backend"))
        for p, n in procs:
            try:
                p.wait(timeout=8)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

