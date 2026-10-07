"""Start an LLM server (vLLM or llama-server) in the foreground, or in the background with a log file."""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_until_healthy(url: str, proc: subprocess.Popen, log: Path, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-40:])
            raise RuntimeError(f"server exited with code {proc.returncode}; last log lines:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"ready after {time.time() - t0:.0f}s: {url}")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"{url} not healthy after {timeout:.0f}s; see {log}")


def launch(cmd: list[str], background: bool, log: str | Path, health_url: str, dry_run: bool = False):
    """Foreground: runs until the server exits (Ctrl-C). Background: returns the Popen once /health answers."""
    print(shlex.join(cmd))
    if dry_run:
        return None
    if not background:
        sys.exit(subprocess.call(cmd))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    print(f"started PID {proc.pid}, log {log}")
    wait_until_healthy(health_url, proc, log)
    return proc
