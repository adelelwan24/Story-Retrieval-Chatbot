"""Start an LLM server (vLLM or llama-server) in the foreground, or in the background with a log file."""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def require_adapter(adapter_dir: str | Path) -> Path:
    """Stop with a clear message unless the LoRA folder has both its config and its weights.

    The weights (adapter_model.safetensors, ~120 MB) are git-ignored, so a fresh clone has no adapter even when
    the folder exists; vLLM then fails late with "... doesn't contain tensors".
    """
    d = Path(adapter_dir)
    weights = [d / "adapter_model.safetensors", d / "adapter_model.bin"]
    if (d / "adapter_config.json").is_file() and any(w.is_file() and w.stat().st_size > 1024 for w in weights):
        return d
    found = sorted(p.name for p in d.iterdir()) if d.is_dir() else []
    raise SystemExit(
        f"LoRA adapter not found in {d}\n"
        f"  present: {found or 'nothing (folder missing)'}\n"
        f"  needed:  adapter_config.json and adapter_model.safetensors\n"
        "The adapter is not in git (models/ and *.safetensors are git-ignored). Unzip the training notebook's\n"
        "genre_lora_qwen35.zip into models/genre_lora_qwen35/, e.g. on Colab:\n"
        "  unzip -o /content/drive/MyDrive/<folder>/genre_lora_qwen35.zip -d models/genre_lora_qwen35")


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
