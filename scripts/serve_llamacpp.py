"""Serve the Q8_0 Qwen3.5-4B GGUF and the genre LoRA with llama.cpp's llama-server (CPU or GPU).

    python scripts/serve_llamacpp.py                          # foreground
    python scripts/serve_llamacpp.py --background             # returns once /health answers, log in results/llamacpp.log
    python scripts/serve_llamacpp.py --server-bin ~/llama.cpp/build/bin/llama-server --threads 8

The adapter is loaded with default scale 0 (--lora-scaled FILE:0), so plain requests (the chatbot) use the base model.
Classification requests switch it on per request with "lora": [{"id": 0, "scale": 1.0}] (storybot.classify does this).
Note: --lora FILE --lora-init-without-apply is NOT enough: requests without a "lora" field still got the adapter at
scale 1 in our test (llama.cpp master, 2026-10-07), because the per-request default comes from the listed scale.
llama-server does not batch requests that use different LoRA settings together.
"""
import argparse
import shutil

from storybot.config import PROJECT_ROOT, load_config, resolve
from storybot.deploy.server import launch

if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "configs" / "classify.yaml")
    gdir = resolve(cfg, cfg.gguf.dir)
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", default=shutil.which("llama-server") or "llama-server")
    ap.add_argument("--model", default=str(gdir / cfg.gguf.base_file))
    ap.add_argument("--lora", default=str(gdir / cfg.gguf.lora_file))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx-size", type=int, default=32768, help="total context, shared by the --parallel slots")
    ap.add_argument("--parallel", type=int, default=2, help="concurrent requests (slots)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: llama.cpp picks)")
    ap.add_argument("--n-gpu-layers", default=None, help="e.g. 99 to put everything on a GPU; default CPU build behaviour")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--log", default=str(PROJECT_ROOT / "results" / "llamacpp.log"))
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()                 # anything else goes to llama-server unchanged

    cmd = [a.server_bin, "-m", a.model,
           "--lora-scaled", f"{a.lora}:0",                              # adapter loaded, off unless a request asks
           "--host", "0.0.0.0", "--port", str(a.port),
           "-c", str(a.ctx_size), "--parallel", str(a.parallel),
           "--jinja",                                                  # model's own chat template + tool calls
           "--reasoning", "off",                                       # thinking off, same as vLLM's default
           "--temp", "0"]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    if a.n_gpu_layers is not None:
        cmd += ["--n-gpu-layers", str(a.n_gpu_layers)]
    launch(cmd + extra, a.background, a.log, f"http://localhost:{a.port}/health", a.dry_run)
