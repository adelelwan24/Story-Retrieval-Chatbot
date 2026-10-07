"""Serve Qwen3.5-4B with vLLM: the base model for the chatbot (Task 1) and the genre LoRA (Task 2), one GPU.

    python scripts/serve_vllm.py                      # foreground, Ctrl-C to stop
    python scripts/serve_vllm.py --background         # Colab/notebook: returns once /health answers, log in results/vllm.log
    python scripts/serve_vllm.py --dry-run            # print the command only

Requests pick the weights with the OpenAI `model` field:
    model="Qwen/Qwen3.5-4B"   base model, used by the chatbot (configs/chat.yaml llm.server_model)
    model="genre-lora"        base + adapter, used by storybot.classify (configs/classify.yaml)
"""
import argparse
import json

from storybot.config import PROJECT_ROOT, load_config, resolve
from storybot.deploy.server import launch

if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "configs" / "classify.yaml")
    c = cfg.classifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=c.base_model)
    ap.add_argument("--adapter", default=str(resolve(cfg, c.artifacts_dir) / "adapter"))
    ap.add_argument("--adapter-name", default=c.adapter_name)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="the model supports 262k; 32k covers the agent and keeps KV memory and startup small")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--mtp", action="store_true",
                    help="MTP speculative decoding (faster chat). Off by default: vLLM only documents LoRA + EAGLE3 "
                         "as tested, so compare eval_genre.py with and without it before keeping it")
    ap.add_argument("--no-adapter", action="store_true", help="base model only (the Task 1 before/after comparison)")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--log", default=str(PROJECT_ROOT / "results" / "vllm.log"))
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()                 # anything else is passed to vllm serve unchanged

    cmd = ["vllm", "serve", a.model,
           "--port", str(a.port),
           "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization),
           "--language-model-only",                                   # text only: no vision encoder in memory
           # Task 1: thinking off by default, Qwen3.5 tool calls parsed into OpenAI tool_calls
           "--reasoning-parser", "qwen3",
           "--default-chat-template-kwargs", '{"enable_thinking": false}',
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    if not a.no_adapter:
        # Task 2: one LoRA, rank buffers sized to the adapter (not 64)
        rank = json.loads(open(f"{a.adapter}/adapter_config.json").read())["r"]
        cmd += ["--enable-lora", "--max-loras", "1", "--max-lora-rank", str(rank),
                "--lora-modules", f"{a.adapter_name}={a.adapter}"]
    if a.mtp:
        cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}']
    launch(cmd + extra, a.background, a.log, f"http://localhost:{a.port}/health", a.dry_run)
