"""Convert Qwen3.5-4B to a Q8_0 GGUF and the genre LoRA to a GGUF adapter, for llama.cpp (Task 3, CPU).

    python scripts/build_gguf.py --llama-cpp ~/llama.cpp
    python scripts/build_gguf.py --llama-cpp ~/llama.cpp --model /path/to/local/Qwen3.5-4B
    python scripts/build_gguf.py --llama-cpp ~/llama.cpp --no-mtp       # extra flags go to convert_hf_to_gguf.py

Needs a llama.cpp checkout (git clone https://github.com/ggml-org/llama.cpp) for its converter scripts, and
`pip install -r <llama.cpp>/requirements/requirements-convert_hf_to_gguf.txt` (or just: pip install gguf sentencepiece
on top of this project's requirements). Writes models/gguf/qwen3.5-4b-q8_0.gguf and models/gguf/genre-lora-f16.gguf.

The adapter stays f16 (about 60 MB): it is small, and quantizing it again on top of the Q8_0 base adds error for no gain.
Only the language model is converted; the vision encoder would be a separate --mmproj file, which we don't need.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from storybot.config import PROJECT_ROOT, load_config, resolve


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


if __name__ == "__main__":
    cfg = load_config(PROJECT_ROOT / "configs" / "classify.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True, type=Path, help="llama.cpp checkout")
    ap.add_argument("--model", default=cfg.classifier.base_model, help="Hub id or local Hugging Face folder")
    ap.add_argument("--adapter", type=Path, default=resolve(cfg, cfg.classifier.artifacts_dir) / "adapter")
    ap.add_argument("--out-dir", type=Path, default=resolve(cfg, cfg.gguf.dir))
    ap.add_argument("--outtype", default="q8_0", help="base model type: q8_0 (default), bf16, f16")
    ap.add_argument("--skip-base", action="store_true", help="only convert the adapter")
    a, extra = ap.parse_known_args()                 # anything else goes to convert_hf_to_gguf.py

    a.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(a.model).expanduser()
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    base_out = a.out_dir / (cfg.gguf.base_file if a.outtype == "q8_0" else f"qwen3.5-4b-{a.outtype}.gguf")
    lora_out = a.out_dir / cfg.gguf.lora_file

    if not a.skip_base:
        run([sys.executable, a.llama_cpp / "convert_hf_to_gguf.py", model_dir, "--outtype", a.outtype, "--outfile", base_out, *extra])
    # The converter maps adapter tensor names with the base model's own rules (including Qwen3.5's
    # Gated DeltaNet head reordering), so it needs the base config.json, not its weights.
    run([sys.executable, a.llama_cpp / "convert_lora_to_gguf.py", a.adapter, "--base", model_dir,
         "--outtype", "f16", "--outfile", lora_out])
    print("wrote", base_out, "and", lora_out)
