# One vLLM instance for Task 1 + Task 2 (Qwen3.5-4B + genre LoRA)

Status: design draft for discussion, nothing built yet (2026-10-07).

## 1. The shape

```
                    ┌──────────── vLLM (one process, one GPU) ────────────┐
 chatbot agent ───► │ model="qwen3.5-4b"   → base weights only            │
 (Task 1, tools)    │                                                     │
 classifier    ───► │ model="genre-lora"   → base weights + LoRA deltas   │
 (Task 2)           │  same frozen backbone in VRAM, adapter ≈ tens of MB │
                    └─────────────────────────────────────────────────────┘
```

```bash
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --language-model-only \                 # skip the vision encoder (we only send text)
  --max-model-len 32768 \                 # not 262k: keeps KV cache small
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-lora --max-loras 1 --max-lora-rank 16 \
  --lora-modules genre-lora=/path/to/genre_adapter
```

- Adapter selection is per request: the OpenAI `model` field is either the base name or the adapter name. Mixed batches are fine; vLLM applies the LoRA only to the rows that asked for it.
- Task 1 preservation is structural: base requests never touch the adapter weights. We still prove it (section 5).
- VRAM: bf16 4B text weights ≈ 8-9 GB + KV cache + adapter. Well under 24 GB; also fits a 16 GB card if `--gpu-memory-utilization` and `--max-model-len` are set sensibly.

## 2. The big decision: how the classifier outputs a genre

The current Task 2 notebook uses LoRA + a custom `Linear(hidden, 100)` head on the last token. **vLLM's multi-LoRA path cannot serve that head on a generation instance.** vLLM can serve a classification head (`modules_to_save` named `score`/`classifier`), but only when the model runs as a pooling/classify model, which is a second engine, not the same instance as the chatbot.

| Option | How | Same instance? | Notes |
|---|---|---|---|
| **A. Generative label (recommended)** | SFT the LoRA so the assistant reply is exactly the genre name; at inference use `structured_outputs: {"choice": [100 genres]}`, `max_tokens` ~16, thinking off | Yes | Output is always a valid label. Constrained decoding is greedy token by token, not a full argmax over the 100 labels; for confidence/top-k we can score labels separately (logprobs) if needed. |
| B. Classification head on vLLM | `Qwen3.5` as sequence classifier, `--runner pooling` | No (second process, second copy of weights) | Breaks the "one backbone in VRAM" story. |
| C. Head outside vLLM | vLLM generation API does not return hidden states | No | Would need a separate HF model anyway. |

Label format choice inside A: use the genre names as-is (they carry meaning the model already knows), keep the 100 strings in one `labels.json` shared by training and serving.

## 3. Training the adapter

- Base: `Qwen/Qwen3.5-4B` (the exact checkpoint vLLM serves; the adapter is only valid on that backbone).
- Format: chat template with `enable_thinking=False`, system "Classify the story into one of the genres", user = title + story (truncated, ending kept), assistant = genre name. Loss on the assistant tokens only. Train and serve with the identical template.
- Data is tiny: 1,000 stories, 10 per genre, ~7 per genre in train. Idea: sample 2-3 windows per training story (different 1-2k token slices) as augmentation; evaluate on whole held-out stories. Same split file as Task 1.
- LoRA: r=16, alpha=32, dropout 0.05, lr ~1e-4 to 2e-4, 3-5 epochs, early stop on val macro-F1 (computed with constrained generation, same as serving).
- Precision: plain bf16 LoRA (no QLoRA) on an A100 or L4 (decided 2026-10-07). Training and serving then use the same bf16 base weights, so there is no quantization mismatch.

### Target modules (Qwen3.5 is hybrid, this is where it bites)
Qwen3.5-4B = 32 layers, 8 × (3 × Gated DeltaNet + 1 × Gated Attention), each followed by an FFN.
- Safe set: full-attention `q_proj,k_proj,v_proj,o_proj`, MLP `gate_proj,up_proj,down_proj`, DeltaNet `in_proj_qkv,in_proj_z,out_proj` (both `in_proj_qkv` and `in_proj_z` together).
- Avoid: `in_proj_a`, `in_proj_b`, `conv1d`, anything in the vision tower.
- Known vLLM bugs this avoids:
  - Partial packed group (`in_proj_qkv` without `in_proj_z`) crashes in vLLM 0.21-0.27 (vllm#47639).
  - Adapters saved by PEFT on transformers 5 use `model.layers.*` keys while vLLM's Qwen3.5 class expects `model.language_model.layers.*`; the adapter then loads **silently as a no-op** (whileai-sdk#588). Fix: rename keys after training, and always run the smoke test below.
- Smoke test after training: serve, send 20 val stories with `model="genre-lora"` and with the base; accuracy must jump and outputs must differ. If they're identical, the adapter isn't applied.

## 4. Task 1 agent on vLLM

- `configs/chat.yaml`: `backend: openai`, `server_model: qwen3.5-4b`. `OpenAICompatLLM` already reads OpenAI `tool_calls`, so no client rewrite; vLLM's `qwen3_coder` parser turns Qwen3.5's tool-call format into that.
- Thinking off by default (latency, and our tool loop doesn't need it); can be turned on per request with `chat_template_kwargs`.
- Sampling: keep temperature 0 for reproducible tool calls and eval; Qwen recommends 0.7/top_p 0.8 for non-thinking chat, which we can use for the demo.
- The HF backend's `<tool_call>` JSON regex is Qwen3-specific; it only matters if we keep the local HF path for Qwen3.5.

## 5. Proving Task 1 is not degraded

1. Run the Task 1 eval set (eval/agent_test_cases.json + agent_qa.csv) with `model=qwen3.5-4b` on a server without the adapter, then on the server with the adapter loaded and genre requests in flight. Temperature 0: answers should match (small batch-nondeterminism aside).
2. Contrast: run the same eval with `model=genre-lora`, to show what the adapter would do to chat if it were merged. This is the "why we keep it separate" evidence.
3. Weight hash of the base checkpoint before/after training (unchanged by construction).

## 6. Risks and open questions

1. **GPU (decided 2026-10-07): A100 or L4** for both training and vLLM serving, bf16 throughout. T4 is not a target.
2. **vLLM version pin.** Pick one version (≥0.17 for Qwen3.5) and pin it in requirements; LoRA on Qwen3.5 has had several bugs across releases.
3. **Task 3 (CPU).** llama.cpp `llama-server` can load a LoRA GGUF and switch it per request, which mirrors this design; needs a check that the converter handles a Qwen3.5 LoRA.
4. Generative label vs head accuracy: with 7 examples per class both should be close; we report it rather than assume.
5. The existing Qwen3-4B-2507 Task 2 notebook becomes the "earlier approach"; its head-based results can stay as a baseline comparison in the docs.

Sources: Qwen3.5-4B model card (huggingface.co/Qwen/Qwen3.5-4B), vLLM recipe (recipes.vllm.ai/Qwen/Qwen3.5-4B), vLLM LoRA docs (docs.vllm.ai/en/latest/features/lora.html), vLLM structured outputs docs, vllm issues #38085, #47639, #36275, whileai-sdk#588.
