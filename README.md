# Story Retrieval Chatbot

Take-home for `FareedKhan/1k_stories_100_genre` (1,000 stories, 100 genres):
Task 1 story chatbot (RAG), Task 2 genre classifier on the same LLM, Task 3 CPU deployment.

Status: data preparation, the Qdrant index, the ReAct story agent (Task 1), the genre LoRA adapter (Task 2) and
serving both from one model (vLLM on GPU, llama.cpp Q8_0 on CPU) are in place. The full run order is in
[Run the whole project](#run-the-whole-project).

## Layout

```
configs/retrieval.yaml      all data/embedding/chunking/Qdrant settings
configs/chat.yaml           LLM backend, agent loop and tool settings (fuzzy thresholds, result counts)
configs/classify.yaml       genre classifier: server URLs, adapter name, label restriction, GGUF file names
src/storybot/
  config.py                 YAML loader; relative paths resolve against this project folder
  data/                     load.py (Hub or local copy), splits.py (7/1/2 stratified, seed 42), prepare.py
  retrieval/                chunking.py, embed.py (ModernBERT late chunking + BM25), qdrant_store.py, indexing.py,
                            search.py (lookups + hybrid search), matching.py (fuzzy title/genre matching)
  chat/                     tools.py (agent tools), agent.py (ReAct loop), llm.py (HF / OpenAI-compatible), prompts.py
  eval/                     agent_eval.py (runs the test cases, checks trace + answer)
  classify/genre.py         Task 2 client: same server as the chatbot, adapter per request, answer restricted to the labels
  deploy/server.py          start vLLM / llama-server in the foreground or background (waits for /health)
scripts/                    prepare_data.py, build_index.py, chat.py, eval_agent.py, make_examples_html.py,
                            serve_vllm.py, build_gguf.py, serve_llamacpp.py, classify_genre.py, eval_genre.py
models/                     genre_lora_qwen35/ (unzipped notebook output) and gguf/ (git-ignored)
eval/                       agent_qa.csv (the 51 test questions with reference answers), agent_test_cases.json
                            (same cases with the checks the scorer runs), stories_sample.csv (the 15 stories they use)
notebooks/02_build_index.ipynb   same steps as the scripts, with a search check at the end
notebooks/03_story_agent.ipynb   loads the collections, runs the agent on 12 example questions via an OpenAI-compatible endpoint
tests/                      offline tests (tiny random ModernBERT, fake stories, toy BM25)
data/                       stories.parquet (git-ignored) and splits.json, created by prepare_data
qdrant_data/                the Qdrant database, created by build_index (git-ignored)
```

## Run the whole project

Everything runs from this folder. Where each step runs:

| Step | Where | Output |
|---|---|---|
| 1-3 setup, data, index | your machine (CPU is fine; a GPU makes indexing faster) | `data/`, `qdrant_data/` |
| 4 train the genre adapter | Colab A100 or L4 | `models/genre_lora_qwen35/` |
| 5A serve on GPU | the GPU machine (Colab A100/L4), vLLM | OpenAI-compatible server on port 8000 |
| 5B serve on CPU (Task 3) | any machine, llama.cpp | OpenAI-compatible server on port 8080 |
| 6-8 chat, classify, evaluate | any machine that can reach the server | answers, `results/` |

Pick 5A or 5B; steps 6-8 are the same for both apart from one flag.

### 1. Set up

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt        # also installs this project (storybot) in editable mode
pytest                                 # offline checks, a few seconds, no model downloads
```

### 2. Prepare the data

```bash
python scripts/prepare_data.py         # data/stories.parquet, data/splits.json, data/story_metadata.json
```

Downloads `FareedKhan/1k_stories_100_genre` once and freezes the 7/1/2 stratified split (seed 42). The training
notebook makes the same split, so Task 1 and Task 2 use the same test stories.

### 3. Build the retrieval index

```bash
python scripts/build_index.py          # qdrant_data/   (--limit 50 for a quick run, --rebuild to redo)
```

Downloads `nomic-ai/modernbert-embed-base` and the BM25 model once. The database folder is `qdrant.path` in
`configs/retrieval.yaml` (default `qdrant_data`); set `qdrant.url` instead to use a Qdrant server.

### 4. Train the genre adapter (Task 2)

Open `../task2-training/task2_genre_lora_qwen35.ipynb` in Colab on an A100 or L4 and run all cells (about 30 minutes
on an A100). It writes `genre_lora_qwen35.zip`. Unzip it so the folder looks like this:

```
models/genre_lora_qwen35/
  adapter/adapter_config.json, adapter/adapter_model.safetensors
  labels.json, prompt.json, results.json, training_curves.png, confusion_matrix.png
```

The adapter is not in git (`models/` and `*.safetensors` are git-ignored), so a fresh clone, for example on Colab,
needs this unzip again: `unzip -o /content/drive/MyDrive/<folder>/genre_lora_qwen35.zip -d models/genre_lora_qwen35`.
`serve_vllm.py` and `build_gguf.py` stop with this hint when `adapter_model.safetensors` is missing.

### 5A. Serve both tasks on a GPU with vLLM

On the GPU machine, with this folder and `models/genre_lora_qwen35/` on it:

```bash
pip install -r requirements.txt "vllm>=0.17"           # Qwen3.5 needs vLLM 0.17 or newer
python scripts/serve_vllm.py --background              # waits until the server answers; log in results/vllm.log
```

One server, port 8000: `model="Qwen/Qwen3.5-4B"` is the chatbot's base model, `model="genre-lora"` is the
classifier. `python scripts/serve_vllm.py --dry-run` prints the full `vllm serve` command; `--mtp` turns on MTP
speculative decoding (check `eval_genre.py` results with and without it). Without `--background` it runs in the
foreground until Ctrl-C.

Point the chatbot and the classifier at it (bash; in PowerShell use `$env:NAME = "value"`):

```bash
export LLM_BASE_URL=http://localhost:8000/v1           # chatbot (configs/chat.yaml llm.base_url)
export LLM_MODEL=Qwen/Qwen3.5-4B
export GENRE_BASE_URL=http://localhost:8000/v1         # classifier (configs/classify.yaml)
```

If the server runs on Colab and the chatbot on your machine, use the tunnel URL (e.g. ngrok or cloudflared) in
both variables, and `LLM_API_KEY` if the server needs a key.

### 5B. Serve both tasks on a CPU with llama.cpp (Task 3)

Once, build llama.cpp and convert the model to Q8_0 and the adapter to GGUF:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build -j --target llama-server
pip install -e ~/llama.cpp/gguf-py sentencepiece

python scripts/build_gguf.py --llama-cpp ~/llama.cpp   # models/gguf/qwen3.5-4b-q8_0.gguf + genre-lora-f16.gguf
```

On Windows you can download a prebuilt llama.cpp release instead of building it; the clone is still needed for the
converter scripts.

Then start the server:

```bash
python scripts/serve_llamacpp.py --background --server-bin ~/llama.cpp/build/bin/llama-server --threads 8
export LLM_BASE_URL=http://localhost:8080/v1
export GENRE_BASE_URL=http://localhost:8080/v1
```

The adapter is loaded switched off, so the chatbot gets the base model; the classifier switches it on per request.
Add `--backend llamacpp` to the classifier commands below.

### 6. Chat with the story agent (Task 1)

```bash
python scripts/chat.py --backend openai --trace        # interactive; 'reset' clears the conversation
python scripts/chat.py --backend openai -q "What is the story with ID 523 about?"
```

`notebooks/03_story_agent.ipynb` runs 12 example questions the same way.

### 7. Classify a story (Task 2)

```bash
python scripts/classify_genre.py --id 523                       # shows the true genre too
python scripts/classify_genre.py --file my_story.txt --title "The Last Signal"
python scripts/classify_genre.py --id 523 --base                # same model, adapter off
```

### 8. Evaluate

```bash
python scripts/eval_genre.py                    # Task 2 on the test split -> results/genre_eval_vllm.json
python scripts/eval_agent.py --backend openai   # Task 1 on the 51 test cases -> results/agent_eval.json
python scripts/make_examples_html.py --backend openai --per-type 2   # results/agent_examples.html
```

`eval_genre.py` reports accuracy, macro-F1, invalid answers and seconds per story for three runs (adapter +
restriction, adapter unrestricted, base model + restriction) and checks that the restriction holds.

Task 1 is not changed by the adapter: chatbot requests never use it (`model="Qwen/Qwen3.5-4B"` on vLLM, scale 0 on
llama.cpp). To show it, run `eval_agent.py` once against a server started with
`python scripts/serve_vllm.py --background --no-adapter` and once against the normal server (save the first
`results/agent_eval.json` under another name before the second run), then compare the two.

## Retrieval design (index side)

`docs/data_pipeline.html` walks through every step from loading the dataset to the finished index (open it in a browser).


| Choice | Value | Why |
|---|---|---|
| Embedder | `nomic-ai/modernbert-embed-base` (768-d, 8k context) | long context for late chunking, no `trust_remote_code` |
| Prefixes | `search_document: ` / `search_query: ` | required by the model; dense only, BM25 never sees them |
| Chunks | ~400 tokens, ~50 overlap, sentence-aligned; stories ≤ 1,000 tokens stay whole | answer sentences stay intact; character spans make late chunking exact |
| `dense_late` | whole story in one pass, mean-pool token vectors per chunk (token midpoint inside span) | each chunk vector sees the whole story's context |
| `dense_prefix` | each chunk alone, `max_length` 1,100 | baseline for the A/B; long enough that whole-story chunks are not cut at 512 |
| `sparse` | FastEmbed `Qdrant/bm25`, Qdrant IDF modifier | exact names and rare words |
| Storage | Qdrant local on-disk mode in `qdrant_data/` | no server to run; built once, reused by the chatbot and the CPU deployment |

Notes:
* Local mode allows one client per folder per process; use `storybot.retrieval.open_client(cfg)`, which reuses it.
* Local mode ignores payload indexes and filters by scanning. That is fine for ~1k stories; the indexes are still created so a Qdrant server gets them.

## Story agent (Task 1)

A ReAct loop (`storybot/chat/agent.py`): the LLM writes a short `Thought:`, calls a tool with Qwen3's native
tool calling, reads the JSON observation, and repeats (at most `agent.max_steps` rounds) until it answers.
Full stories fetched by id or title come back in `AgentResult.stories`, so the UI prints them verbatim
instead of having the LLM retype them.

| Tool | What it does |
|---|---|
| `get_story_by_id(story_id)` | full story from the `stories` collection |
| `get_story_by_title(title)` | fuzzy title match against `data/story_metadata.json`, then an exact payload filter on the original title; no match above `title_threshold` -> semantic search results instead |
| `search_stories(query, genre?, num_stories=6)` | hybrid RRF search grouped by `story_id`, so results are distinct stories; the genre is fuzzy-matched and used as a filter, or dropped (with a note) when nothing matches |
| `ask_about_story(question, story_id? / title?)` | passages from ONE story only (the whole story when it is short) |
| `summarize_story(story_id? / title?, length)` | LLM summary; long stories are summarised part by part, then merged |
| `list_genres()` / `list_stories_by_genre(genre)` | browsing; the genre is fuzzy-matched |

Fuzzy matching uses the same normalisation as `save_story_metadata` (`_normalize_title`) and scores with
`max(ratio, token_sort_ratio)` from RapidFuzz; partial-substring scorers are avoided so a short query does not
grab an unrelated long title. Thresholds live in `configs/chat.yaml`.

LLM backends (`llm.backend` in `configs/chat.yaml`):
* `hf`: `Qwen/Qwen3-4B-Instruct-2507` through transformers, 4-bit NF4 on a CUDA GPU (the earlier local setup).
* `openai`: any OpenAI-compatible server; the current setup is `Qwen/Qwen3.5-4B` on vLLM or llama.cpp (step 5A/5B).
  Set `LLM_API_KEY` for the key. HTTPS certificates are checked against the OS certificate store (`truststore`), so a
  company proxy trusted by Windows/macOS works as is. Otherwise set `llm.ca_bundle` or `LLM_CA_BUNDLE` to the CA's PEM file.

## Agent test cases

`eval/agent_test_cases.json` holds 51 cases written from the 15 stories in `eval/stories_sample.csv`, one group per
behaviour: lookups by id and by title (exact, typo, partial, missing), theme searches (6+ distinct stories),
genre filters with typos and a missing genre, finding one story from a description, questions about one story
(including names shared across stories: Sarah, Jack, Victor), a question the story cannot answer, summaries,
follow-ups, browsing and out-of-scope questions. Each case states what the trace and the answer must show.

```bash
python scripts/eval_agent.py --backend openai                 # all cases -> results/agent_eval.json
python scripts/eval_agent.py --backend openai --ids ask-15 ask-16
```

A case passes when all its checks pass. "Expected story found" is reported separately: on the full index,
other stories can rightly outrank the sample ones in a theme search. The "says not found" check is a keyword heuristic.

## Examples page

```bash
python scripts/make_examples_html.py --base-url https://your-endpoint/v1 --model your-model --per-type 2
python scripts/make_examples_html.py --backend openai                # all 51 test cases
```

Runs the questions from `eval/agent_test_cases.json` through the agent and writes `results/agent_examples.html`:
one tab per example type, and for each example the question, every Thought / Action / Observation step, the
answer (trimmed to 500 words, `--max-words`) and the expected answer. The raw traces go to
`results/agent_examples_runs.json`; `--from-json` re-renders the page from them without calling the LLM.
Step-by-step instructions (Windows): `docs/run_examples.md`.

## Serving Task 1 + Task 2 from one model

One `Qwen/Qwen3.5-4B` serves both tasks. The chatbot uses the base model; the genre classifier uses the same
weights plus the LoRA adapter from `task2-training/task2_genre_lora_qwen35.ipynb`, switched on per request.
Unzip the notebook's `genre_lora_qwen35.zip` into `models/genre_lora_qwen35/` first.

| | GPU (vLLM) | CPU (llama.cpp, Q8_0) |
|---|---|---|
| start | `python scripts/serve_vllm.py --background` | `python scripts/build_gguf.py --llama-cpp ~/llama.cpp` once, then `python scripts/serve_llamacpp.py --background` |
| chatbot request | `model="Qwen/Qwen3.5-4B"` | no `lora` field (adapter loaded with default scale 0) |
| classifier request | `model="genre-lora"` | `"lora": [{"id": 0, "scale": 1.0}]` |
| answer restricted to the labels by | `structured_outputs: {"choice": labels}` | GBNF `grammar: root ::= "Adventure" \| ...` |

Both restrictions turn `labels.json` into a grammar and mask every token that would leave it, so the answer is always
exactly one genre. The prompt (`prompt.json`) is the one used in training: the genre list in the system prompt, the
story cut to its first 2048 tokens with the server's own tokenizer, thinking off.

```python
from storybot.classify import build_classifier
from storybot.config import PROJECT_ROOT, load_config
clf = build_classifier(load_config(PROJECT_ROOT / "configs" / "classify.yaml"))          # backend: vllm | llamacpp
clf.predict("The Last Signal", story_text)   # GenrePrediction(genre='Science Fiction', raw=..., valid=True, seconds=...)
```

`python scripts/eval_genre.py [--backend llamacpp]` runs the test split three ways (adapter + restriction,
adapter unrestricted, base model + restriction), plus a probe that asks for a non-genre answer, and writes
`results/genre_eval_<backend>.json`.

