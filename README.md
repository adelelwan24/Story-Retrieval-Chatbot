# Story Retrieval Chatbot

Take-home for `FareedKhan/1k_stories_100_genre` (1,000 stories, 100 genres):
Task 1 story chatbot (RAG), Task 2 genre classifier on the same LLM, Task 3 CPU deployment.

Status: data preparation, the Qdrant index and the ReAct story agent are in place. Classifier and deployment come next
(see `../structure/project_structure.md` for the full plan).

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
                            serve_vllm.py, build_gguf.py, serve_llamacpp.py, eval_genre.py
models/                     genre_lora_qwen35/ (unzipped notebook output) and gguf/ (git-ignored)
eval/                       agent_qa.csv (the 51 test questions with reference answers), agent_test_cases.json
                            (same cases with the checks the scorer runs), stories_sample.csv (the 15 stories they use)
notebooks/02_build_index.ipynb   same steps as the scripts, with a search check at the end
notebooks/03_story_agent.ipynb   loads the collections, runs the agent on 12 example questions via an OpenAI-compatible endpoint
tests/                      offline tests (tiny random ModernBERT, fake stories, toy BM25)
data/                       stories.parquet (git-ignored) and splits.json, created by prepare_data
qdrant_data/                the Qdrant database, created by build_index (git-ignored)
```

## Run

```bash
pip install -r requirements.txt
python scripts/prepare_data.py              # data/stories.parquet + data/splits.json
python scripts/build_index.py               # qdrant_data/  (add --limit 50 for a quick run, --rebuild to redo)
python scripts/chat.py --trace              # chat with the agent (Qwen3-4B via transformers)
pytest                                      # offline, CPU, a few seconds
```

The database folder is set in `configs/retrieval.yaml` under `qdrant.path` (default `qdrant_data`, relative to
this folder; an absolute path also works). Set `qdrant.url` instead to use a Qdrant server.
Embedding uses the GPU when one is available and the CPU otherwise.

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
* `hf`: `Qwen/Qwen3-4B-Instruct-2507` through transformers, 4-bit NF4 on a CUDA GPU (Colab T4).
* `openai`: any OpenAI-compatible server, e.g. the CPU build `llama-server -m qwen3-4b-instruct-q4_k_m.gguf --jinja`.
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

