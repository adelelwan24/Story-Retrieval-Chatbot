# Story Retrieval Chatbot


Status: data preparation, the Qdrant index and the ReAct story agent are in place. Classifier and deployment come next
(see `../structure/project_structure.md` for the full plan).

## Layout

```
configs/retrieval.yaml      all data/embedding/chunking/Qdrant settings
configs/chat.yaml           LLM backend, agent loop and tool settings (fuzzy thresholds, result counts)
src/storybot/
  config.py                 YAML loader; relative paths resolve against this project folder
  data/                     load.py (Hub or local copy), splits.py (7/1/2 stratified, seed 42), prepare.py
  retrieval/                chunking.py, embed.py (ModernBERT late chunking + BM25), qdrant_store.py, indexing.py,
                            search.py (lookups + hybrid search), matching.py (fuzzy title/genre matching)
  chat/                     tools.py (agent tools), agent.py (ReAct loop), llm.py (HF / OpenAI-compatible), prompts.py
  classify/ eval/ deploy/   to be filled
scripts/                    prepare_data.py, build_index.py, chat.py
notebooks/02_build_index.ipynb   same steps as the scripts, with a search check at the end
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
