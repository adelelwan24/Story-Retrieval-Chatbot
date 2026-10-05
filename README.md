# Story Retrieval Chatbot

Take-home for `FareedKhan/1k_stories_100_genre` (1,000 stories, 100 genres):
Task 1 story chatbot (RAG), Task 2 genre classifier on the same LLM, Task 3 CPU deployment.

Status: data preparation and the Qdrant index are in place. Chat, classifier and deployment come next
(see `../structure/project_structure.md` for the full plan).

## Layout

```
configs/retrieval.yaml      all data/embedding/chunking/Qdrant settings
src/storybot/
  config.py                 YAML loader; relative paths resolve against this project folder
  data/                     load.py (Hub or local copy), splits.py (7/1/2 stratified, seed 42), prepare.py
  retrieval/                chunking.py, embed.py (ModernBERT late chunking + BM25), qdrant_store.py, indexing.py
  chat/ classify/ eval/ deploy/   to be filled
scripts/                    prepare_data.py, build_index.py
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
