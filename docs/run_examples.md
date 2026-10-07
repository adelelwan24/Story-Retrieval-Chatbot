# Generate the agent examples page (results/agent_examples.html)

The page shows, for each example question, the agent's Thought / Action / Observation steps and its real answer
(trimmed to 500 words), grouped in tabs by example type. It is built on your machine because it calls your LLM.

## 1. Update the project

1. Download `story-chatbot.zip` and extract it over your existing `story-chatbot` folder.
   The zip has no `data/`, `qdrant_data/` or `.venv/`, so your data, index and environment are kept.
   If you edited `configs/chat.yaml`, back it up first and copy your values back afterwards.
2. In the project folder, activate your environment and install the new requirements (adds `rapidfuzz`, `truststore`):

   ```powershell
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 2. Make sure the index exists (skip if you already built it)

```powershell
python scripts/prepare_data.py
python scripts/build_index.py
```

`build_index.py` prints "index already built" and stops when the index is there.

## 3. Point it at your LLM endpoint

The endpoint must be OpenAI-compatible and support tool calling (llama.cpp: start `llama-server` with `--jinja`).

PowerShell:

```powershell
$env:LLM_API_KEY = "your-key"                 # leave out for a local server without a key
$env:LLM_CA_BUNDLE = "C:\path\to\ca.pem"      # only if you get CERTIFICATE_VERIFY_FAILED
```

Command Prompt: `set LLM_API_KEY=your-key`

## 4. Generate the page

```powershell
# 2 examples of each of the 11 types (22 questions)
python scripts/make_examples_html.py --base-url https://your-endpoint/v1 --model your-model-name --per-type 2

# all 51 test questions
python scripts/make_examples_html.py --base-url https://your-endpoint/v1 --model your-model-name

# chosen questions only (ids from eval/agent_qa.csv)
python scripts/make_examples_html.py --base-url https://your-endpoint/v1 --model your-model-name --ids ask-15 title-05 theme-01
```

It prints one line per question while it runs, then writes:

* `results/agent_examples.html`: open it in a browser
* `results/agent_examples_runs.json`: the raw traces

## 5. Options

| Option | What it does |
|---|---|
| `--per-type N` | at most N examples of each type |
| `--ids ...` | only these question ids |
| `--max-words 300` | trim answers to 300 words instead of 500 |
| `--no-reference` | hide the "Expected" answer under each example |
| `--from-json results/agent_examples_runs.json` | rebuild the page from saved traces, without calling the LLM |
| `--out results/my_page.html` | write the page somewhere else |

`--base-url` and `--model` can also be set once in `configs/chat.yaml` (`llm.backend: openai`, `base_url`,
`server_model`) or as the env vars `LLM_BASE_URL` / `LLM_MODEL`; then run `python scripts/make_examples_html.py --per-type 2`.

## If something goes wrong

* `CERTIFICATE_VERIFY_FAILED`: set `LLM_CA_BUNDLE` to your company or endpoint CA file (step 3).
* An example shows a red "Error" row: that call to the endpoint failed (timeout, rate limit); the other examples still run.
  Re-run just that one with `--ids`.
* Answers with no tool calls at all: the endpoint is ignoring `tools`; check it supports tool calling.
