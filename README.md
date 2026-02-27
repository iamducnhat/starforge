# Local Coding Assistant

## Run

```bash
python3 main.py
```

## .env Defaults

The app auto-loads `.env` at startup. Current default backend/model:
- provider: `openrouter`
- model: value from `.env` (`ASSISTANT_MODEL`)

Set your API key in `.env`:
```bash
OPENROUTER_API_KEY=your_key_here
```

Startup now prints model connection info (provider, endpoint, streaming mode, connect attempts).
If requested OpenRouter model is unavailable, the app auto-falls back (default: `arcee-ai/trinity-large-preview:free`).
Default generation limits:
- online max output tokens (default): `32768`
- local (Ollama) context/maxout are auto-tuned by RAM + model size when `ASSISTANT_AUTO_LIMITS=1`
- online model context window is fetched (when available) and maxout is auto-capped from that window

## Optional

```bash
# choose local Ollama model
python3 main.py --model qwen2.5:3b-instruct

# explicitly use Ollama
python3 main.py --provider ollama --model qwen3:8b

# use OpenRouter online model
OPENROUTER_API_KEY=your_key \
python3 main.py --provider openrouter --model openai/gpt-4o-mini

# list OpenRouter models
OPENROUTER_API_KEY=your_key \
python3 main.py --list-models

# one-shot prompt
python3 main.py --once "How do I parse JSON in Python?"

# start with autonomous mode enabled
python3 main.py --autonomous --autonomous-steps 8

# autonomous infinite loop (stop with Ctrl+C)
python3 main.py --autonomous --autonomous-steps 0
```

Notes:
- Streaming is enabled for all providers.
- If native stream fails, the app falls back to chunked token streaming from a non-stream response.
- In CLI, use `/status` to view runtime settings.
- In CLI, use `/maxout <n>` to change max output tokens at runtime, or `/maxout` to view current value.
- In CLI, use `/ctx <n>` to change context window (if model supports), or `/ctx` to view current value.
- In CLI, use `/autosize` to auto-apply recommended context/maxout for current model.
- In CLI, use `/autosize status` to view current auto-limit recommendation.
- In CLI, use `/stream <auto|native|chunk>` to change stream mode, or `/stream` to view current mode.
- In CLI, use `/compact [on|off|status]` to control context compaction.
- In CLI, use `/auto`, `/auto on [steps|infinite]`, `/auto off` for autonomous self-operating mode.
- Autonomous can run infinite (`0` / `infinite`) and stops when:
  - user presses `Ctrl+C`, or
  - model outputs `AUTONOMOUS_DONE` / `AUTONOMOUS_BORED`.
- Autonomous mode is constrained to workspace root (`mymodel`) via workspace tools path guard.
- Context compaction is enabled by default to keep long sessions stable.
  - Disable: `ASSISTANT_COMPACT_CONTEXT=0`
  - Context char threshold: `ASSISTANT_MAX_CONTEXT_CHARS` (default `180000`)
- Web tools now include:
  - `read_web(url, ...)`: fetch one page, return text + links + code snippets.
  - `scrape_web(start_url, ...)`: crawl a site and return discovered links/pages.
  - `get_current_datetime()`: returns current runtime date/time for time-sensitive queries.
- `create_function` supports two modes:
  - code function (`code` field), or
  - tool-macro function (`tool_name`/`tool_args` or `tool_calls`) for reusable tool-call workflows.
- If `edit_file` hits "file not found", create the file first with `create_file`.

## Fine-tune Data (Function Tool Use)

```bash
# build focused training set for "create function" requests
python3 finetune/build_function_dataset.py

# generate large synthetic tool-use corpus
python3 finetune/generate_synthetic_tool_use.py --per-topic 120

# build merged train/val
python3 finetune/build_full_tool_dataset.py \
  --train-output finetune/train_tool_use_full.jsonl \
  --val-output finetune/val_tool_use_full.jsonl
```

## Fine-tune Training (LoRA SFT)

```bash
# install dependencies
python3 -m pip install -r finetune/requirements-train.txt

# train
python3 finetune/train_lora_sft.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file finetune/train_tool_use_full.jsonl \
  --val-file finetune/val_tool_use_full.jsonl \
  --output-dir finetune/output/lora_tool_use \
  --epochs 2 --batch-size 1 --grad-accum 16 --max-length 2048 --bf16

# one-shot full pipeline
./finetune/run_full_finetune.sh
```

## Notes

- Storage is filesystem-only: `memory/blocks` and `functions`.
- If Ollama is unavailable, a fallback response is used.
- `search_web` supports `level`: `quick`, `balanced`, `deep`, `auto` (default).
- Workspace coding tools are available:
  - `list_files`, `read_file`, `create_file`, `edit_file`, `search_project`
  - `create_plan`, `list_plans`, `get_plan`, `add_todo`, `update_todo`
