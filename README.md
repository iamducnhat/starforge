# Local RAG Assistant

A local coding assistant for tool-driven development workflows, with:

- CLI chat mode
- OpenAI-compatible API server mode
- Workspace-safe file tools
- Hybrid memory (keyword + semantic retrieval)
- Planner-based autonomous loop with reflection
- Optional fine-tuning utilities

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Install](#install)
6. [Quick Start](#quick-start)
7. [CLI Commands](#cli-commands)
8. [Model Providers](#model-providers)
9. [Configuration](#configuration)
10. [Tools](#tools)
11. [Memory System](#memory-system)
12. [Autonomous Mode](#autonomous-mode)
13. [Server Mode (OpenAI-Compatible)](#server-mode-openai-compatible)
14. [Testing](#testing)
15. [Fine-Tuning](#fine-tuning)
16. [Troubleshooting](#troubleshooting)
17. [Notes and Constraints](#notes-and-constraints)

## What This Project Does

This assistant orchestrates LLM responses with local tools. Instead of only generating text, it can:

- inspect and edit files
- search code and symbols
- execute controlled terminal sessions
- retrieve local memory blocks
- search/read web sources
- create and execute reusable functions/tool-macros
- run autonomous iterative improvement loops

It supports provider backends including `openrouter`, `ollama`, `google`, and `nvidia`.

## Architecture Overview

At a high level:

1. `main.py` builds the model + tool system + chat engine.
2. `assistant/chat_engine.py` manages conversation flow, tool-call loops, autonomous runs, session persistence, and reflection.
3. `assistant/tools.py` dispatches tools and wraps calls with reliability retries/fallbacks.
4. `assistant/workspace_tools.py` handles file/project/plan/symbol operations with workspace path guards.
5. `assistant/memory.py` manages memory blocks and hybrid retrieval.
6. `assistant/server.py` exposes an OpenAI-compatible `/v1/chat/completions` API.

## Project Structure

```text
assistant/
  chat_engine.py           # core conversation + tool loop + autonomous logic
  model.py                 # provider adapters and model routing
  tools.py                 # tool dispatcher + reliability wrapper
  workspace_tools.py       # file/project/plan/symbol tools
  memory.py                # memory blocks + semantic retrieval
  server.py                # OpenAI-compatible API
  prompt.py                # system prompt and tool protocol

functions/                 # persisted reusable functions/tool-macros
memory/
  blocks/                  # memory blocks
  plans/                   # plan JSON files
  sessions/                # session logs

tests/                     # test suite
finetune/                  # synthetic dataset + LoRA training scripts
main.py                    # entry point
```

## Requirements

Runtime dependencies (`requirements.txt`):

- `fastapi`
- `uvicorn`
- `anyio`
- `pydantic`
- `requests`
- `python-dotenv`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quick Start

### 1) Configure environment

Create/update `.env` in repo root.

Minimal OpenRouter example:

```bash
ASSISTANT_PROVIDER=openrouter
ASSISTANT_MODEL=arcee-ai/trinity-large-preview:free
OPENROUTER_API_KEY=your_key_here
```

Minimal Ollama example:

```bash
ASSISTANT_PROVIDER=ollama
ASSISTANT_MODEL=qwen3:8b
OLLAMA_URL=http://127.0.0.1:11434
```

### 2) Run CLI

```bash
python3 main.py
```

### 3) One-shot prompt

```bash
python3 main.py --once "How do I parse JSON in Python?"
```

### 4) Start API server

```bash
python3 main.py --server --host 0.0.0.0 --port 8000
```

## CLI Commands

In interactive CLI mode (`python3 main.py`), use:

- `/help`
- `/reset`
- `/status`
- `/maxout <n>` / `/maxout`
- `/ctx <n>` / `/ctx`
- `/autosize` / `/autosize status`
- `/stream <auto|native|chunk>` / `/stream`
- `/temperature <n>` / `/temperature`
- `/top_p <n>` / `/top_p`
- `/compact [on|off|status]`
- `/autostop [on|off]`
- `/session`
- `/session list`
- `/session new`
- `/session save <name>`
- `/session open <name>`
- `/auto`
- `/auto on [steps|infinite]`
- `/auto off`

## Model Providers

Set with `--provider` or `ASSISTANT_PROVIDER`:

- `openrouter`
- `ollama`
- `google`
- `nvidia`
- `auto`

### OpenRouter

- Uses `OPENROUTER_API_KEY`
- Supports model listing via:

```bash
python3 main.py --list-models
```

### Ollama

- Uses `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- Auto limit tuning can be enabled with `ASSISTANT_AUTO_LIMITS=1`

### Google

- Uses `GOOGLE_API_KEY` (or `GEMINI_API_KEY` fallback)

### Nvidia

- Uses `NVIDIA_API_KEY`

## Configuration

### Common runtime env vars

- `ASSISTANT_PROVIDER` (default `openrouter`)
- `ASSISTANT_MODEL` (default `arcee-ai/trinity-large-preview:free`)
- `ASSISTANT_STREAM_MODE` (`auto|native|chunk`, default `auto`)
- `ASSISTANT_STREAM_TIMEOUT` (default `35` for online providers)
- `ASSISTANT_TEMPERATURE` (default `0.2`)
- `ASSISTANT_TOP_P` (default `0.95` where applicable)
- `ASSISTANT_NUM_PREDICT` (max output token setting)
- `ASSISTANT_NUM_CTX` (context window for local Ollama)
- `ASSISTANT_AUTO_LIMITS` (`1` enabled by default)
- `ASSISTANT_AUTONOMOUS_STEPS` (`0` means infinite)
- `ASSISTANT_COMPACT_CONTEXT` (default on)
- `ASSISTANT_MAX_CONTEXT_CHARS` (default `180000`)
- `ASSISTANT_TOOL_REFLECTION` (default on)
- `ASSISTANT_PLAN_STEP_CAP` (default `8`)
- `ASSISTANT_TOOL_RETRIES` (default `2`)
- `ASSISTANT_TOOL_RETRY_BACKOFF` (default `0.35`)
- `ASSISTANT_LOG_FILE` (server mode sets `log.txt` if unset)

### Provider-specific env vars

OpenRouter:

- `OPENROUTER_URL` (default `https://openrouter.ai/api/v1`)
- `OPENROUTER_API_KEY`
- `OPENROUTER_FALLBACK_MODEL`
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_NAME`
- `OPENROUTER_PROVIDER`
- `OPENROUTER_PROVIDER_ONLY`

Ollama:

- `OLLAMA_URL`

Google:

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY` (fallback alias)

Nvidia:

- `NVIDIA_API_KEY`
- `NVIDIA_ENABLE_REASONING`

## Tools

Registered tools include:

Memory/tools:

- `find_in_memory`
- `search_memory`
- `create_block`
- `create_function`
- `run_function`

Web/tools:

- `search_web`
- `read_web`
- `scrape_web`
- `extract_code_snippets`
- `get_current_datetime`

Workspace/tools:

- `list_files`
- `read_file`
- `create_file`
- `create_folder`
- `delete_file`
- `write_file` (legacy alias)
- `edit_file`
- `search_project`
- `index_symbols`
- `lookup_symbol`
- `summarize_file`

Planning/todo tools:

- `create_plan`
- `list_plans`
- `get_plan`
- `add_todo`
- `update_todo`

Terminal tool:

- `run_terminal(action="start|send|read|close", ...)`

### Tool reliability layer

`ToolSystem.safe_tool_call()` adds:

- retry for transient failures (timeouts/rate limits/network)
- fallback argument strategy for fragile web tools
- attempt metadata in tool results

## Memory System

Memory blocks are filesystem-backed in `memory/blocks`.

Each block stores metadata (`info.json`) + content (`knowledge.md`).

Retrieval:

- `find_in_memory(keywords)` uses hybrid scoring:
  - keyword overlap
  - semantic similarity (local hashed embedding vectors)
- `search_memory(query)` performs semantic search directly

## Autonomous Mode

Run with:

```bash
python3 main.py --autonomous --autonomous-steps 8
# or infinite
python3 main.py --autonomous --autonomous-steps 0
```

Autonomous flow now includes:

1. planner step: model generates executable plan steps
2. `TaskState` tracks `goal`, `steps`, `current_step`, `history`
3. loop executes current step
4. reflection decides `advance`, `retry`, `replan`, `done`, or `bored`

Per-tool reflection can also trigger retries with revised calls.

## Server Mode (OpenAI-Compatible)

Start server:

```bash
python3 main.py --server --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

### Example: health check

```bash
curl -s http://127.0.0.1:8000/health
```

### Example: non-streaming completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Give me a Python hello world"}
    ]
  }'
```

### Example: streaming completion

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain quicksort briefly"}
    ]
  }'
```

## Testing

Run core tests:

```bash
PYTHONPATH=$PWD python3 -m pytest -q \
  tests/test_memory.py \
  tests/test_tool_calls.py \
  tests/test_utils.py \
  tests/test_code_intelligence.py \
  tests/test_tool_reliability.py
```

Run all tests:

```bash
PYTHONPATH=$PWD python3 -m pytest -q tests
```

## Fine-Tuning

### Build datasets

```bash
python3 finetune/build_function_dataset.py
python3 finetune/generate_synthetic_tool_use.py --per-topic 120
python3 finetune/build_full_tool_dataset.py \
  --train-output finetune/train_tool_use_full.jsonl \
  --val-output finetune/val_tool_use_full.jsonl
```

### Train LoRA SFT

```bash
python3 -m pip install -r finetune/requirements-train.txt

python3 finetune/train_lora_sft.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file finetune/train_tool_use_full.jsonl \
  --val-file finetune/val_tool_use_full.jsonl \
  --output-dir finetune/output/lora_tool_use \
  --epochs 2 --batch-size 1 --grad-accum 16 --max-length 2048 --bf16
```

Or run pipeline:

```bash
./finetune/run_full_finetune.sh
```

## Troubleshooting

### OpenRouter key missing

Symptom: startup/chat mentions missing key.

Fix:

- set `OPENROUTER_API_KEY`
- or use `--provider ollama`

### 402 payment required from provider

Symptom: model returns payment required.

Fix:

- add provider credits
- set `OPENROUTER_FALLBACK_MODEL` to a free model

### Context or output too large/small

Use CLI runtime controls:

- `/ctx <n>`
- `/maxout <n>`
- `/autosize`

### Repeated autonomous loops

Use:

- `/autostop on`
- smaller `--autonomous-steps`
- refine objective to be more concrete

## Notes and Constraints

- Storage is filesystem-only (no database).
- Workspace path guard prevents file operations outside project root.
- System prompt asks the agent to place generated task artifacts in `workspaces/<task_name>/`.
- For time-sensitive factual queries, the assistant is configured to call date/time + web tools first.

