# Starforge

An autonomous coding agent that learns from its own mistakes.

> Not just a chatbot. A system that improves every time it runs.

## Table of Contents

- [Starforge](#starforge)
  - [Table of Contents](#table-of-contents)
  - [Why Starforge?](#why-starforge)
  - [Demo](#demo)
  - [Core Capabilities](#core-capabilities)
    - [Self-Debugging Loop](#self-debugging-loop)
    - [Root-Cause Memory](#root-cause-memory)
    - [Strategy + Skill Learning](#strategy--skill-learning)
    - [Autonomous Execution](#autonomous-execution)
    - [State-Aware Learning](#state-aware-learning)
  - [Recent Feature Updates](#recent-feature-updates)
  - [What This Means](#what-this-means)
  - [Quick Start](#quick-start)
    - [1. Install](#1-install)
    - [2. Configure](#2-configure)
    - [3. Run CLI](#3-run-cli)
    - [4. Run Autonomous Mode](#4-run-autonomous-mode)
    - [5. Start API Server](#5-start-api-server)
    - [6. Run One Autonomous Objective And Exit](#6-run-one-autonomous-objective-and-exit)
  - [Architecture (High-Level)](#architecture-high-level)
  - [Memory System](#memory-system)
  - [Autonomous Loop](#autonomous-loop)
  - [Fine-Tuning Pipeline](#fine-tuning-pipeline)
  - [Notes](#notes)
  - [Vision](#vision)
  - [If you find this interesting](#if-you-find-this-interesting)

## Why Starforge?

Most AI agents:

* fail → retry → fail again
* forget everything after each run

**Starforge is different.**

It:

* debugs itself (hypothesis → fix → test)
* remembers root causes (not just errors)
* improves over time (strategies + skills + trajectories)
* works directly inside your codebase

## Demo

![Starforge Demo](./assets/demo.gif)

## Core Capabilities

### Self-Debugging Loop

* detects failures
* generates hypotheses
* applies fixes
* validates with tests
* learns from outcomes

### Root-Cause Memory

* stores reusable fixes (not just logs)
* applies deterministic repair templates
* confidence-based matching

### Strategy + Skill Learning

* saves successful multi-step plans
* reuses and adapts them
* builds parameterized skill templates

### Autonomous Execution

* DAG-based planning (dependency-aware)
* execution → validation → scoring → decision
* retry / replan / repair loops

### State-Aware Learning

Each step tracks:

* files changed
* tests fixed
* new errors / regressions
* progress signals

## Recent Feature Updates

* **Workspace-aware repo editing**: the assistant now pre-inspects explicit file and folder paths, detects project context before major edits, indexes symbols for large codebases, and uses safer `edit_file` matching when whitespace, trailing newlines, or Unicode punctuation drift from the original text.
* **More reliable workspace tools**: terminal sessions now use bounded output buffers with truncation reporting and idle cleanup, project search streams `rg` results with safer limits, and shell execution uses a non-login shell for more deterministic behavior inside the repo.
* **Stronger autonomous convergence controls**: autonomous runs skip plan/todo mutation tools during execution, cache duplicate read/test calls until the workspace changes, suppress blind retries of identical failing commands, track validation signals and test-failure snapshots, preserve completed steps across replans, and enforce retry, replan, no-progress, and tool-call budgets to avoid unbounded loops.
* **Faster deterministic repair triggers**: when Starforge already has the failing assertion, the implicated test, and the buggy source line in hand, it can apply a same-loop immediate fix instead of burning extra inspection turns before patching.
* **Higher-signal validation**: workspace validation now separates “validation completed” from “tests passed”, classifies collection errors and unparsed non-zero exits more clearly, includes untracked/staged changes, and can focus diff reporting on the files touched by the current step instead of the whole dirty repo.
* **Runtime guardrails for long sessions**: the engine now compacts oversized history entries, clips large tool payloads before storing them in context, and applies a memory guard that can evict cold memory/skill state, close idle terminals, clear DNS cache, and stop runs that exceed a hard memory limit.
* **Richer learned skills and memory reuse**: auto-learned skills can now store structured metadata such as `skill`, `inputs`, `steps_template`, and `match_conditions`, while hot caches for strategies, knowledge blocks, root causes, and skills are ranked and trimmed for reuse.
* **API and research upgrades**: streamed API responses now handle disconnects and queue backpressure more cleanly, streamed usage accounting is available when requested, web search can persist results on demand, and `--autonomous-objective` supports one-shot autonomous runs from the CLI.

## What This Means

Starforge doesn’t just solve tasks.

It builds **experience**.

Over time, it becomes:

* faster
* more accurate
* less dependent on trial-and-error

## Quick Start

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create `.env`:

```bash
ASSISTANT_PROVIDER=ollama
ASSISTANT_MODEL=qwen3:8b
OLLAMA_URL=http://127.0.0.1:11434
```

### 3. Run CLI

```bash
python3 main.py
```

### 4. Run Autonomous Mode

```bash
python3 main.py --autonomous --autonomous-steps 8
```

### 5. Start API Server

```bash
python3 main.py --server --host 0.0.0.0 --port 8000
```

### 6. Run One Autonomous Objective And Exit

```bash
python3 main.py --autonomous-objective "fix the failing tests" --autonomous-steps 8
```

## Architecture (High-Level)

* `chat_engine.py` → core loop + autonomous logic
* `tools.py` → tool dispatcher + reliability layer
* `workspace_tools.py` → safe file/system operations
* `memory.py` → memory, strategies, root-cause learning
* `server.py` → OpenAI-compatible API
* `model.py` → provider routing + runtime DNS fallbacks

## Memory System

Starforge uses multiple memory layers:

* **Blocks** → general knowledge
* **Strategies** → successful multi-step plans with scored reuse
* **Root Causes** → deterministic repair patterns with feedback loops
* **Trajectories** → full execution logs for training

## Autonomous Loop

```
plan → execute → validate → score → decide
```

Includes:

* dependency-aware execution
* project-context bootstrap + workspace preinspection
* duplicate tool-call suppression for repeated reads and test commands
* same-loop immediate-fix heuristics for simple high-confidence bugs
* validator-driven retry / replan gating
* focused workspace validation with explicit failure modes
* test-driven repair loop + root-cause and deterministic fast-path fixes
* convergence controller with retry, replan, and no-progress limits

## Fine-Tuning Pipeline

Starforge logs interaction trajectories:

```bash
memory/interaction_trajectories.jsonl
```

Build dataset:

```bash
python3 finetune/build_interaction_dataset.py \
  --input memory/interaction_trajectories.jsonl \
  --output finetune/interaction_train.jsonl \
  --min-score 0.7
```

Train LoRA:

```bash
python3 finetune/train_lora_sft.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file finetune/interaction_train.jsonl
```

## Notes

* filesystem-based (no database)
* workspace-root-safe file operations, with `workspaces/<task_name>/` as the default scratch area
* bounded caches and queues for long-running autonomous sessions
* designed for iterative improvement, not one-shot answers

## Vision

Starforge is a step toward:

> self-improving AI systems that learn from real-world execution

## If you find this interesting

Give it a star — it helps a lot
