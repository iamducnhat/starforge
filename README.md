# Local Coding Assistant

## Run

```bash
python3 main.py
```

## Optional

```bash
# choose local Ollama model
python3 main.py --model qwen2.5:3b-instruct

# one-shot prompt
python3 main.py --once "How do I parse JSON in Python?"
```

## Fine-tune Data (Function Tool Use)

```bash
# build focused training set for "create function" requests
python3 finetune/build_function_dataset.py
```

## Notes

- Storage is filesystem-only: `memory/blocks` and `functions`.
- If Ollama is unavailable, a fallback response is used.
- `search_web` supports `level`: `quick`, `balanced`, `deep`, `auto` (default).
