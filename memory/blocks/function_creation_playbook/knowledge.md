## Compressed Summary
- Before generating new function code, always do pre-research.
- Required flow: `find_in_memory` -> `search_web` -> synthesize code -> optionally `create_function`.
- If memory has reusable logic, reuse/adapt before creating new implementation.

## Reusable Patterns
- Pattern: function request handling
  - Extract intent keywords from user request.
  - Call `find_in_memory(keywords)`.
  - Call `search_web("how to <task> in python")`.
  - Build code from verified sources and memory results.
- Pattern: avoid premature code
  - Do not output final code before at least one web research call for non-trivial implementation tasks.
- Pattern: function quality checklist
  - Clear signature and docstring.
  - Input validation.
  - Error handling and deterministic return shape.
  - Minimal dependencies.

## Minimal Snippets
```json
{"tool_calls":[
  {"tool":"find_in_memory","args":{"keywords":["download","file","python","research"]}},
  {"tool":"search_web","args":{"query":"how to download file in python and read pdf content"}}
]}
```

```python
def download_file(url: str, dest_path: str, timeout: int = 30) -> str:
    """Download file from URL and return destination path."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    return dest_path
```
