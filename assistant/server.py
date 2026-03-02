"""
OpenAI-compatible REST API server.

Endpoints
---------
GET  /                          – status / health (informal)
GET  /health                    – health check (OpenAI-client compatible)
GET  /v1/models                 – list available model aliases
POST /v1/chat/completions       – chat completions (streaming + non-streaming)

Design goals
~~~~~~~~~~~~
* Full compatibility with the OpenAI Chat Completions API spec so that
  **Xcode 26 AI Assistant**, Cursor, Continue.dev, and any standard OpenAI
  client can connect with no patching.
* Streaming responses use proper `text/event-stream` SSE format.
* Tool calls run **server-side only** – they never appear as text artifacts
  in the client-facing stream.  The client sees clean content deltas.
* Each HTTP request gets its own isolated ChatEngine history, seeded from
  the `messages` list supplied by the client (stateless from the client
  perspective, stateful only within a single request's tool-call loop).
* If the client sends a `role=system` message it overrides the default
  system prompt for that request, allowing Xcode / the user to inject
  their own context.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat_engine import ChatEngine
from .cli_format import print_tool_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Local Coding Assistant – OpenAI-Compatible API",
    description=(
        "Drop-in replacement for the OpenAI Chat Completions API. "
        "Compatible with Xcode 26 AI Assistant, Cursor, Continue.dev, etc."
    ),
    version="1.0.0",
)

# CORS – allow all origins so that Xcode, web UIs and local tools work
# without cross-origin issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance set by main.py via start_server()
engine: Optional[ChatEngine] = None


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatToolCallFunction(BaseModel):
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ChatToolCallFunction


def _extract_text_content(raw: Any) -> str:
    """
    Normalise a content field to a plain string.

    OpenAI spec allows content to be:
      - a plain string             "Hello"
      - a list of content parts    [{"type": "text", "text": "Hello"}, ...]

    Xcode 26 always sends the list form, which Pydantic rejects if the
    field is typed as ``Optional[str]``.  This helper flattens both forms.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                # OpenAI multimodal format: {"type": "text", "text": "..."}
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                # Fallback: any "content" key
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(raw)


class ChatMessage(BaseModel):
    role: str
    # Accept str OR list-of-parts (Xcode 26 always sends the list form)
    content: Optional[Union[str, List[Any]]] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ChatToolCall]] = None

    def text(self) -> str:
        """Return the message content as a plain string."""
        return _extract_text_content(self.content)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Helper: build a per-request ChatEngine
# ---------------------------------------------------------------------------

def _build_request_engine(request: ChatCompletionRequest) -> tuple[ChatEngine, str]:
    """
    Create an isolated ChatEngine for a single request.

    Returns
    -------
    (req_engine, user_message)
        req_engine  – fresh engine seeded with the message history
        user_message – the final user turn extracted from messages
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Determine system prompt.  If the client supplies a system message use it
    # (Xcode 26 sends its own coding context here), otherwise fall back to our
    # internal SYSTEM_PROMPT.
    system_prompt = engine.system_prompt
    non_system_messages = []
    for msg in request.messages:
        if msg.role == "system":
            # Last system message wins (Xcode sends exactly one at position 0).
            txt = msg.text()
            if txt:
                system_prompt = txt
        else:
            non_system_messages.append(msg)

    # Build isolated engine
    req_engine = ChatEngine(
        model=engine.model,
        tools=engine.tools,
        system_prompt=system_prompt,
        max_history=engine.max_history,
        max_tool_rounds=engine.max_tool_rounds,
        autonomous_enabled=False,
    )

    # Reconstruct conversation history from all messages except the final user turn.
    # The final user turn is passed directly to handle_turn() / handle_turn_stream().
    req_engine.history = []
    history_messages = non_system_messages[:-1] if len(non_system_messages) > 1 else []

    for msg in history_messages:
        role = msg.role
        content = msg.text()  # flatten str-or-list → plain str

        if role == "assistant" and msg.tool_calls:
            # Encode tool calls in the internal JSON format
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                calls.append({"tool": tc.function.name, "args": args})
            content = json.dumps({"tool_calls": calls}, ensure_ascii=False)

        if role == "tool":
            tool_name = msg.name or "unknown"
            try:
                json.loads(content)  # already valid JSON – keep as-is
                content = json.dumps(
                    {"tool": tool_name, "result": json.loads(content)},
                    ensure_ascii=False,
                )
            except Exception:
                content = json.dumps(
                    {"tool": tool_name, "result": content},
                    ensure_ascii=False,
                )

        req_engine.history.append({"role": role, "content": content})

    # Extract the user's latest message (flatten list content → str)
    last_msg = non_system_messages[-1] if non_system_messages else None
    user_message = last_msg.text() if last_msg else ""

    return req_engine, user_message


def _token_estimate(text: str) -> int:
    """Rough token count heuristic (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Informal status endpoint."""
    return {
        "status": "running",
        "model": engine.model.model_name if engine else "none",
    }


@app.get("/health")
async def health():
    """
    Health-check endpoint expected by many OpenAI-compatible clients.
    Returns HTTP 200 when the server is ready.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """
    Return the list of available model identifiers.

    Xcode 26 calls this endpoint to discover models when you add a custom
    provider.  We expose the actual underlying model plus common OpenAI-style
    aliases so that tools that hard-code model IDs still work.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    model_info = engine.model.info()
    actual_model_id = model_info.get("model", "local-model")
    created = int(time.time())

    # Always include the real model first, then common aliases
    model_ids = [actual_model_id]
    aliases = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
    ]
    for alias in aliases:
        if alias != actual_model_id:
            model_ids.append(alias)

    data = []
    for mid in model_ids:
        data.append({
            "id": mid,
            "object": "model",
            "created": created,
            "owned_by": "assistant" if mid == actual_model_id else "openai",
            "permission": [],
        })

    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI Chat Completions endpoint.

    Supports both streaming (stream=true) and non-streaming responses.
    Tool calls run entirely server-side; the client only sees clean text
    content deltas – matching ChatGPT / Claude behaviour.
    """
    req_engine, user_message = _build_request_engine(request)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # ── Request log ──────────────────────────────────────────────────────
    _preview = user_message[:100].replace("\n", " ")
    _mode = "stream" if request.stream else "sync"
    print(
        f"\n[{time.strftime('%H:%M:%S')}] ▶ chat/{_mode}  "
        f"model={request.model}  msgs={len(request.messages)}  "
        f'user="{_preview}{"…" if len(user_message) > 100 else ""}"',
        flush=True,
    )
    _t0 = time.time()


    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if not request.stream:
        try:
            response_text = req_engine.handle_turn(
                user_message,
                on_tool=lambda name, args, result: print_tool_event(name, args, result),
            )
        except Exception as exc:
            logger.exception("handle_turn failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt_tokens = sum(_token_estimate(m.content or "") for m in request.messages)
        completion_tokens = _token_estimate(response_text)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    chunk_queue: queue.Queue[dict | None] = queue.Queue()

    # Track whether we are inside <think>…</think> blocks.
    # Reasoning chunks are forwarded to the client as `reasoning_content` deltas
    # (standard in DeepSeek / Gemini Thinking models; ignored by clients that
    # don’t understand them).  Only content between the tags is suppressed from
    # the regular `content` delta.
    state = {"in_reasoning": False}

    def on_chunk(chunk: str) -> None:
        """Called by handle_turn_stream for every text chunk emitted by the model."""
        if chunk == "<think>":
            state["in_reasoning"] = True
            return
        if chunk == "</think>":
            state["in_reasoning"] = False
            return
        if state["in_reasoning"]:
            # Forward thinking as reasoning_content so clients that support it
            # (e.g. Gemini Thinking, DeepSeek-R1) can display it.
            chunk_queue.put({"type": "reasoning", "data": chunk})
        else:
            chunk_queue.put({"type": "content", "data": chunk})

    def run_engine() -> None:
        """Runs in a background thread – streams chunks into chunk_queue."""
        try:
            req_engine.handle_turn_stream(
                user_message,
                on_chunk=on_chunk,
                # Tool events are logged server-side only; never sent to client
                on_tool=lambda name, args, result: print_tool_event(name, args, result),
            )
        except Exception as exc:
            logger.exception("handle_turn_stream failed")
            chunk_queue.put({"type": "content", "data": f"\n[Server error: {exc}]"})
        finally:
            chunk_queue.put(None)  # sentinel – stream is done

    threading.Thread(target=run_engine, daemon=True).start()

    async def stream_generator() -> AsyncGenerator[str, None]:
        # --- initial role chunk (required by OpenAI spec) ---
        initial = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(initial, ensure_ascii=False)}\n\n"

        total_content = ""

        while True:
            item = await anyio.to_thread.run_sync(chunk_queue.get)
            if item is None:
                break  # stream finished

            data_str = item.get("data", "")
            item_type = item.get("type", "content")
            if not data_str:
                continue

            if item_type == "content":
                total_content += data_str

            # reasoning_content is a standard extension (DeepSeek, Gemini Thinking).
            # Clients that don’t understand it silently ignore the key.
            if item_type == "reasoning":
                delta = {"reasoning_content": data_str}
            else:
                delta = {"content": data_str}

            chunk_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # --- final [STOP] chunk ---
        prompt_tokens = sum(_token_estimate(m.content or "") for m in request.messages)
        completion_tokens = _token_estimate(total_content)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        include_usage = bool(
            request.stream_options and request.stream_options.get("include_usage")
        )

        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        if include_usage:
            final_chunk["usage"] = usage
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

        # Some clients (e.g. Continue.dev) expect a separate usage-only chunk
        if include_usage:
            usage_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------

def start_server(chat_engine: ChatEngine, host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the uvicorn server.

    Called from main.py when ``--server`` flag is provided.
    """
    global engine
    engine = chat_engine
    import uvicorn
    logger.info("Starting OpenAI-compatible API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
