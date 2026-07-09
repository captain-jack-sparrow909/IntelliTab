#!/usr/bin/env python3
"""
MLX Code Completion Server — low-latency IPC loop.

- Model stays loaded
- stdin reader thread so cancel can interrupt generation
- Streaming tokens over length-prefixed JSON
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

from protocol import (
    decode_next_message,
    encode_message,
    encode_complete,
    encode_error,
    encode_token,
    message_writer,
)
from model import ModelEngine

DEFAULT_MODEL_PATH = Path.home() / ".mlx-models" / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"


def _max_tokens_from(msg: dict, default: int) -> int:
    # Accept both snake_case and camelCase from the extension.
    v = msg.get("max_tokens", msg.get("maxTokens", default))
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def handle_complete(engine: ModelEngine, msg: dict, write) -> None:
    msg_id = msg.get("id")
    context = msg.get("context") or {}
    language = context.get("language") or msg.get("language") or ""
    max_tokens = _max_tokens_from(msg, 32)
    use_streaming = msg.get("streaming", True)
    stop_on_newline = bool(msg.get("stop_on_newline", msg.get("stopOnNewline", False)))

    if not context:
        write(encode_error("Empty context", msg_id))
        return

    t0 = time.perf_counter()
    try:
        if context.get("intent"):
            prompt = engine.build_intent_prompt(
                context["intent"], language, context.get("before", "")
            )
            # Intent (comment→code) needs more tokens; still capped for latency.
            max_tokens = max(max_tokens, 128)
            stop_on_newline = False
        else:
            prompt = engine.build_fim_prompt(
                context.get("before", ""),
                context.get("after", ""),
                language=language,
            )

        if use_streaming:
            tokens_sent = 0
            first_token_ms = None
            for token, is_final in engine.stream(
                prompt,
                max_tokens=max_tokens,
                msg_id=msg_id,
                stop_on_newline=stop_on_newline,
            ):
                if token:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    write(encode_token(token, msg_id))
                    tokens_sent += 1
                if is_final:
                    break
            write(encode_token("", msg_id))
            total_ms = (time.perf_counter() - t0) * 1000
            sys.stderr.write(
                f"[server] id={msg_id} tokens={tokens_sent} "
                f"ttft={first_token_ms or -1:.0f}ms total={total_ms:.0f}ms "
                f"stop_nl={stop_on_newline} max={max_tokens}\n"
            )
            sys.stderr.flush()
        else:
            completion = engine.generate(prompt, max_tokens=max_tokens, msg_id=msg_id)
            write(encode_complete(completion, msg_id))
    except Exception as e:
        sys.stderr.write(f"[server] Error handling completion: {e}\n")
        sys.stderr.flush()
        write(encode_error(str(e), msg_id))


def stdin_reader(q: queue.Queue, engine: ModelEngine) -> None:
    """Read IPC messages continuously so cancel can arrive mid-generation."""
    while True:
        try:
            msg = decode_next_message()
        except EOFError:
            q.put(None)
            return
        except Exception as e:
            sys.stderr.write(f"[server] Error reading message: {e}\n")
            sys.stderr.flush()
            continue

        if msg.get("type") == "cancel":
            engine.request_cancel(msg.get("id"))
            # Also cancel whatever is active if id omitted.
            if msg.get("id") is None:
                engine.request_cancel(None)
            # Don't enqueue cancel as work — flag only.
            continue

        q.put(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Code Completion Server")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--quantization", type=str, default="4bit")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--streaming", action="store_true", default=True)
    args = parser.parse_args()

    engine = ModelEngine(
        model_path=args.model,
        quantization=args.quantization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    write = message_writer()
    write(encode_message({"type": "ready"}))
    sys.stderr.flush()

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=stdin_reader, args=(q, engine), daemon=True)
    t.start()

    while True:
        msg = q.get()
        if msg is None:
            sys.stderr.write("[server] Client disconnected, shutting down\n")
            sys.stderr.flush()
            break

        msg_type = msg.get("type")
        if msg_type == "complete":
            handle_complete(engine, msg, write)
        elif msg_type == "ping":
            write(encode_message({"type": "pong", "id": msg.get("id")}))
        else:
            sys.stderr.write(f"[server] Unknown message type: {msg_type}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
