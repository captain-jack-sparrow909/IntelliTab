#!/usr/bin/env python3
"""
MLX Code Completion Server — Phases A–E.

- Model stays loaded
- stdin reader thread so cancel can interrupt generation
- mode=fim (mid-line) vs mode=intent (comment) vs fim+ml (block continue)
- Phase C: prefix KV cache
- Phase D: speculative decoding (optional draft model on quality path)
- Phase E: dual-model routing (fast 3B mid-line FIM, quality 7B for hard paths)
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
from model import DualModelRouter, resolve_model_path


def _max_tokens_from(msg: dict, default: int) -> int:
    v = msg.get("max_tokens", msg.get("maxTokens", default))
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def handle_complete(router: DualModelRouter, msg: dict, write) -> None:
    msg_id = msg.get("id")
    context = msg.get("context") or {}
    language = context.get("language") or msg.get("language") or ""
    max_tokens = _max_tokens_from(msg, router.max_tokens)
    use_streaming = msg.get("streaming", True)
    stop_on_newline = bool(msg.get("stop_on_newline", msg.get("stopOnNewline", False)))

    # Dual policy: explicit mode from extension, or infer from intent field.
    mode = (context.get("mode") or ("intent" if context.get("intent") else "fim")).lower()
    is_intent = mode == "intent" or bool(context.get("intent"))

    if not context:
        write(encode_error("Empty context", msg_id))
        return

    t0 = time.perf_counter()
    try:
        stop_when_balanced = False
        stop_on_derail = False
        multi_line = False
        if is_intent:
            multi_line = False
            # Route before building prompt (tokenizer may differ per model).
            engine = router.pick(is_intent=True, multi_line=False)
            prompt = engine.build_intent_prompt(
                context.get("intent") or "Implement the code",
                language,
                context.get("before", ""),
            )
            # Multi-line bodies: enough tokens to finish. Early-stop is conservative
            # (single guard if/return must NOT end generation — deepEqual/clone etc.).
            max_tokens = max(max_tokens, 120)
            max_tokens = min(max_tokens, 180)
            stop_on_newline = False
            stop_when_balanced = True
            stop_on_derail = True
            mode = "intent"
        else:
            # Multi-line FIM: continuing an open block (after `{`, partial body).
            multi_line = (
                bool(context.get("multiLine") or context.get("multi_line"))
                or (
                    "stop_on_newline" in msg
                    and not bool(msg.get("stop_on_newline"))
                )
                or (
                    "stopOnNewline" in msg
                    and not bool(msg.get("stopOnNewline"))
                )
            )
            engine = router.pick(is_intent=False, multi_line=multi_line)
            prompt = engine.build_fim_prompt(
                context.get("before", ""),
                context.get("after", ""),
                language=language,
            )
            if multi_line:
                # Cap decode: long multi-line is where models invent prose / next fns.
                max_tokens = max(max_tokens, 48)
                max_tokens = min(max_tokens, 96)
                stop_on_newline = False
                # Do NOT use stop_when_balanced: generated text often closes
                # braces that were opened in the prefix (`if (...) { |`), so
                # balance checks misfire and cut after one return.
                stop_when_balanced = False
                stop_on_derail = True
                mode = "fim+ml"
            else:
                # One line is enough for mid-expression FIM (Phase E → fast model).
                max_tokens = max(16, min(max_tokens, 32))
                if "stop_on_newline" not in msg and "stopOnNewline" not in msg:
                    stop_on_newline = True
                mode = "fim"

        route = router.route_name(engine)
        prompt_chars = len(prompt)
        if use_streaming:
            tokens_sent = 0
            first_token_ms = None
            for token, is_final in engine.stream(
                prompt,
                max_tokens=max_tokens,
                msg_id=msg_id,
                stop_on_newline=stop_on_newline,
                stop_when_balanced=stop_when_balanced,
                stop_on_derail=stop_on_derail,
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
                f"[server] id={msg_id} mode={mode} route={route} "
                f"model={Path(engine.model_path).name} prompt={prompt_chars}c "
                f"tokens={tokens_sent} ttft={first_token_ms or -1:.0f}ms "
                f"total={total_ms:.0f}ms stop_nl={stop_on_newline} max={max_tokens}\n"
            )
            sys.stderr.flush()
        else:
            completion = engine.generate(prompt, max_tokens=max_tokens, msg_id=msg_id)
            write(encode_complete(completion, msg_id))
    except Exception as e:
        sys.stderr.write(f"[server] Error handling completion: {e}\n")
        sys.stderr.flush()
        write(encode_error(str(e), msg_id))


def stdin_reader(q: queue.Queue, router: DualModelRouter) -> None:
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
            router.request_cancel(msg.get("id"))
            if msg.get("id") is None:
                router.request_cancel(None)
            continue

        q.put(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX Code Completion Server")
    parser.add_argument("--model", type=str, default="", help="Path to quality MLX model dir")
    parser.add_argument("--quantization", type=str, default="4bit")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--streaming", action="store_true", default=True)
    # Phase D: speculative decoding (quality path only)
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to smaller draft model. Empty string disables. Default: auto-pick smaller sibling.",
    )
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable speculative decoding even if a draft model is available.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Draft tokens per verification step (1–8, default 3).",
    )
    # Phase E: dual-model routing
    parser.add_argument(
        "--fast-model",
        type=str,
        default=None,
        help="Path to fast mid-line FIM model. Empty disables dual routing. Default: auto 3B.",
    )
    parser.add_argument(
        "--no-dual-model",
        action="store_true",
        help="Disable Phase E dual-model routing (always use quality model).",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model or None)
    sys.stderr.write(f"[server] Quality model: {model_path}\n")
    sys.stderr.flush()

    # --draft-model "" means disable; omit flag means auto.
    draft_explicit = args.draft_model  # None = auto, "" = off, path = use
    speculative = not args.no_speculative
    if draft_explicit is not None and draft_explicit.strip() == "":
        speculative = False
        draft_explicit = ""

    dual_model = not args.no_dual_model
    fast_explicit = args.fast_model  # None = auto, "" = off, path = use
    if fast_explicit is not None and fast_explicit.strip() == "":
        dual_model = False
        fast_explicit = ""

    router = DualModelRouter.create(
        quality_path=model_path,
        quantization=args.quantization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        draft_model_path=draft_explicit,
        speculative=speculative,
        num_draft_tokens=args.num_draft_tokens,
        fast_model_path=fast_explicit,
        dual_model=dual_model,
    )

    write = message_writer()
    ready = {
        "type": "ready",
        "model": router.quality.model_path,
        "fast_model": router.fast.model_path if router.fast else "",
        "dual_model": router.dual_enabled,
        "speculative": router.draft_model is not None,
        "draft_model": router.draft_model_path or "",
        "num_draft_tokens": router.num_draft_tokens if router.draft_model else 0,
    }
    write(encode_message(ready))
    sys.stderr.flush()

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=stdin_reader, args=(q, router), daemon=True)
    t.start()

    while True:
        msg = q.get()
        if msg is None:
            sys.stderr.write("[server] Client disconnected, shutting down\n")
            sys.stderr.flush()
            break

        msg_type = msg.get("type")
        if msg_type == "complete":
            handle_complete(router, msg, write)
        elif msg_type == "ping":
            write(encode_message({"type": "pong", "id": msg.get("id")}))
        else:
            sys.stderr.write(f"[server] Unknown message type: {msg_type}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
