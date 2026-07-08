#!/usr/bin/env python3
"""
MLX Code Completion Server

Persistent Python process that:
- Loads Qwen2.5-Coder model once on startup
- Listens for completion requests on stdin
- Streams completions back on stdout
- Never unloads the model or tokenizer

Protocol: length-prefixed JSON over stdin/stdout pipes.
"""

import sys
import json
import argparse
from pathlib import Path

from protocol import decode_next_message, encode_message, encode_complete, encode_error, encode_token, message_writer
from model import ModelEngine


# Default model path — can be overridden via config
DEFAULT_MODEL_PATH = Path.home() / ".mlx-models" / "Qwen2.5-Coder-7B-Instruct-MLX-4bit"


def handle_complete(engine: ModelEngine, msg: dict, write) -> None:
    """Handle a completion request."""
    msg_id = msg.get("id")
    context = msg.get("context", "")
    language = msg.get("language", "")
    max_tokens = msg.get("max_tokens", 64)
    use_streaming = msg.get("streaming", True)

    if not context:
        write(encode_error("Empty context", msg_id))
        return

    try:
        if use_streaming:
            # Streaming mode: send tokens one by one
            if context.get("intent"):
                prompt = engine.build_intent_prompt(
                    context["intent"], language, context.get("before", "")
                )
            else:
                prompt = engine.build_fim_prompt(context["before"], context["after"])
            tokens_sent = 0
            for token, is_final in engine.stream(prompt, max_tokens=max_tokens, msg_id=msg_id):
                if not token:
                    continue
                write(encode_token(token, msg_id))
                tokens_sent += 1
                if is_final:
                    break
            # Send final empty token to signal completion
            write(encode_token("", msg_id))
        else:
            # Non-streaming: send full completion at once
            if context.get("intent"):
                prompt = engine.build_intent_prompt(
                    context["intent"], language, context.get("before", "")
                )
            else:
                prompt = engine.build_fim_prompt(context["before"], context["after"])
            completion = engine.generate(prompt, max_tokens=max_tokens, msg_id=msg_id)
            write(encode_complete(completion, msg_id))

    except Exception as e:
        sys.stderr.write(f"[server] Error handling completion: {e}\n")
        sys.stderr.flush()
        write(encode_error(str(e), msg_id))


def handle_cancel(msg: dict, write) -> None:
    """Handle a cancellation request."""
    msg_id = msg.get("id")
    sys.stderr.write(f"[server] Cancelled request {msg_id}\n")
    sys.stderr.flush()
    # In streaming mode, the caller simply stops reading tokens.
    # We acknowledge with a cancel message.
    write(encode_message({"type": "cancelled", "id": msg_id}))


def handle_ping(msg: dict, write) -> None:
    """Handle a health-check ping."""
    write(encode_message({"type": "pong", "id": msg.get("id")}))


def main():
    parser = argparse.ArgumentParser(description="MLX Code Completion Server")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to MLX model directory")
    parser.add_argument("--quantization", type=str, default="4bit",
                        help="Quantization level (4bit, 8bit, BF16)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Stream completions token-by-token")
    args = parser.parse_args()

    # Initialize model engine (loads model once)
    engine = ModelEngine(
        model_path=args.model,
        quantization=args.quantization,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Initialize message writer
    write = message_writer()

    # Send ready signal
    write(encode_message({"type": "ready"}))
    sys.stderr.flush()

    # Main message loop
    while True:
        try:
            msg = decode_next_message()
        except EOFError:
            sys.stderr.write("[server] Client disconnected, shutting down\n")
            sys.stderr.flush()
            break
        except Exception as e:
            sys.stderr.write(f"[server] Error reading message: {e}\n")
            sys.stderr.flush()
            continue

        msg_type = msg.get("type")

        if msg_type == "complete":
            handle_complete(engine, msg, write)
        elif msg_type == "cancel":
            handle_cancel(msg, write)
        elif msg_type == "ping":
            handle_ping(msg, write)
        else:
            sys.stderr.write(f"[server] Unknown message type: {msg_type}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
