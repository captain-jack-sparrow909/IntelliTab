"""
MLX model engine for low-latency code completion.

Design goals:
- Keep the model + tokenizer resident
- Prefer short prompts (prefill dominates TTFT)
- Support cooperative cancel between tokens
- Early-stop for single-line ghost text
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler


QUANT_CONFIG = {
    "4bit": {"quant_group_size": 32, "bits": 4},
    "6bit": {"quant_group_size": 32, "bits": 6},
    "8bit": {"quant_group_size": 32, "bits": 8},
    "BF16": {},
    "IQ4_XS": {"quant_group_size": 32, "bits": 4},
}

# Qwen2.5-Coder FIM special tokens (present in many Qwen coder tokenizers).
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def _is_already_quantized(model_path: str) -> bool:
    p = Path(model_path)
    if p.joinpath("quantized.safetensors.index.json").exists():
        return True
    config_path = p.joinpath("config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if "quantization" in config:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _truncate(text: str, max_chars: int, from_end: bool = False) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if from_end:
        return text[-max_chars:]
    return text[:max_chars]


class ModelEngine:
    """Manages the MLX model and generates completions."""

    def __init__(
        self,
        model_path: str,
        quantization: str = "4bit",
        max_tokens: int = 32,
        temperature: float = 0.0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cancel_lock = threading.Lock()
        self._cancel_ids: set[int] = set()
        self._active_id: Optional[int] = None

        sys.stderr.write(f"[mlx] Loading model from: {model_path}\n")
        sys.stderr.flush()

        model_config = {}
        if not _is_already_quantized(model_path):
            quant_cfg = QUANT_CONFIG.get(quantization, QUANT_CONFIG["4bit"])
            if quant_cfg:
                model_config["quantization"] = quant_cfg
                sys.stderr.write(f"[mlx] Applying quantization: {quantization}\n")
                sys.stderr.flush()

        self.model, self.tokenizer = load(
            model_path,
            model_config=model_config if model_config else None,
        )

        # Prefer true FIM when the tokenizer knows the special tokens.
        vocab = getattr(self.tokenizer, "get_vocab", lambda: {})() or {}
        self._has_fim = FIM_PREFIX in vocab or FIM_PREFIX in getattr(
            self.tokenizer, "special_tokens_map", {}
        )
        # Also try encoding — some tokenizers keep specials outside get_vocab.
        if not self._has_fim:
            try:
                ids = self.tokenizer.encode(FIM_PREFIX, add_special_tokens=False)
                self._has_fim = len(ids) == 1
            except Exception:
                self._has_fim = False

        sys.stderr.write(
            f"[mlx] Model loaded successfully (fim={'yes' if self._has_fim else 'no'})\n"
        )
        sys.stderr.flush()

        self.sampler = make_sampler(temp=temperature, top_p=1.0)

    # --- cancellation --------------------------------------------------------

    def request_cancel(self, msg_id: Optional[int] = None) -> None:
        with self._cancel_lock:
            if msg_id is None:
                if self._active_id is not None:
                    self._cancel_ids.add(self._active_id)
            else:
                self._cancel_ids.add(msg_id)

    def _is_cancelled(self, msg_id: Optional[int]) -> bool:
        if msg_id is None:
            return False
        with self._cancel_lock:
            return msg_id in self._cancel_ids

    def _clear_cancel(self, msg_id: Optional[int]) -> None:
        if msg_id is None:
            return
        with self._cancel_lock:
            self._cancel_ids.discard(msg_id)

    # --- prompts -------------------------------------------------------------

    def build_fim_prompt(
        self,
        before: str,
        after: str,
        language: str = "",
    ) -> str:
        """Build a short fill-in-the-middle / continuation prompt.

        Prefill cost scales with prompt length — keep context tight.
        """
        # Hard cap characters (roughly ~1–1.5k tokens max context for speed).
        before = _truncate(before, 3500, from_end=True)
        after = _truncate(after, 800, from_end=False)

        if self._has_fim:
            # Native FIM — shortest path, best for IDE completion latency.
            return f"{FIM_PREFIX}{before}{FIM_SUFFIX}{after}{FIM_MIDDLE}"

        # Instruct fallback: still short — no giant system essay.
        lang = (language or "code").strip() or "code"
        system = (
            "Code completion only. Output solely the text to insert at <CURSOR>. "
            "No markdown, no explanation, do not repeat existing code."
        )
        user = f"<{lang}>\n{before}<CURSOR>{after}\n</{lang}>\nInsert:"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def build_intent_prompt(
        self,
        intent: str,
        language: str = "javascript",
        context: str = "",
    ) -> str:
        intent = intent.strip()
        language = (language or "javascript").strip()
        lang_name = {
            "js": "JavaScript",
            "javascript": "JavaScript",
            "ts": "TypeScript",
            "typescript": "TypeScript",
            "py": "Python",
            "python": "Python",
            "java": "Java",
            "go": "Go",
            "rust": "Rust",
            "cpp": "C++",
            "c": "C",
            "csharp": "C#",
            "rb": "Ruby",
            "php": "PHP",
            "sh": "Shell",
            "shellscript": "Shell",
            "html": "HTML",
            "css": "CSS",
        }.get(language.lower(), language)

        # Intent can afford a bit more context, but still truncate.
        surrounding = _truncate((context or "").strip(), 2000, from_end=True)
        system = (
            f"Write {lang_name} only. Match surrounding style. "
            "Code only — no fences, no explanation."
        )
        user = f"{intent}"
        if surrounding:
            user += f"\n\nContext:\n{surrounding}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    # --- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        msg_id: Optional[int] = None,
    ) -> str:
        max_tok = max_tokens or self.max_tokens
        try:
            return generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tok,
                sampler=self.sampler,
            )
        except Exception as e:
            sys.stderr.write(f"[mlx] Generation error: {e}\n")
            sys.stderr.flush()
            return ""

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        msg_id: Optional[int] = None,
        stop_on_newline: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        """Yield (token, is_final). Honors cancel + optional first-newline stop."""
        max_tok = max_tokens or self.max_tokens
        self._active_id = msg_id
        self._clear_cancel(msg_id)

        try:
            count = 0
            emitted = ""
            for item in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tok,
                sampler=self.sampler,
            ):
                if self._is_cancelled(msg_id) or (should_stop and should_stop()):
                    sys.stderr.write(f"[mlx] cancelled id={msg_id} after {count} tok\n")
                    sys.stderr.flush()
                    yield ("", True)
                    return

                token = item.text
                if not token:
                    count += 1
                    continue

                # Single-line mode: cut at first newline (keep nothing after).
                if stop_on_newline:
                    if "\n" in token:
                        before_nl, _sep, _rest = token.partition("\n")
                        if before_nl:
                            yield (before_nl, True)
                        else:
                            yield ("", True)
                        return
                    if "\n" in emitted:
                        yield ("", True)
                        return

                emitted += token
                is_final = count >= max_tok - 1
                yield (token, is_final)
                count += 1
                if is_final:
                    return
        except Exception as e:
            sys.stderr.write(f"[mlx] Stream generation error: {e}\n")
            sys.stderr.flush()
            yield ("", True)
        finally:
            self._clear_cancel(msg_id)
            if self._active_id == msg_id:
                self._active_id = None
