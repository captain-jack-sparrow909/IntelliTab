"""
MLX model engine — FIM-first, latency-tuned.

Critical: Qwen FIM often emits a leading "\\n" before the real code.
Early-stop must ignore leading newlines or completions become empty
and the UI thrash-retries for seconds.
"""

from __future__ import annotations

import json
import re
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

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"

DEFAULT_MODEL_DIR = Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-4bit"
# Prefer quality (7B Instruct) when both are installed; 3B is faster fallback.
FALLBACK_MODEL_DIRS = [
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-7B-Instruct-MLX-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-Instruct-MLX-4bit",
]


def resolve_model_path(explicit: Optional[str] = None) -> str:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        sys.stderr.write(f"[mlx] configured model path missing: {p}\n")

    for candidate in FALLBACK_MODEL_DIRS:
        if candidate.is_dir() and (candidate / "config.json").exists():
            return str(candidate)

    return str(DEFAULT_MODEL_DIR)


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


def _lang_name(language: str) -> str:
    language = (language or "code").strip()
    return {
        "js": "JavaScript",
        "javascript": "JavaScript",
        "jsx": "JavaScript",
        "ts": "TypeScript",
        "typescript": "TypeScript",
        "tsx": "TypeScript",
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
        "swift": "Swift",
        "kotlin": "Kotlin",
    }.get(language.lower(), language or "code")


class ModelEngine:
    def __init__(
        self,
        model_path: str,
        quantization: str = "4bit",
        max_tokens: int = 40,
        temperature: float = 0.0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cancel_lock = threading.Lock()
        self._cancel_ids: set[int] = set()
        self._active_id: Optional[int] = None

        model_path = resolve_model_path(model_path)
        self.model_path = model_path

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

        self._has_fim = self._detect_fim_support()
        self._has_chat = bool(getattr(self.tokenizer, "chat_template", None))
        name = Path(model_path).name.lower()
        self._is_instruct = "instruct" in name

        sys.stderr.write(
            f"[mlx] Ready fim={'yes' if self._has_fim else 'no'} "
            f"chat={'yes' if self._has_chat else 'no'} "
            f"kind={'instruct' if self._is_instruct else 'base'}\n"
        )
        sys.stderr.flush()

        self.sampler = make_sampler(temp=temperature, top_p=1.0)

    def _detect_fim_support(self) -> bool:
        tok = self.tokenizer
        try:
            ids = tok.encode(FIM_PREFIX, add_special_tokens=False)
            if hasattr(ids, "__len__") and len(ids) == 1:
                return True
        except Exception:
            pass
        try:
            vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
            if vocab and FIM_PREFIX in vocab:
                return True
        except Exception:
            pass
        try:
            enc = getattr(tok, "added_tokens_encoder", {}) or {}
            if FIM_PREFIX in enc:
                return True
        except Exception:
            pass
        name = Path(self.model_path).name.lower()
        if "qwen" in name and "coder" in name:
            return True
        return False

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

    def build_fim_prompt(
        self,
        before: str,
        after: str,
        language: str = "",
    ) -> str:
        before = _truncate(before, 2800, from_end=True)
        after = _truncate(after, 500, from_end=False)

        # Instruct: few-shot style "suffix only" — models love to re-emit the line
        # or wrap in ``` otherwise (seen in logs: raw="```javascript").
        if self._is_instruct and self._has_chat:
            lang = _lang_name(language)
            system = (
                "You are a code completion engine for an IDE.\n"
                "Rules:\n"
                "1. Output ONLY the characters that should be inserted at <CURSOR>.\n"
                "2. Do NOT repeat any characters that already appear before <CURSOR>.\n"
                "3. Do NOT output markdown fences or language tags (no ```).\n"
                "4. Do NOT output explanations.\n"
                "5. Prefer a short completion (usually the rest of the current line).\n"
                f"6. Language: {lang}."
            )
            # One concrete example so the model copies the "suffix only" format.
            example = (
                "Example:\n"
                "Before: const add = (a, b) => <CURSOR>\n"
                "After: \\n\n"
                "Your output: a + b\n"
                "(NOT: const add = (a, b) => a + b)\n"
                "(NOT: ```javascript ... ```)"
            )
            user = (
                f"{example}\n\n"
                f"Before: {before}<CURSOR>\n"
                f"After: {after if after else '(end)'}\n"
                "Your output:"
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        if self._has_fim:
            return f"{FIM_PREFIX}{before}{FIM_SUFFIX}{after}{FIM_MIDDLE}"

        return before

    def build_intent_prompt(
        self,
        intent: str,
        language: str = "javascript",
        context: str = "",
    ) -> str:
        intent = intent.strip()
        lang = _lang_name(language)
        surrounding = _truncate((context or "").strip(), 2400, from_end=True)

        if self._has_fim and not self._is_instruct:
            comment = {
                "Python": f"# {intent}",
                "Shell": f"# {intent}",
                "Ruby": f"# {intent}",
            }.get(lang, f"// {intent}")
            prefix = (
                f"{surrounding.rstrip()}\n{comment}\n" if surrounding else f"{comment}\n"
            )
            return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{FIM_MIDDLE}"

        if self._has_chat:
            system = (
                f"You complete {lang} code inside an IDE.\n"
                "Rules:\n"
                "1. Output ONLY the code to insert at the cursor (usually a function body).\n"
                "2. Do NOT repeat the function/const signature if it is already in the file.\n"
                "3. Do NOT use placeholders like 'Your code here', TODO, or '...'.\n"
                "4. Do NOT use markdown fences.\n"
                "5. Write complete, balanced braces/parens; finish every statement.\n"
                "6. Prefer a clear iterative or recursive solution; keep it short.\n"
                "7. Match indentation of the surrounding code.\n"
                "8. Be mathematically correct. Examples:\n"
                "   - factorial: 0! = 1 and 1! = 1 (never return 0 for n===0).\n"
                "   - Prefer: if (n <= 1) return 1; return n * f(n-1);\n"
                "9. Do NOT add useless nested blocks like `{ { return x; } }` — use a single block.\n"
                "10. One clean control-flow structure; no duplicated ifs."
            )
            user = f"Task:\n{intent}"
            if surrounding:
                user += (
                    "\n\nFile context (cursor is inside; insert BODY ONLY):\n"
                    f"{surrounding}\n"
                    "\nCorrect body only:"
                )
            else:
                user += "\n\nCode:"
            # Few-shot when the task looks like factorial — anchors base case.
            if re.search(r"factorial|factorialize", intent + "\n" + surrounding, re.I):
                user += (
                    "\n\nReference (structure only; adapt names):\n"
                    "if (n <= 1) {\n  return 1;\n}\n"
                    "return n * factorial(n - 1);\n"
                )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        return f"{surrounding}\n# {intent}\n" if surrounding else f"# {intent}\n"

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
        """
        Yield (token, is_final).

        stop_on_newline semantics (fixed for FIM):
        - Ignore leading newlines / blank tokens (common FIM preamble)
        - After any non-whitespace has been emitted, stop at the next newline
          (keeps a full first line, not 0–2 words)
        """
        max_tok = max_tokens or self.max_tokens
        self._active_id = msg_id
        self._clear_cancel(msg_id)

        try:
            count = 0
            emitted_any = False
            skipping_lead = True
            buf = ""  # accumulate so we can drop ``` fences before stopping

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

                token = item.text or ""
                if not token:
                    count += 1
                    continue

                count += 1

                # Strip markdown fence openers the model often emits first.
                # e.g. "```javascript\n" must not become the whole completion.
                if skipping_lead:
                    buf += token
                    stripped = buf.lstrip("\r\n")
                    if stripped.startswith("```"):
                        rest = stripped[3:]
                        nl = rest.find("\n")
                        if nl == -1:
                            if count >= max_tok:
                                yield ("", True)
                                return
                            continue
                        stripped = rest[nl + 1 :].lstrip("\r\n")
                    for tag in (
                        "javascript",
                        "typescript",
                        "python",
                        "java",
                        "go",
                        "rust",
                        "tsx",
                        "jsx",
                    ):
                        if stripped.lower().startswith(tag + "\n"):
                            stripped = stripped[len(tag) + 1 :]
                            break
                    if not stripped:
                        buf = ""
                        continue
                    # First real code chunk
                    skipping_lead = False
                    token = stripped
                    buf = ""

                if stop_on_newline:
                    if "\n" in token:
                        before_nl, _sep, _rest = token.partition("\n")
                        # Ignore newline-only if we have no real code yet
                        if not before_nl.strip() and not emitted_any:
                            continue
                        if before_nl:
                            if ";" in before_nl and emitted_any:
                                idx = before_nl.index(";")
                                yield (before_nl[: idx + 1], True)
                            else:
                                yield (before_nl, True)
                        yield ("", True)
                        return

                    if emitted_any and ";" in token:
                        idx = token.index(";")
                        yield (token[: idx + 1], True)
                        return

                    if token.strip():
                        emitted_any = True
                    yield (token, False)
                    if count >= max_tok:
                        yield ("", True)
                        return
                    continue

                yield (token, count >= max_tok)
                if count >= max_tok:
                    return
        except Exception as e:
            sys.stderr.write(f"[mlx] Stream generation error: {e}\n")
            sys.stderr.flush()
            yield ("", True)
        finally:
            self._clear_cancel(msg_id)
            if self._active_id == msg_id:
                self._active_id = None
