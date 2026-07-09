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
        # Tight context = faster prefill (dominant latency cost).
        before = _truncate(before, 2200, from_end=True)
        after = _truncate(after, 400, from_end=False)

        # Native FIM is ~2–3× faster than chat on Instruct models and accurate
        # for mid-expression sites (after `=>`, `=`, `(`, etc.).
        # Mid-identifier after `const name` is ambiguous for pure FIM,
        # so use a compact chat prompt only in that case.
        use_fim = self._has_fim and _fim_site_ok(before)

        if use_fim:
            return f"{FIM_PREFIX}{before}{FIM_SUFFIX}{after}{FIM_MIDDLE}"

        if self._has_chat:
            lang = _lang_name(language)
            system = (
                "IDE code completion. Output ONLY text to insert at <CURSOR>. "
                "No markdown, no explanation, do not repeat code before <CURSOR>."
            )
            user = f"{before}<CURSOR>{after}\nInsert ({lang}):"
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
        surrounding = _truncate((context or "").strip(), 1600, from_end=True)

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
            target = _nearest_signature_name(intent, surrounding)
            system = (
                f"You are a {lang} code completion engine inside an IDE.\n"
                "Output ONLY the code to insert at the cursor.\n"
                "Rules:\n"
                "- No markdown fences, no prose explanations.\n"
                "- No placeholders (no TODO, 'Your code here', stub comments).\n"
                "- Produce complete, runnable code: every try has except/finally, "
                "names you use must be defined or imported, blocks must be indented correctly.\n"
                "- Emit the FULL body through the final return; do not stop after the first "
                "guard if/return.\n"
                "- Never write invalid `return ...; else` (else cannot follow return).\n"
                "- If the request is a multi-line specification, implement the full request "
                "in one coherent snippet.\n"
                "- Do not invent trailing dead code or unfinished function stubs.\n"
                "- Prefer correct, idiomatic code; keep it as short as correctness allows.\n"
                "- Do not paste unrelated functions from the surrounding file."
            )
            user_parts = [f"Request:\n{intent}"]
            if target:
                user_parts.append(f"Current function name: {target}")
            if surrounding:
                user_parts.append(
                    "Surrounding file (for names/types/style only):\n" + surrounding
                )
            user_parts.append("Code to insert:")
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(user_parts)},
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
        stop_when_balanced: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        """
        Yield (token, is_final).

        stop_on_newline: FIM mid-line — first real line only.
        stop_when_balanced: intent bodies — stop once braces/parens balance.
        """
        max_tok = max_tokens or self.max_tokens
        self._active_id = msg_id
        self._clear_cancel(msg_id)

        try:
            count = 0
            emitted_any = False
            skipping_lead = True
            buf = ""
            acc = ""

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
                    skipping_lead = False
                    token = stripped
                    buf = ""

                if stop_on_newline:
                    if "\n" in token:
                        before_nl, _sep, _rest = token.partition("\n")
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

                yield (token, False)
                acc += token
                emitted_any = True

                if stop_when_balanced and _looks_complete_body(acc):
                    yield ("", True)
                    return

                if count >= max_tok:
                    yield ("", True)
                    return
        except Exception as e:
            sys.stderr.write(f"[mlx] Stream generation error: {e}\n")
            sys.stderr.flush()
            yield ("", True)
        finally:
            self._clear_cancel(msg_id)
            if self._active_id == msg_id:
                self._active_id = None


def _fim_site_ok(before: str) -> bool:
    """True when FIM is the better (and faster) choice for this cursor site."""
    s = before.rstrip("\n")
    if not s:
        return True
    if s[-1] in "=.([{,;:+-*/%<>!&|? \t":
        return True
    if s.endswith("=>") or s.endswith("->") or s.endswith("::"):
        return True
    # `const name` without `=` — pure FIM often continues the identifier wrongly
    if re.search(r"(?:const|let|var)\s+[\w$]+$", s):
        return False
    if re.search(r"\.\w+$", s):
        return True
    return True


def _nearest_signature_name(intent: str, surrounding: str) -> str:
    """Best-effort name of the function currently being written."""
    for blob in (intent, surrounding):
        for line in reversed(blob.splitlines()):
            m = re.search(
                r"(?:function\s+|const\s+|let\s+|var\s+|def\s+)([\w$]+)",
                line,
            )
            if m:
                return m.group(1)
            m = re.search(r"([\w$]+)\s*\([^)]*\)\s*(?:=>|\{|:)", line)
            if m and m.group(1) not in ("if", "for", "while", "switch", "catch"):
                return m.group(1)
    return ""


def _looks_complete_body(text: str) -> bool:
    """True when a multi-line body looks finished (early-stop to save decode).

    Conservative: a single guard `if (...) { return ...; }` is NOT enough —
    many real functions start that way and continue (deepClone, deepEqual, etc.).
    """
    t = text.strip()
    if len(t) < 40:
        return False

    # Balanced braces
    depth = 0
    for ch in t:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    # Balanced parens / brackets
    for open_c, close_c in (("(", ")"), ("[", "]")):
        p = 0
        for ch in t:
            if ch == open_c:
                p += 1
            elif ch == close_c:
                p -= 1
                if p < 0:
                    return False
        if p != 0:
            return False

    # Must end on a finished statement
    if re.search(r"\b(for|if|while|function|const|let|var|return|else)\s*$", t):
        return False
    if re.search(r"[=+\-*/%,.]\s*$", t):
        return False
    if not re.search(r"[;}]\s*$", t):
        return False

    if re.search(r"\btry\s*:", t) and not re.search(r"\bexcept\b|\bfinally\b", t):
        return False
    if re.search(r"\btry\s*\{", t) and not re.search(r"\bcatch\b|\bfinally\b", t):
        return False
    if re.search(r"(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:\s*$", t):
        return False

    # Only a single top-level if-return guard → almost certainly incomplete
    # (e.g. deepEqual starts with if (a===b) return true; then more logic).
    stripped = re.sub(r"\s+", " ", t).strip()
    if re.fullmatch(
        r"if\s*\([^)]*\)\s*\{?\s*return\s+[^;]+;\s*\}?",
        stripped,
    ):
        return False
    if re.fullmatch(
        r"if\s*\([^)]*\)\s*return\s+[^;]+;",
        stripped,
    ):
        return False
    # One if-block only, short, one return — treat as guard-only
    if (
        t.count("if") == 1
        and t.count("return") == 1
        and not re.search(r"\b(for|while|const|let|var)\b", t)
        and len(t) < 120
    ):
        return False

    if (
        re.search(r"\b(const|let|var)\s+[\w$]+", t)
        and "return" not in t
        and not re.search(r"\b(throw|console\.|process\.|raise\b|logging\.)", t)
    ):
        return False

    # Substance bar is high: many functions open with several guard returns.
    has_loop = bool(re.search(r"\b(for|while)\b", t))
    returns = t.count("return")
    ends_ok = bool(re.search(r"[;}]\s*$", t))

    if not ends_ok:
        return False

    # Loop bodies: only stop when the *last* statement is a return that comes
    # after the loop closes (not `return false` inside the for-body).
    if has_loop:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if not lines or not lines[-1].startswith("return"):
            return False
        # `}\nreturn ...` — return follows a closed block
        if len(lines) >= 2 and lines[-2].rstrip(";") == "}":
            return True
        return False

    # No loop yet: do not early-stop (guard-only phase of deepEqual/etc.).
    if re.search(r"\bdef\s+\w+", t) and re.search(r"\breturn\b", t) and not re.search(
        r":\s*$", t
    ):
        if re.search(r"\btry\s*:", t):
            return bool(re.search(r"\bexcept\b|\bfinally\b", t)) and len(t) > 100
        return False
    return False
