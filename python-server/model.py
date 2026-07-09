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
import copy
from typing import Any, Callable, List, Optional, Tuple

from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

try:
    from mlx_lm.models.cache import (
        can_trim_prompt_cache,
        make_prompt_cache,
        trim_prompt_cache,
    )

    _HAS_PROMPT_CACHE = True
except ImportError:  # pragma: no cover
    _HAS_PROMPT_CACHE = False


class PrefixKVCache:
    """
    Single-slot prefix KV cache for IDE completions.

    Supports optional draft-model cache for speculative decoding: mlx-lm expects
    prompt_cache = main_layers + draft_layers when draft_model is set.
    """

    def __init__(self) -> None:
        self._tokens: List[int] = []
        self._main: Any = None
        self._draft: Any = None  # list of draft layer caches, or None

    def _combined(self, main: Any, draft: Any) -> Any:
        if draft is None:
            return main
        return list(main) + list(draft)

    def fetch(
        self, tokens: List[int], want_draft: bool
    ) -> Tuple[Any, List[int], str]:
        """Return (cache_or_None, remaining_tokens, hit_kind)."""
        if self._main is None or not self._tokens or not tokens:
            return None, tokens, "miss"
        if want_draft and self._draft is None:
            return None, tokens, "miss"
        if not want_draft and self._draft is not None:
            # Mode switch (spec on→off): safer to miss once
            return None, tokens, "miss"

        n = min(len(self._tokens), len(tokens))
        i = 0
        while i < n and self._tokens[i] == tokens[i]:
            i += 1
        if i == 0:
            return None, tokens, "miss"

        try:
            main = copy.deepcopy(self._main)
            draft = copy.deepcopy(self._draft) if want_draft else None
        except Exception:
            return None, tokens, "miss"

        trim_n = 0
        if i < len(self._tokens):
            trim_n = len(self._tokens) - i
        if i >= len(tokens) and len(tokens) > 1:
            # exact: trim 1 more so we can feed last token
            trim_n = len(self._tokens) - len(tokens) + 1

        if trim_n > 0:
            if not can_trim_prompt_cache(main):
                return None, tokens, "miss"
            if draft is not None and not can_trim_prompt_cache(draft):
                return None, tokens, "miss"
            trim_prompt_cache(main, trim_n)
            if draft is not None:
                trim_prompt_cache(draft, trim_n)

        if i >= len(tokens):
            if len(tokens) > 1:
                return self._combined(main, draft), tokens[-1:], "exact"
            return None, tokens, "miss"

        return self._combined(main, draft), tokens[i:], "prefix"

    def store(self, tokens: List[int], cache: Any, n_main_layers: int, used_draft: bool) -> None:
        if cache is None or not tokens:
            return
        try:
            if used_draft:
                self._main = list(cache[:n_main_layers])
                self._draft = list(cache[n_main_layers:])
            else:
                self._main = list(cache)
                self._draft = None
            self._tokens = list(tokens)
        except Exception:
            self._main = None
            self._draft = None
            self._tokens = []


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

# Smaller draft models for speculative decoding (same family as target).
DRAFT_MODEL_CANDIDATES = [
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-0.5B-Instruct-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-0.5B-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-1.5B-Instruct-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-1.5B-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-Instruct-MLX-4bit",
]

# Phase E: fast mid-line FIM model.
# Prefer 3B Instruct: better mid-line semantics (e.g. `sub` → a-b); base 3B often maps sub→a+b.
FAST_MODEL_CANDIDATES = [
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-Instruct-MLX-4bit",
    Path.home() / ".mlx-models" / "Qwen2.5-Coder-3B-4bit",
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


def resolve_draft_model_path(
    main_path: str,
    explicit: Optional[str] = None,
    enabled: bool = True,
) -> Optional[str]:
    """Pick a smaller draft model for speculative decoding, or None."""
    if not enabled:
        return None
    if explicit is not None and explicit.strip() == "":
        # Explicit empty disables draft
        return None
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir() and (p / "config.json").exists():
            # Don't use the same path as the main model
            if p.resolve() != Path(main_path).expanduser().resolve():
                return str(p)
            sys.stderr.write("[mlx] draft model same as main — ignoring\n")
            return None
        sys.stderr.write(f"[mlx] draft model path missing: {p}\n")

    main = Path(main_path).expanduser().resolve()
    main_name = main.name.lower()

    def _rank(name: str) -> int:
        # Prefer smaller drafts
        if "0.5b" in name:
            return 0
        if "1.5b" in name:
            return 1
        if "3b" in name:
            return 2
        return 9

    # If main is already tiny, skip draft
    if "0.5b" in main_name:
        return None

    candidates = sorted(
        [c for c in DRAFT_MODEL_CANDIDATES if c.is_dir() and (c / "config.json").exists()],
        key=lambda c: _rank(c.name.lower()),
    )
    for c in candidates:
        if c.resolve() == main:
            continue
        # Draft should not be larger than main
        if "7b" in main_name and _rank(c.name.lower()) <= 2:
            return str(c)
        if "3b" in main_name and _rank(c.name.lower()) <= 1:
            return str(c)
        if "7b" not in main_name and "3b" not in main_name:
            if _rank(c.name.lower()) < _rank(main_name):
                return str(c)
    return None


def resolve_fast_model_path(
    quality_path: str,
    explicit: Optional[str] = None,
    enabled: bool = True,
) -> Optional[str]:
    """
    Phase E: smaller model for single-line mid-expression FIM only.

    Returns None when dual routing is off, quality is already small, or no
    distinct fast checkpoint is available.
    """
    if not enabled:
        return None
    if explicit is not None and explicit.strip() == "":
        return None
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir() and (p / "config.json").exists():
            if p.resolve() != Path(quality_path).expanduser().resolve():
                return str(p)
            sys.stderr.write("[mlx] fast model same as quality — dual routing off\n")
            return None
        sys.stderr.write(f"[mlx] fast model path missing: {p}\n")
        return None

    quality = Path(quality_path).expanduser().resolve()
    qname = quality.name.lower()
    # Quality already mid-line sized — no second model.
    if any(tag in qname for tag in ("0.5b", "1.5b", "3b")):
        return None

    for c in FAST_MODEL_CANDIDATES:
        if c.is_dir() and (c / "config.json").exists() and c.resolve() != quality:
            return str(c)
    return None


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
        draft_model_path: Optional[str] = None,
        speculative: bool = True,
        num_draft_tokens: int = 3,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_draft_tokens = max(1, min(int(num_draft_tokens), 8))
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
        self._n_main_layers = len(getattr(self.model, "layers", []) or [])

        self._has_fim = self._detect_fim_support()
        self._has_chat = bool(getattr(self.tokenizer, "chat_template", None))
        name = Path(model_path).name.lower()
        self._is_instruct = "instruct" in name

        # Phase D: optional draft model for speculative decoding
        self.draft_model = None
        self.draft_model_path: Optional[str] = None
        draft_path = resolve_draft_model_path(
            model_path, explicit=draft_model_path, enabled=speculative
        )
        if draft_path:
            try:
                sys.stderr.write(f"[mlx] Loading draft model from: {draft_path}\n")
                sys.stderr.flush()
                d_cfg = {}
                if not _is_already_quantized(draft_path):
                    q = QUANT_CONFIG.get(quantization, QUANT_CONFIG["4bit"])
                    if q:
                        d_cfg["quantization"] = q
                self.draft_model, _draft_tok = load(
                    draft_path,
                    model_config=d_cfg if d_cfg else None,
                )
                self.draft_model_path = draft_path
                sys.stderr.write(
                    f"[mlx] Speculative decoding ON "
                    f"(draft={Path(draft_path).name}, num_draft={self.num_draft_tokens})\n"
                )
                sys.stderr.flush()
            except Exception as e:
                self.draft_model = None
                self.draft_model_path = None
                sys.stderr.write(f"[mlx] Draft model load failed (continuing without): {e}\n")
                sys.stderr.flush()
        else:
            sys.stderr.write("[mlx] Speculative decoding OFF (no draft model)\n")
            sys.stderr.flush()

        sys.stderr.write(
            f"[mlx] Ready fim={'yes' if self._has_fim else 'no'} "
            f"chat={'yes' if self._has_chat else 'no'} "
            f"kind={'instruct' if self._is_instruct else 'base'} "
            f"spec={'yes' if self.draft_model is not None else 'no'}\n"
        )
        sys.stderr.flush()

        self.sampler = make_sampler(temp=temperature, top_p=1.0)

        # Phase C: prefix / KV cache across requests (same quality, less prefill).
        self._prefix_cache: Optional[PrefixKVCache] = None
        if _HAS_PROMPT_CACHE:
            self._prefix_cache = PrefixKVCache()
            sys.stderr.write("[mlx] Prefix KV cache enabled (Phase C)\n")
            sys.stderr.flush()
        else:
            sys.stderr.write("[mlx] Prefix KV cache unavailable (upgrade mlx-lm)\n")
            sys.stderr.flush()

    def _encode_prompt(self, prompt: str) -> List[int]:
        """Tokenize like mlx_lm.stream_generate does."""
        tok = self.tokenizer
        bos = getattr(tok, "bos_token", None)
        add_special = bos is None or not prompt.startswith(str(bos))
        ids = tok.encode(prompt, add_special_tokens=add_special)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return list(ids)

    def _prepare_prompt_cache(
        self, prompt: str
    ) -> Tuple[Any, List[int], List[int], str]:
        """
        Resolve a prompt_cache + remaining tokens for generation.

        Returns:
            cache, remaining_token_ids, full_token_ids, hit_kind
            hit_kind: "exact" | "prefix" | "miss" | "off"
        """
        full = self._encode_prompt(prompt)
        if not full:
            full = [0]

        if self._prefix_cache is None:
            return None, full, full, "off"

        want_draft = self.draft_model is not None
        try:
            cache, remaining, hit = self._prefix_cache.fetch(full, want_draft=want_draft)
        except Exception as e:
            sys.stderr.write(f"[mlx] cache fetch error: {e}\n")
            sys.stderr.flush()
            return None, full, full, "miss"

        if cache is None:
            return None, full, full, "miss"

        return cache, list(remaining), full, hit

    def _new_prompt_cache(self) -> Any:
        """Fresh cache for main model, or main+draft when speculative is on."""
        if not _HAS_PROMPT_CACHE:
            return None
        main = make_prompt_cache(self.model)
        if self.draft_model is None:
            return main
        draft = make_prompt_cache(self.draft_model)
        return list(main) + list(draft)

    def _store_prompt_cache(
        self,
        full_tokens: List[int],
        prompt_cache: Any,
        generated_tokens: int,
    ) -> None:
        """Trim decode tokens off the cache and store prompt-only state."""
        if self._prefix_cache is None or prompt_cache is None:
            return
        try:
            # Speculative path: combined main+draft list; trim each half equally
            used_draft = self.draft_model is not None and len(prompt_cache) > self._n_main_layers
            if generated_tokens > 0:
                if used_draft:
                    main = list(prompt_cache[: self._n_main_layers])
                    draft = list(prompt_cache[self._n_main_layers :])
                    if can_trim_prompt_cache(main):
                        trim_prompt_cache(main, int(generated_tokens))
                    if can_trim_prompt_cache(draft):
                        trim_prompt_cache(draft, int(generated_tokens))
                    prompt_cache = list(main) + list(draft)
                elif can_trim_prompt_cache(prompt_cache):
                    trim_prompt_cache(prompt_cache, int(generated_tokens))
            self._prefix_cache.store(
                full_tokens,
                prompt_cache,
                n_main_layers=self._n_main_layers or len(prompt_cache),
                used_draft=used_draft,
            )
        except Exception as e:
            sys.stderr.write(f"[mlx] cache store error: {e}\n")
            sys.stderr.flush()

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
                "- No markdown fences, no prose, no comments that explain the code, "
                "no sentences like 'This code defines…'.\n"
                "- Stop after the code for THIS function/block. Do not start another "
                "function, class, or file section.\n"
                "- No placeholders (no TODO, 'Your code here', stub comments).\n"
                "- NEVER write usage examples, demos, sample data, test harnesses, "
                "or toy async tasks (no setTimeout demos, no 'Task 1 completed', "
                "no arrays of sample callbacks).\n"
                "- NEVER redeclare function parameters or the function itself; "
                "parameters already exist in scope — use them.\n"
                "- NEVER emit export default, module.exports, imports, or any code "
                "outside this function body. Do not close the outer function with "
                "an extra }; if the signature is already above the cursor.\n"
                "- Do not invent loggers, globals, or helpers that are not in scope "
                "(no t.log, logger.x, undefined single-letter objects).\n"
                "- Use parameters only as their names imply and as surrounding code shows. "
                "If a parameter is named `tasks` and used with call syntax elsewhere, "
                "treat elements as callables (e.g. task()), not as objects with invented "
                ".run()/.name unless those appear in the surrounding file.\n"
                "- Prefer standard library APIs (Promise, Array, Map, Set, Math). "
                "Do not invent fake frameworks.\n"
                "- Do not fake async work with setTimeout sleeps, Wait for N second "
                "comments, or console.log Task completed loops. Implement real logic.\n"
                "- If a parameter is named concurrency/limit/max, you MUST use it to "
                "bound parallelism (e.g. Promise.race / worker slots). Never ignore it.\n"
                "- No dead code after return (no } else after return true).\n"
                "- Produce complete, runnable implementation code: every try has "
                "except/finally, names you use must be defined or imported, "
                "blocks must be indented correctly.\n"
                "- Emit the FULL body through the final return; do not stop after the first "
                "guard if/return.\n"
                "- Never write invalid `return ...; else` (else cannot follow return).\n"
                "- If the request is a multi-line specification, implement the full request "
                "in one coherent snippet.\n"
                "- Do not invent trailing dead code or unfinished function stubs.\n"
                "- Prefer correct, idiomatic code; keep it as short as correctness allows.\n"
                "- Do not paste unrelated functions from the surrounding file.\n"
                "- Match the indentation style of the surrounding code; do not nest wildly."
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
        stop_on_derail: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        """
        Yield (token, is_final).

        stop_on_newline: FIM mid-line — first real line only.
        stop_when_balanced: intent bodies — stop once braces/parens balance.
        stop_on_derail: multi-line — stop on prose / brace spam / junk (default on).
        """
        max_tok = max_tokens or self.max_tokens
        self._active_id = msg_id
        self._clear_cancel(msg_id)

        prompt_cache = None
        full_tokens: List[int] = []
        gen_tokens = 0

        try:
            prompt_cache, remaining, full_tokens, hit = self._prepare_prompt_cache(prompt)
            # Always own a cache object so we can store it after generation.
            if prompt_cache is None and _HAS_PROMPT_CACHE:
                prompt_cache = self._new_prompt_cache()

            use_spec = self.draft_model is not None
            sys.stderr.write(
                f"[mlx] cache={hit} prompt_tok={len(full_tokens)} "
                f"prefill_tok={len(remaining)} spec={'on' if use_spec else 'off'}\n"
            )
            sys.stderr.flush()

            stream_kwargs: dict = {
                "max_tokens": max_tok,
                "sampler": self.sampler,
            }
            if prompt_cache is not None:
                stream_kwargs["prompt_cache"] = prompt_cache
            if use_spec:
                stream_kwargs["draft_model"] = self.draft_model
                stream_kwargs["num_draft_tokens"] = self.num_draft_tokens

            count = 0
            emitted_any = False
            skipping_lead = True
            buf = ""
            acc = ""
            draft_accepts = 0
            _spec_logged = False

            def _log_spec() -> None:
                nonlocal _spec_logged
                if _spec_logged or not use_spec or gen_tokens <= 0:
                    return
                _spec_logged = True
                sys.stderr.write(
                    f"[mlx] spec draft_accepts≈{draft_accepts}/{gen_tokens} "
                    f"({100.0 * draft_accepts / max(gen_tokens, 1):.0f}%)\n"
                )
                sys.stderr.flush()

            def _finish_store() -> None:
                _log_spec()
                self._store_prompt_cache(full_tokens, prompt_cache, gen_tokens)

            for item in stream_generate(
                self.model,
                self.tokenizer,
                prompt=remaining,
                **stream_kwargs,
            ):
                # generation_tokens tracks decode length for accurate cache trim
                if getattr(item, "generation_tokens", None):
                    gen_tokens = int(item.generation_tokens)
                if getattr(item, "from_draft", False):
                    draft_accepts += 1

                if self._is_cancelled(msg_id) or (should_stop and should_stop()):
                    sys.stderr.write(f"[mlx] cancelled id={msg_id} after {count} tok\n")
                    sys.stderr.flush()
                    _finish_store()
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
                                _finish_store()
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
                        # Store BEFORE final yields — consumer may break on is_final.
                        _finish_store()
                        if before_nl:
                            if ";" in before_nl and emitted_any:
                                idx = before_nl.index(";")
                                yield (before_nl[: idx + 1], True)
                            else:
                                yield (before_nl, True)
                        else:
                            yield ("", True)
                        return

                    if emitted_any and ";" in token:
                        idx = token.index(";")
                        _finish_store()
                        yield (token[: idx + 1], True)
                        return

                    if token.strip():
                        emitted_any = True
                    yield (token, False)
                    if count >= max_tok:
                        _finish_store()
                        yield ("", True)
                        return
                    continue

                # Multi-line path: yield token then check derail / balance.
                yield (token, False)
                acc += token
                emitted_any = True

                if stop_on_derail and _looks_derailed(acc):
                    sys.stderr.write(f"[mlx] derail-stop after {count} tok\n")
                    sys.stderr.flush()
                    _finish_store()
                    yield ("", True)
                    return

                if stop_when_balanced and _looks_complete_body(acc):
                    _finish_store()
                    yield ("", True)
                    return

                if count >= max_tok:
                    _finish_store()
                    yield ("", True)
                    return

            _finish_store()
        except Exception as e:
            sys.stderr.write(f"[mlx] Stream generation error: {e}\n")
            sys.stderr.flush()
            yield ("", True)
        finally:
            self._clear_cancel(msg_id)
            if self._active_id == msg_id:
                self._active_id = None


class DualModelRouter:
    """
    Phase E: route single-line mid-expression FIM to a smaller "fast" model;
    keep multi-line FIM + intent on the quality model (typically 7B + draft).

    Quality is unchanged on hard paths; mid-line gains speed from a lighter prefill.
    """

    def __init__(
        self,
        quality: "ModelEngine",
        fast: Optional["ModelEngine"] = None,
    ) -> None:
        self.quality = quality
        self.fast = fast
        self.max_tokens = quality.max_tokens

    @property
    def dual_enabled(self) -> bool:
        return self.fast is not None

    @property
    def model_path(self) -> str:
        return self.quality.model_path

    @property
    def draft_model(self) -> Any:
        return self.quality.draft_model

    @property
    def draft_model_path(self) -> Optional[str]:
        return self.quality.draft_model_path

    @property
    def num_draft_tokens(self) -> int:
        return self.quality.num_draft_tokens

    def request_cancel(self, msg_id: Optional[int] = None) -> None:
        self.quality.request_cancel(msg_id)
        if self.fast is not None:
            self.fast.request_cancel(msg_id)

    def pick(self, *, is_intent: bool, multi_line: bool) -> "ModelEngine":
        """Quality for hard paths; fast for single-line FIM only."""
        if is_intent or multi_line or self.fast is None:
            return self.quality
        return self.fast

    def route_name(self, engine: "ModelEngine") -> str:
        if self.fast is not None and engine is self.fast:
            return "fast"
        return "quality"

    @staticmethod
    def create(
        quality_path: str,
        quantization: str = "4bit",
        max_tokens: int = 40,
        temperature: float = 0.0,
        draft_model_path: Optional[str] = None,
        speculative: bool = True,
        num_draft_tokens: int = 3,
        fast_model_path: Optional[str] = None,
        dual_model: bool = True,
    ) -> "DualModelRouter":
        quality = ModelEngine(
            model_path=quality_path,
            quantization=quantization,
            max_tokens=max_tokens,
            temperature=temperature,
            draft_model_path=draft_model_path,
            speculative=speculative,
            num_draft_tokens=num_draft_tokens,
        )

        fast_path = resolve_fast_model_path(
            quality.model_path,
            explicit=fast_model_path,
            enabled=dual_model,
        )
        fast: Optional[ModelEngine] = None
        if fast_path:
            try:
                sys.stderr.write(
                    f"[mlx] Phase E: loading fast mid-line model from: {fast_path}\n"
                )
                sys.stderr.flush()
                # Fast path: short FIM only — no speculative draft (decode is tiny).
                fast = ModelEngine(
                    model_path=fast_path,
                    quantization=quantization,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    draft_model_path="",
                    speculative=False,
                    num_draft_tokens=num_draft_tokens,
                )
                sys.stderr.write(
                    f"[mlx] Phase E dual routing ON "
                    f"(fast={Path(fast_path).name}, quality={Path(quality.model_path).name})\n"
                )
                sys.stderr.flush()
            except Exception as e:
                fast = None
                sys.stderr.write(
                    f"[mlx] Phase E fast model load failed (quality-only): {e}\n"
                )
                sys.stderr.flush()
        else:
            sys.stderr.write("[mlx] Phase E dual routing OFF (no distinct fast model)\n")
            sys.stderr.flush()

        return DualModelRouter(quality=quality, fast=fast)


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


def _is_prose_line(line: str) -> bool:
    """True when a line looks like natural-language explanation, not code."""
    t = line.strip()
    if not t or len(t) < 12:
        return False
    # Clear code markers
    if re.match(
        r"^(const|let|var|function|async|await|return|if|else|for|while|switch|"
        r"case|break|continue|try|catch|finally|throw|class|export|import|from|"
        r"def|class|pass|yield|with|except|raise|new|this\.|super\.|"
        r"public|private|protected|static|interface|type|enum|package|using|"
        r"fn|func|impl|struct|match|loop|mut|pub)\b",
        t,
    ):
        return False
    if t[0] in "{}()[];.,/*#`":
        return False
    if re.search(r"[{};=<>]|=>|::|\(\)|^\s*\.", t):
        # has code punctuation — usually code
        if not re.match(
            r"^(this|the|here|note|above|below|we|you|it|in)\b", t, re.I
        ):
            return False
    # Explicit prose openers / textbook gloss
    if re.match(
        r"^(this|the|here|note|example|above|below|we |you |it |in this|"
        r"the following|as you can|explanation|description)\b",
        t,
        re.I,
    ):
        return True
    if re.search(
        r"\b(defines?|implements?|takes? a|parameter|method that|"
        r"function that|class that|as input|as output|is called when)\b",
        t,
        re.I,
    ) and not re.search(r"[{};=]", t):
        return True
    # Long sentence-like line without code punctuation
    words = t.split()
    if len(t) > 55 and len(words) >= 8 and not re.search(r"[{};=<>()]", t):
        return True
    return False


def _looks_derailed(text: str) -> bool:
    """True when multi-line generation has gone off the rails — stop early."""
    if not text or len(text) < 20:
        return False

    # Markdown fence mid-body
    if "\n```" in text or text.startswith("```"):
        return True

    # Demo / usage-example leakage (promisePool → sample tasks, etc.)
    if re.search(r"Task\s*\d+\s+completed", text, re.I):
        return True
    if re.search(r"Task\s*\$\{[^}]+\}\s*completed", text, re.I):
        return True
    if re.search(r"resolve\(\s*['\"`]Task\s*\d+", text, re.I):
        return True
    if re.search(r"['\"`]Task\s*\d+\s+completed['\"`]", text, re.I):
        return True
    if re.search(r"//\s*simulate\b", text, re.I) or re.search(
        r"//\s*wait for \d", text, re.I
    ):
        return True
    # Sleep-loop toy implementations (3+ setTimeouts or delay spam)
    if len(re.findall(r"\bsetTimeout\s*\(", text)) >= 3:
        return True
    if len(re.findall(r"setTimeout\s*\(\s*resolve\s*,\s*\d+\s*\)", text)) >= 3:
        return True
    if len(re.findall(r"\bsetTimeout\s*\(", text)) >= 2 and re.search(
        r"console\.log\s*\(\s*[`'\"].*Task", text, re.I
    ):
        return True
    # Sample task arrays: const tasks = [ async () => { setTimeout...
    if re.search(
        r"(?:const|let|var)\s+tasks\s*=\s*\[\s*(?:async\s*)?\([^)]*\)\s*=>",
        text,
    ) and re.search(r"setTimeout", text):
        return True
    if (text.count("setTimeout") >= 2) and (
        len(re.findall(r"async\s*\([^)]*\)\s*=>", text)) >= 2
    ):
        return True
    # Dead control flow: return …; } else
    if re.search(r"\breturn\b[^;\n]*;\s*\n\s*\}\s*else\b", text):
        return True
    # File-level leakage after body
    if re.search(r"\n\s*export\s+default\b", text):
        return True
    if re.search(r"\n\s*module\.exports\b", text):
        return True
    if re.search(r"\n\s*export\s+(?:async\s+)?(?:function|class|const)\b", text):
        return True
    # Collapsed closer soup
    if re.search(r"\}\s*\}\s*\)", text) or re.search(r"\}\s*;\s*export\b", text):
        return True
    # Glued keywords (awaitPromise, }if)
    if re.search(r"\bawait[A-Za-z_$]", text):
        return True
    if re.search(r"\}if\b|\}for\b|\}while\b|\}return\b", text):
        return True
    # Invented single-letter logger: t.log / l.error (never console)
    if re.search(
        r"(?<![A-Za-z0-9_$])[a-z]\.(?:log|error|warn|info|debug)\s*\(",
        text,
    ):
        return True

    lines = text.splitlines()
    # Prose line
    for ln in lines:
        if _is_prose_line(ln):
            return True

    # Consecutive duplicate non-trivial code lines
    prev = None
    for ln in lines:
        s = ln.strip()
        if s and prev is not None and s == prev and re.search(r"[A-Za-z]", s) and len(s) > 8:
            return True
        if s:
            prev = s

    # Brace-close spam: 4+ nearly consecutive lines that are only `}`
    close_run = 0
    for ln in lines:
        if re.fullmatch(r"\s*\}[;,)]*\s*", ln):
            close_run += 1
            if close_run >= 4:
                return True
        elif ln.strip():
            close_run = 0

    # Runaway indentation (model nesting collapse)
    indents = []
    for ln in lines:
        if not ln.strip():
            continue
        indents.append(len(ln) - len(ln.lstrip(" \t")))
    if indents:
        base = indents[0]
        if any(i > base + 24 or i > 48 for i in indents):
            return True

    # Too many unmatched closes relative to opens in the generation alone
    depth = 0
    min_depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            min_depth = min(min_depth, depth)
    if min_depth <= -4:
        return True

    return False


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
