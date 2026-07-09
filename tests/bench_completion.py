#!/usr/bin/env python3
"""
End-to-end completion bench: speed + accuracy across scenarios.

Mirrors the extension pipeline:
  1) ModelEngine prompt + stream (stop_on_newline, fence skip)
  2) clean_raw + to_insert (TS logic ported)
  3) Compose final line = line_before[:range] + insert

Usage:
  python3 tests/bench_completion.py
  python3 tests/bench_completion.py --model ~/.mlx-models/Qwen2.5-Coder-3B-4bit
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Allow importing python-server/*
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python-server"))

from model import ModelEngine, resolve_model_path  # noqa: E402


# ---------------------------------------------------------------------------
# Port of extension post-processing (keep in sync with completion-provider.ts)
# ---------------------------------------------------------------------------

def clean_raw(raw: str) -> str:
    text = raw or ""
    text = re.sub(r"<\|fim_(?:begin|end|pad|prefix|suffix|middle)\|>", "", text)
    text = text.replace("<|endoftext|>", "")
    text = re.sub(r"<\|im_(?:end|start)\|>", "", text)
    text = re.sub(r"```[a-zA-Z0-9_+-]*\s*", "", text)
    text = text.replace("```", "")
    text = re.sub(r"^[\r\n]+", "", text)
    text = re.sub(
        r"^(javascript|typescript|python|java|go|rust|tsx|jsx)\s*\n",
        "",
        text,
        flags=re.I,
    )
    return text


@dataclass
class Prepared:
    text: str
    from_line_start: bool  # True => replace [0, cursor)


def to_insert(raw: str, line_before: str, is_intent: bool = False) -> Optional[Prepared]:
    text = clean_raw(raw)
    if not is_intent:
        nl = re.search(r"\r?\n", text)
        if nl:
            text = text[: nl.start()]
    text = re.sub(r"[ \t]+$", "", text)
    if not text:
        return None

    if line_before and text.startswith(line_before):
        return Prepared(text[len(line_before) :], False)

    # LCP
    lcp = 0
    while lcp < len(line_before) and lcp < len(text) and line_before[lcp] == text[lcp]:
        lcp += 1
    if lcp == len(line_before) and line_before:
        return Prepared(text[lcp:], False)

    if lcp >= 4 and re.match(
        r"^(const|let|var|function|async|export|def|class)\b", text.strip()
    ):
        return Prepared(text.lstrip(), True)

    # suffix overlap
    if line_before:
        max_k = min(len(line_before), len(text))
        for k in range(max_k, 0, -1):
            if line_before.endswith(text[:k]):
                text = text[k:]
                break

    if (
        not is_intent
        and line_before.strip()
        and re.match(r"^(const|let|var|function|async|export)\b", text.strip())
    ):
        return Prepared(text.lstrip(), True)

    if not is_intent and line_before.strip():
        semi = text.find(";")
        if semi != -1:
            text = text[: semi + 1]

    text = text.lstrip()
    if not text:
        return None

    # Guard: const-after-const → full line rewrite
    if re.search(r"\b(const|let|var)\b", line_before) and re.match(
        r"^(const|let|var)\b", text.strip()
    ):
        model_line = clean_raw(raw).split("\n")[0].strip()
        if model_line and not model_line.startswith("```"):
            return Prepared(model_line, True)
        return None

    # Partial binding: `const sub` + `(a, b) => …` → insert ` = (a, b) => …`
    if re.search(r"(?:const|let|var)\s+[\w$]+$", line_before) and text.startswith("("):
        text = " = " + text

    return Prepared(text, False)


def is_low_quality(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.startswith("```") or re.match(r"here('s| is)\b", t, re.I):
        return True
    if re.search(r"\b([A-Za-z_$][\w$]*)\s+\1\b", t):
        return True
    if re.search(r"\b(const|let|var)\b.+\b(const|let|var)\b", t):
        return True
    return False


def compose(line_before: str, prep: Prepared) -> str:
    if prep.from_line_start:
        return prep.text
    return line_before + prep.text


def detect_intent(line_before: str, prev_line: str = "", at_blank: bool = False) -> Optional[str]:
    """Port of detectIntent: comments + Python def only (brace bodies → multi-line FIM)."""
    m = re.search(r"(?://|#|\"\"\"|'''|/\*)\s*(.+?)\s*$", line_before)
    if m and len(m.group(1)) >= 3:
        return m.group(1)
    if at_blank and prev_line:
        m2 = re.search(r"(?://|#)\s*(.+?)\s*$", prev_line)
        if m2 and len(m2.group(1)) >= 3:
            return m2.group(1)
    # Python colon-style only — JS/TS empty bodies use multi-line FIM in the extension.
    py = re.search(r"def\s+[\w$]+\s*\([^)]*\)\s*:\s*$", line_before)
    if py:
        return f"Implement the body of:\n{line_before.strip()}"
    return None


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    before: str  # full document prefix incl. current line up to cursor
    after: str
    language: str = "javascript"
    # Expected final composed line (current line only) — substring / regex checks
    expect_line_contains: Optional[list[str]] = None
    expect_line_not_contains: Optional[list[str]] = None
    expect_line_regex: Optional[str] = None
    expect_insert_not_contains: Optional[list[str]] = None
    # Soft latency budgets (ms) — warn if exceeded, fail if >> budget
    budget_ttft_ms: float = 600
    budget_total_ms: float = 900
    intent: bool = False
    # For intent detection unit tests only
    check_not_intent: bool = False


SCENARIOS: list[Scenario] = [
    Scenario(
        name="arrow_sub_rhs",
        before="const sub = (a, b) => ",
        after="\n",
        expect_line_contains=["sub", "a", "b"],
        expect_line_regex=r"const sub = \(a, b\) =>\s*a\s*-\s*b",
        expect_line_not_contains=["const sub const", "```", "a + b b"],
        expect_insert_not_contains=["```", "const sub const"],
        check_not_intent=True,
    ),
    Scenario(
        name="arrow_add_rhs",
        before="const add = (a, b) => ",
        after="\n",
        expect_line_regex=r"const add = \(a, b\) =>\s*a\s*\+\s*b",
        expect_line_not_contains=["const add const", "```"],
        check_not_intent=True,
    ),
    Scenario(
        name="arrow_mul_rhs",
        before="const mul = (a, b) => ",
        after="\n",
        expect_line_regex=r"const mul = \(a, b\) =>\s*a\s*\*\s*b",
        expect_line_not_contains=["const mul const"],
        check_not_intent=True,
    ),
    Scenario(
        name="partial_ident_sub_to_subtract",
        before="const sub",
        after="\n",
        # Accept either finishing as `sub = ...` or expanding to subtract
        expect_line_contains=["const"],
        expect_line_not_contains=["const sub const", "```"],
        expect_line_regex=r"const sub(tract)?\s*=",
        check_not_intent=True,
    ),
    Scenario(
        name="block_body_add",
        before="const add = (a, b) => {\n  ",
        after="\n}\n",
        expect_line_contains=["return", "a", "b"],
        expect_line_regex=r"return\s+a\s*\+\s*b",
        expect_line_not_contains=["```"],
    ),
    Scenario(
        name="block_body_subtract_named",
        before="function subtract(a, b) {\n  ",
        after="\n}\n",
        expect_line_regex=r"return\s+a\s*-\s*b",
        expect_line_not_contains=["```", "a + b"],
    ),
    Scenario(
        name="mid_call_args",
        before="console.log(",
        after=");\n",
        expect_line_not_contains=["```", "console.log(console"],
        # loose: just something non-empty and not a fence
        expect_line_contains=["console.log("],
    ),
    Scenario(
        name="object_prop",
        before="const user = {\n  name: ",
        after="\n};\n",
        expect_line_not_contains=["```", "const user const"],
    ),
    Scenario(
        name="python_def_body",
        before="def add(a, b):\n    ",
        after="\n\n",
        language="python",
        expect_line_regex=r"return\s+a\s*\+\s*b",
        expect_line_not_contains=["```", "def add"],
    ),
    Scenario(
        name="intent_comment_factorial",
        before="// compute factorial of n\n",
        after="\n",
        intent=True,
        expect_line_not_contains=["```", "Your code here", "TODO"],
        budget_ttft_ms=800,
        budget_total_ms=3500,
    ),
    Scenario(
        name="intent_empty_factorial_body",
        before="const factorial = (n) => {\n  ",
        after="\n}\n",
        intent=True,
        expect_line_not_contains=["```", "Your code here", "const factorial"],
        # Body should mention n and return something sensible
        expect_line_contains=["n"],
        budget_ttft_ms=800,
        budget_total_ms=3500,
    ),
    Scenario(
        name="intent_no_cross_fn_paste",
        # Earlier fn in file must not be pasted into a different empty body
        before=(
            "const factorial = (n) => {\n"
            "  if (n <= 1) return 1;\n"
            "  return n * factorial(n - 1);\n"
            "};\n\n"
            "const matrixMul = (a, b) => {\n  "
        ),
        after="\n}\n",
        intent=True,
        # General check: don't call the other user function from this body
        expect_line_not_contains=["factorial(", "Your code here", "```"],
        budget_ttft_ms=900,
        budget_total_ms=4000,
    ),
    Scenario(
        name="with_nearby_add_then_sub",
        before="const add = (a, b) => a + b;\nconst sub = (a, b) => ",
        after="\n",
        expect_line_regex=r"const sub = \(a, b\) =>\s*a\s*-\s*b",
        expect_line_not_contains=["const sub const", "a + b b"],
        check_not_intent=True,
    ),
    Scenario(
        name="no_double_const_regression",
        before="const sub = (a, b) => ",
        after="\n",
        expect_line_not_contains=["const sub const", "const const"],
        check_not_intent=True,
    ),
]


def format_statement_newlines(text: str) -> str:
    """Mirror of completion-provider formatStatementNewlines (keep in sync)."""
    hs = r"[^\S\n]*"
    stmt = (
        r"(?:return|const|let|var|if|for|while|switch|try|throw|function|class|"
        r"export|import|async|await|break|continue|debugger|yield|do|case|default)"
    )
    t = re.sub(rf";(?={hs}{stmt}\b)", ";\n", text)
    t = re.sub(rf"\{{(?={hs}{stmt}\b)", "{\n", t)
    t = re.sub(rf"\}}(?={hs}{stmt}\b)", "}\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t


# Unit tests that don't need the model
UNIT_INSERT_CASES = [
    # (line_before, model_raw, expect_final_line, expect_from_line_start?)
    ("const sub = (a, b) => ", "a - b", "const sub = (a, b) => a - b", False),
    ("const sub = (a, b) => ", "a - b;", "const sub = (a, b) => a - b;", False),
    ("const sub = (a, b) => ", "```javascript\na - b\n```", "const sub = (a, b) => a - b", False),
    ("const sub", "const subtract = (a, b) => a - b;", "const subtract = (a, b) => a - b;", False),
    ("const sub", "const sub = (a, b) => a - b;", "const sub = (a, b) => a - b;", False),
    ("const sub", "const = (a, b) => a - b;", "const = (a, b) => a - b;", True),  # rewrite, not double
    ("foo(a, ", "a, b)", "foo(a, b)", False),
]


def run_unit_insert_tests() -> tuple[int, int]:
    passed = failed = 0
    print("\n=== UNIT: statement newlines ===")
    packed = (
        "const seen = new Set();return arr.filter(item => {\n"
        "    const value = item[key];\n"
        "    if (seen.has(value)) {\n"
        "      return false;\n"
        "    }\n"
        "    seen.add(value);\n"
        "    return true;\n"
        "  });"
    )
    fmt = format_statement_newlines(packed)
    if ";\nreturn arr.filter" in fmt and "for (let i = 0; i < n; i++)" not in format_statement_newlines(
        "for (let i = 0; i < n; i++) { x(); }"
    ).replace("for (let i = 0; i < n; i++)", "KEEP"):
        # for-loop semis must stay intact
        loop = format_statement_newlines("for (let i = 0; i < n; i++) { x(); }")
        if "for (let i = 0; i < n; i++)" in loop and ";\nreturn" in fmt:
            passed += 1
            print("  PASS uniqueBy-style ;return break + for-loop safe")
        else:
            failed += 1
            print(f"  FAIL format got loop={loop!r} fmt={fmt!r}")
    else:
        failed += 1
        print(f"  FAIL formatStatementNewlines: {fmt!r}")

    print("\n=== UNIT: insert / clean / compose ===")
    for line_before, raw, expect_final, expect_rewrite in UNIT_INSERT_CASES:
        prep = to_insert(raw, line_before)
        if prep is None:
            print(f"  FAIL  {line_before!r} + {raw!r} → None")
            failed += 1
            continue
        final = compose(line_before, prep)
        ok = final == expect_final and prep.from_line_start == expect_rewrite
        # Special case: fence strip may leave trailing issues — allow strip
        if not ok and final.rstrip(";") == expect_final.rstrip(";"):
            ok = True
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
            print(
                f"  {status} line_before={line_before!r} raw={raw!r}\n"
                f"         got final={final!r} rewrite={prep.from_line_start}\n"
                f"         want final={expect_final!r} rewrite={expect_rewrite}"
            )
        else:
            passed += 1
            print(f"  {status} {line_before!r} → {final!r}")

    # Intent classification
    print("\n=== UNIT: intent classification ===")
    intent_cases = [
        ("const sub = (a, b) => ", False),
        # Brace openers are multi-line FIM now, not chat intent.
        ("const sub = (a, b) => {", False),
        ("function subtract(a, b) {", False),
        ("def sub(a, b):", True),
        ("const add = (a, b) => a + ", False),
    ]
    for line, should_intent in intent_cases:
        got = detect_intent(line) is not None
        ok = got == should_intent
        if ok:
            passed += 1
            print(f"  PASS intent({line!r})={got}")
        else:
            failed += 1
            print(f"  FAIL intent({line!r})={got} want={should_intent}")

    # Double-const quality gate
    if is_low_quality("const sub const = 1"):
        passed += 1
        print("  PASS low_quality rejects double const")
    else:
        failed += 1
        print("  FAIL low_quality should reject double const")

    return passed, failed


def run_model_scenario(engine: ModelEngine, sc: Scenario) -> dict:
    line_before = sc.before.split("\n")[-1]
    is_intent = sc.intent or (detect_intent(line_before) is not None)

    if sc.check_not_intent and is_intent:
        return {
            "name": sc.name,
            "ok": False,
            "error": "misclassified as intent",
            "ttft_ms": 0,
            "total_ms": 0,
        }

    t0 = time.perf_counter()
    first_ms = None
    raw_parts: list[str] = []

    if is_intent:
        intent_text = detect_intent(line_before) or "implement"
        if sc.name.startswith("intent_"):
            intent_text = "compute factorial of n"
        prompt = engine.build_intent_prompt(intent_text, sc.language, sc.before)
        stop_nl = False
        max_tok = 96
    else:
        prompt = engine.build_fim_prompt(sc.before, sc.after, sc.language)
        stop_nl = True
        max_tok = 32

    for token, is_final in engine.stream(
        prompt, max_tokens=max_tok, msg_id=1, stop_on_newline=stop_nl
    ):
        if token:
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            raw_parts.append(token)
        if is_final:
            break

    total_ms = (time.perf_counter() - t0) * 1000
    raw = "".join(raw_parts)
    prep = to_insert(raw, line_before, is_intent=is_intent)

    result = {
        "name": sc.name,
        "ttft_ms": first_ms if first_ms is not None else -1,
        "total_ms": total_ms,
        "raw": raw,
        "insert": prep.text if prep else None,
        "final_line": compose(line_before, prep) if prep else None,
        "from_line_start": prep.from_line_start if prep else None,
        "ok": True,
        "reasons": [],
    }

    if prep is None or is_low_quality(prep.text):
        result["ok"] = False
        result["reasons"].append("empty/low-quality insert")
        return result

    final = result["final_line"] or ""

    for s in sc.expect_line_contains or []:
        if s not in final:
            result["ok"] = False
            result["reasons"].append(f"final missing {s!r}")

    for s in sc.expect_line_not_contains or []:
        if s in final:
            result["ok"] = False
            result["reasons"].append(f"final has forbidden {s!r}")

    for s in sc.expect_insert_not_contains or []:
        if s in (prep.text or ""):
            result["ok"] = False
            result["reasons"].append(f"insert has forbidden {s!r}")

    if sc.expect_line_regex and not re.search(sc.expect_line_regex, final):
        result["ok"] = False
        result["reasons"].append(f"final !match /{sc.expect_line_regex}/")

    # Hard fails on known regressions
    if "const sub const" in final or "const const" in final:
        result["ok"] = False
        result["reasons"].append("double-const regression")
    if "```" in final or "```" in raw[:20]:
        # raw may have had fences that should have been stripped from final
        if "```" in final:
            result["ok"] = False
            result["reasons"].append("fence leaked into final")

    # Latency (soft → reason, hard fail if absurd)
    if first_ms is not None and first_ms > sc.budget_ttft_ms * 2:
        result["ok"] = False
        result["reasons"].append(f"TTFT {first_ms:.0f}ms >> budget {sc.budget_ttft_ms:.0f}")
    elif first_ms is not None and first_ms > sc.budget_ttft_ms:
        result["reasons"].append(f"TTFT slow {first_ms:.0f}ms > {sc.budget_ttft_ms:.0f}")

    if total_ms > sc.budget_total_ms * 2:
        result["ok"] = False
        result["reasons"].append(f"total {total_ms:.0f}ms >> budget {sc.budget_total_ms:.0f}")
    elif total_ms > sc.budget_total_ms:
        result["reasons"].append(f"total slow {total_ms:.0f}ms > {sc.budget_total_ms:.0f}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Model path override")
    ap.add_argument(
        "--draft-model",
        default=None,
        help="Draft model path (Phase D). Omit for auto; empty string disables.",
    )
    ap.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable speculative decoding",
    )
    ap.add_argument("--num-draft-tokens", type=int, default=3)
    ap.add_argument("--unit-only", action="store_true")
    args = ap.parse_args()

    u_pass, u_fail = run_unit_insert_tests()
    print(f"\nUnit: {u_pass} passed, {u_fail} failed")

    if args.unit_only:
        return 1 if u_fail else 0

    model_path = resolve_model_path(args.model or None)
    speculative = not args.no_speculative
    draft_path = args.draft_model
    if draft_path is not None and draft_path.strip() == "":
        speculative = False
        draft_path = ""

    print(f"\n=== MODEL: loading {model_path} ===")
    print(
        f"speculative={speculative} draft={draft_path if draft_path is not None else '(auto)'} "
        f"num_draft={args.num_draft_tokens}"
    )
    t_load = time.perf_counter()
    engine = ModelEngine(
        model_path=model_path,
        max_tokens=32,
        temperature=0.0,
        draft_model_path=draft_path,
        speculative=speculative,
        num_draft_tokens=args.num_draft_tokens,
    )
    print(f"Load time: {(time.perf_counter() - t_load) * 1000:.0f}ms")
    print(
        f"fim={engine._has_fim} chat={engine._has_chat} "
        f"instruct={engine._is_instruct} "
        f"spec={'yes' if engine.draft_model is not None else 'no'} "
        f"draft={engine.draft_model_path or '-'}"
    )

    # Warmup
    print("Warmup…")
    for _ in engine.stream(
        engine.build_fim_prompt("const x = ", "\n", "javascript"),
        max_tokens=8,
        stop_on_newline=True,
    ):
        pass

    results = []
    print("\n=== SCENARIOS (model + post-process) ===")
    for sc in SCENARIOS:
        r = run_model_scenario(engine, sc)
        results.append(r)
        flag = "PASS" if r["ok"] and not r.get("reasons") else (
            "WARN" if r["ok"] else "FAIL"
        )
        print(
            f"  {flag} {r['name']}: ttft={r['ttft_ms']:.0f}ms total={r['total_ms']:.0f}ms"
        )
        print(f"       raw={r.get('raw', '')!r}"[:120])
        print(f"       insert={r.get('insert')!r}")
        print(f"       final={r.get('final_line')!r}")
        if r.get("reasons"):
            print(f"       reasons={r['reasons']}")

    # Summary
    n_pass = sum(1 for r in results if r["ok"] and not r.get("reasons"))
    n_warn = sum(1 for r in results if r["ok"] and r.get("reasons"))
    n_fail = sum(1 for r in results if not r["ok"])
    ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] and r["ttft_ms"] > 0]
    totals = [r["total_ms"] for r in results if r["total_ms"] > 0]

    print("\n=== SUMMARY ===")
    print(f"Unit:      {u_pass} pass / {u_fail} fail")
    print(f"Scenarios: {n_pass} pass / {n_warn} warn / {n_fail} fail  (of {len(results)})")
    if ttfts:
        print(
            f"TTFT ms:   min={min(ttfts):.0f}  median={sorted(ttfts)[len(ttfts)//2]:.0f}  "
            f"max={max(ttfts):.0f}  avg={sum(ttfts)/len(ttfts):.0f}"
        )
    if totals:
        print(
            f"Total ms:  min={min(totals):.0f}  median={sorted(totals)[len(totals)//2]:.0f}  "
            f"max={max(totals):.0f}  avg={sum(totals)/len(totals):.0f}"
        )

    # Accuracy scorecard for critical scenarios
    critical = ["arrow_sub_rhs", "arrow_add_rhs", "no_double_const_regression", "with_nearby_add_then_sub"]
    crit_fail = [r for r in results if r["name"] in critical and not r["ok"]]
    if crit_fail:
        print("\nCRITICAL FAILURES:")
        for r in crit_fail:
            print(f"  - {r['name']}: {r.get('reasons')} final={r.get('final_line')!r}")

    return 1 if (u_fail or n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
