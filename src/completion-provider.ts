/**
 * Inline completion provider.
 *
 * General-purpose completion (no per-function recipes):
 * - FIM for mid-line; instruct body fill for empty blocks / comments
 * - Finish intent fully; protect in-flight body gens from cancel thrash
 * - Structural filters only (braces, placeholders, cross-file-fn paste)
 * - Strip re-emitted signatures; reject junk tokens
 */

import * as vscode from "vscode";
import { extractContext, detectIntent, DocumentContext } from "./context-extractor";
import { BackendIPC } from "./backend-ipc";

let logFn: ((msg: string) => void) | null = null;

export function setLogger(fn: ((msg: string) => void) | null): void {
    logFn = fn;
}

function log(msg: string): void {
    logFn?.(msg);
}

interface PreparedCompletion {
    text: string;
    range: vscode.Range;
}

interface GenState {
    promise: Promise<PreparedCompletion | null>;
    result: PreparedCompletion | null;
    done: boolean;
    contextKey: string;
    isIntent: boolean;
    intentText?: string;
}

export class CompletionProvider implements vscode.InlineCompletionItemProvider {
    private maxTokens: number;
    private linesBefore: number;
    private linesAfter: number;
    private backend: BackendIPC;

    private completions = new Map<string, PreparedCompletion>();
    private activeGen: GenState | null = null;
    private retriggerTimer: ReturnType<typeof setTimeout> | null = null;

    constructor(
        backend: BackendIPC,
        _debounceMs: number,
        maxTokens: number,
        outputChannel: vscode.OutputChannel | null,
        linesBefore: number = 80,
        linesAfter: number = 20,
    ) {
        this.backend = backend;
        this.maxTokens = maxTokens;
        this.linesBefore = linesBefore;
        this.linesAfter = linesAfter;
        setLogger((msg: string) => outputChannel?.appendLine(msg));
        log("Inline provider ready");
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken,
    ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList | null> {
        if (document.uri.scheme !== "file" && document.uri.scheme !== "untitled") {
            return null;
        }
        if (!this.backend.isRunning() || !this.isCompletionValid(document, position)) {
            return null;
        }

        const t0 = Date.now();
        const intent = detectIntent(document, position);
        const contextData = extractContext(
            document,
            position,
            this.linesBefore,
            this.linesAfter,
            intent,
        );
        const isIntent = contextData.mode === "intent";
        const linePrefix = document
            .lineAt(position.line)
            .text.substring(0, position.character);

        // Intent keys ignore whitespace-only prefix so indent typing doesn't restart.
        const contextKey = isIntent
            ? `i:${intent}:${position.line}`
            : `f:${linePrefix}|${hashShort(contextData.after)}|${position.line}:${position.character}`;

        const cached = this.completions.get(contextKey);
        if (cached) {
            return this.toInlineItems(cached);
        }

        // Join in-flight work for the same key.
        if (this.activeGen && this.activeGen.contextKey === contextKey) {
            const joined = await waitUntilDone(this.activeGen, isIntent ? 3500 : 800);
            return joined ? this.toInlineItems(joined) : null;
        }

        // Protect in-flight INTENT: do not cancel mid-body generation.
        if (this.activeGen && !this.activeGen.done && this.activeGen.isIntent) {
            if (isIntent && intent && intent === this.activeGen.intentText) {
                const joined = await waitUntilDone(this.activeGen, 3500);
                return joined ? this.toInlineItems(joined) : null;
            }
            if (!isIntent) {
                const joined = await waitUntilDone(this.activeGen, 3500);
                if (joined) {
                    return this.toInlineItems(joined);
                }
            }
        }

        if (this.activeGen && this.activeGen.contextKey !== contextKey) {
            if (this.activeGen.isIntent && !this.activeGen.done) {
                const joined = await waitUntilDone(this.activeGen, 3500);
                if (joined && isIntent) {
                    return this.toInlineItems(joined);
                }
            } else {
                this.backend.cancelActive();
                this.activeGen = null;
            }
        }

        const gen = this.startGeneration(
            contextData,
            isIntent,
            document,
            position,
            linePrefix,
            contextKey,
            intent,
        );
        this.activeGen = gen;

        const result = await waitUntilDone(gen, isIntent ? 3500 : 800);
        log(
            `[Provider] mode=${contextData.mode} ` +
                `out=${result ? JSON.stringify(result.text.slice(0, 80)) : "null"} ` +
                `${Date.now() - t0}ms`,
        );
        return result ? this.toInlineItems(result) : null;
    }

    private startGeneration(
        contextData: DocumentContext,
        isIntent: boolean,
        document: vscode.TextDocument,
        position: vscode.Position,
        linePrefix: string,
        contextKey: string,
        intentText?: string,
    ): GenState {
        const state: GenState = {
            promise: Promise.resolve(null),
            result: null,
            done: false,
            contextKey,
            isIntent,
            intentText,
        };

        const t0 = Date.now();
        let firstTokenAt = 0;
        let accumulated = "";

        const stopOnNewline = !isIntent;
        // FIM: short decode. Intent: enough for a full body; server early-stops when balanced.
        const maxTok = isIntent
            ? Math.min(160, Math.max(this.maxTokens, 96))
            : Math.max(16, Math.min(28, this.maxTokens || 28));

        const prepare = (raw: string): PreparedCompletion | null => {
            let cleaned = cleanRaw(raw);
            if (!cleaned) {
                return null;
            }

            if (isIntent) {
                cleaned = refineBodyInsert(cleaned, contextData.before, linePrefix);
                cleaned = finishIncompleteBlock(cleaned);
                cleaned = fixUnreachableElseAfterReturn(cleaned);
            }
            // Cheap whitespace-only formatting (no model cost): break packed statements.
            cleaned = formatStatementNewlines(cleaned);
            if (isIntent || cleaned.includes("\n")) {
                cleaned = normalizeIndentation(cleaned, document, position);
            }

            const prepared = toInsert(cleaned, linePrefix, document, position, isIntent);
            if (!prepared || !prepared.text) {
                return null;
            }
            // Format again after insert shaping (range/body only) if still packed.
            let finalText = formatStatementNewlines(prepared.text);
            if (finalText.includes("\n") || isIntent) {
                finalText = normalizeIndentation(finalText, document, position);
            }
            const formatted: PreparedCompletion = { text: finalText, range: prepared.range };

            if (isLowQuality(formatted.text, contextData, isIntent)) {
                return null;
            }
            if (isIntent && !isStructurallyComplete(formatted.text)) {
                log(
                    `[Provider] reject incomplete structure: ${JSON.stringify(
                        formatted.text.slice(0, 80),
                    )}`,
                );
                return null;
            }
            return formatted;
        };

        const onToken = (tokenText: string) => {
            if (!tokenText) {
                return;
            }
            if (!firstTokenAt) {
                firstTokenAt = Date.now();
                log(`[Provider] TTFT ${firstTokenAt - t0}ms mode=${contextData.mode}`);
            }
            accumulated += tokenText;
        };

        state.promise = this.backend
            .complete(
                {
                    before: contextData.before,
                    after: contextData.after,
                    language: contextData.language,
                    intent: contextData.intent,
                    mode: contextData.mode,
                },
                onToken,
                {
                    maxTokens: maxTok,
                    stopOnNewline,
                    // Intent: never cancel-previous at start of same-key join
                    cancelPrevious: false,
                },
            )
            .then(() => {
                state.done = true;
                const prepared = prepare(accumulated);
                if (prepared) {
                    state.result = prepared;
                    this.completions.set(contextKey, prepared);
                    if (this.completions.size > 40) {
                        const first = this.completions.keys().next().value;
                        if (first !== undefined) {
                            this.completions.delete(first);
                        }
                    }
                    log(
                        `[Provider] done ${Date.now() - t0}ms ttft=${
                            firstTokenAt ? firstTokenAt - t0 : -1
                        }ms insert=${JSON.stringify(prepared.text.slice(0, 100))} ` +
                            `rangeCol=${prepared.range.start.character}`,
                    );
                    this.scheduleRetrigger();
                    return prepared;
                }
                log(
                    `[Provider] rejected raw=${JSON.stringify(accumulated.slice(0, 120))} ` +
                        `in ${Date.now() - t0}ms`,
                );
                state.result = null;
                return null;
            })
            .catch((err: Error) => {
                state.done = true;
                log(`[Provider] error: ${err.message}`);
                return null;
            });

        return state;
    }

    private scheduleRetrigger(): void {
        if (this.retriggerTimer !== null) {
            return;
        }
        this.retriggerTimer = setTimeout(() => {
            this.retriggerTimer = null;
            void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
        }, 16);
    }

    private toInlineItems(prep: PreparedCompletion): vscode.InlineCompletionList {
        return {
            items: [new vscode.InlineCompletionItem(prep.text, prep.range)],
        };
    }

    private isCompletionValid(document: vscode.TextDocument, position: vscode.Position): boolean {
        const beforeCursor = document
            .lineAt(position.line)
            .text.substring(0, position.character);
        if (beforeCursor.includes("/*") && !beforeCursor.includes("*/")) {
            return false;
        }
        return true;
    }

    dispose(): void {
        if (this.retriggerTimer !== null) {
            clearTimeout(this.retriggerTimer);
        }
        this.completions.clear();
        this.activeGen = null;
        this.backend.cancelActive();
    }
}

// --- helpers -----------------------------------------------------------------

function hashShort(s: string): string {
    let h = 0;
    const n = Math.min(s.length, 120);
    for (let i = 0; i < n; i++) {
        h = (h * 31 + s.charCodeAt(i)) | 0;
    }
    return `${s.length}:${h}`;
}

function waitUntilDone(
    state: GenState,
    timeoutMs: number,
): Promise<PreparedCompletion | null> {
    if (state.done) {
        return Promise.resolve(state.result);
    }
    return new Promise((resolve) => {
        const start = Date.now();
        const interval = setInterval(() => {
            if (state.done) {
                clearInterval(interval);
                resolve(state.result);
                return;
            }
            if (Date.now() - start >= timeoutMs) {
                clearInterval(interval);
                // Prefer waiting a bit more for intent rather than returning null mid-way
                resolve(state.result);
            }
        }, 16);
    });
}

/**
 * If the model re-emits a full function wrapper, keep only the body.
 */
function refineBodyInsert(text: string, before: string, lineBefore: string): string {
    let t = text.trimStart();

    // Strip a full function/const wrapper when signature already exists above.
    const wrapper = t.match(
        /^(?:export\s+)?(?:async\s+)?(?:function\s+[\w$]+\s*\([^)]*\)\s*\{|const\s+[\w$]+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{)\s*\n?([\s\S]*)$/,
    );
    if (wrapper) {
        let body = wrapper[1];
        // Drop trailing closing braces that belong to the wrapper
        body = body.replace(/\n?\}\s*;?\s*$/, "");
        // Only use body strip if surrounding already has a similar signature
        if (
            /function\s+[\w$]+|(?:const|let|var)\s+[\w$]+\s*=/.test(before) ||
            lineBefore.trim() === ""
        ) {
            t = body;
        }
    }

    // Remove leading duplicate of current indent-only line
    if (!lineBefore.trim()) {
        t = t.replace(/^\s*\n/, "");
    }

    return t;
}

/**
 * Remove illegal / unreachable `else` after `return` (common model glitch):
 *   if (a === b) { return true; else { return false; } }
 * → if (a === b) { return true; }
 * Structural only — no algorithm-specific rewrites.
 */
function fixUnreachableElseAfterReturn(text: string): string {
    let t = text;
    // return <expr>; else { ... }  (brace-balanced simple blocks)
    for (let i = 0; i < 6; i++) {
        const next = t.replace(
            /return\s+([^;]+);\s*else\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/g,
            "return $1;",
        );
        if (next === t) {
            break;
        }
        t = next;
    }
    // return <expr>; else return <expr>;
    t = t.replace(/return\s+([^;]+);\s*else\s+return\s+[^;]+;/g, "return $1;");
    return t;
}

/**
 * Insert newlines between *same-line packed* statements only.
 * e.g. `const s = new Set();return arr.filter(...)` → two lines.
 * Does not re-break already-formatted multi-line code.
 * Safe for `for (;;)` (semis followed by exprs, not statement keywords).
 * Pure string ops — negligible latency; semantics unchanged.
 */
function formatStatementNewlines(text: string): string {
    if (!text || !text.trim()) {
        return text;
    }
    let t = text;

    // Horizontal whitespace only — so we only fix "packed on one line".
    const hs = "[^\\S\\n]*"; // spaces/tabs, not newlines
    const stmt =
        "(?:return|const|let|var|if|for|while|switch|try|throw|function|class|" +
        "export|import|async|await|break|continue|debugger|yield|do|case|default)";

    // ";return" / ";const" on the same line
    t = t.replace(new RegExp(`;(?=${hs}${stmt}\\b)`, "g"), ";\n");

    // "{return" / "{const" packed after brace on same line
    t = t.replace(new RegExp(`\\{(?=${hs}${stmt}\\b)`, "g"), "{\n");

    // "}return" / "}const" packed after closing brace on same line
    t = t.replace(new RegExp(`\\}(?=${hs}${stmt}\\b)`, "g"), "}\n");

    t = t.replace(/\n{3,}/g, "\n\n");
    t = t.replace(/[ \t]+\n/g, "\n");
    return t;
}

/** Drop incomplete tails and flatten useless nested blocks. */
function finishIncompleteBlock(text: string): string {
    let t = text.replace(/\s+$/, "");
    const lines = t.split("\n");
    while (lines.length > 0) {
        const last = lines[lines.length - 1].trim();
        if (!last) {
            lines.pop();
            continue;
        }
        if (
            /^(for|while|if|else if|switch|catch|function|const|let|var)\b.*[({,]\s*$/.test(
                last,
            ) ||
            /^(for|while|if)\s*\([^)]*$/.test(last) ||
            last.endsWith("&&") ||
            last.endsWith("||") ||
            last.endsWith("?") ||
            last.endsWith(":") ||
            last.endsWith(",")
        ) {
            lines.pop();
            continue;
        }
        break;
    }
    t = lines.join("\n");

    // Collapse useless nested blocks: { { stmt } } → { stmt }
    for (let i = 0; i < 4; i++) {
        const next = t.replace(
            /\{\s*\{\s*([^/*{}][^{}]*?)\s*\}\s*\}/g,
            "{ $1 }",
        );
        if (next === t) {
            break;
        }
        t = next;
    }

    // Balance braces carefully — only add closers if clearly truncated mid-block
    const bal = braceBalance(t);
    if (bal > 0 && bal <= 2 && /\{\s*$/.test(t.trim()) === false) {
        // Prefer not inventing structure; only close if last non-ws opened a block
        const trimmed = t.trimEnd();
        if (/[;{}]\s*$/.test(trimmed) || /return\b[\s\S]*$/.test(trimmed)) {
            t = t + "\n" + "}".repeat(bal);
        }
    } else if (bal < 0) {
        let b = bal;
        while (b < 0 && /\}\s*;?\s*$/.test(t)) {
            t = t.replace(/\}\s*;?\s*$/, "");
            b++;
        }
    }

    return t;
}

function braceBalance(s: string): number {
    let n = 0;
    for (const ch of s) {
        if (ch === "{") {
            n++;
        } else if (ch === "}") {
            n--;
        }
    }
    return n;
}

function isStructurallyComplete(text: string): boolean {
    const t = text.trim();
    if (!t) {
        return false;
    }
    // Must not end mid-token keyword
    if (/\b(for|if|while|function|const|let|var|return)\s*$/.test(t)) {
        return false;
    }
    if (braceBalance(t) !== 0) {
        return false;
    }
    // Parens roughly balanced
    let p = 0;
    for (const ch of t) {
        if (ch === "(") {
            p++;
        } else if (ch === ")") {
            p--;
        }
        if (p < 0) {
            return false;
        }
    }
    if (p !== 0) {
        return false;
    }
    return true;
}

function normalizeIndentation(
    text: string,
    document: vscode.TextDocument,
    position: vscode.Position,
): string {
    const lineText = document.lineAt(position.line).text;
    const beforeCursor = lineText.substring(0, position.character);
    const lineIndent = (lineText.match(/^\s*/) || [""])[0];
    const atIndentOnly = beforeCursor.trim().length === 0;
    const baseIndent = atIndentOnly ? beforeCursor : lineIndent;

    let lines = text.split("\n");
    const nonEmpty = lines.filter((l) => l.trim().length > 0);
    if (nonEmpty.length === 0) {
        return "";
    }
    const minIndent = Math.min(
        ...nonEmpty.map((l) => (l.match(/^\s*/) || [""])[0].length),
    );
    lines = lines.map((l) => {
        if (l.trim().length === 0) {
            return "";
        }
        return l.length >= minIndent ? l.slice(minIndent) : l.trimStart();
    });

    if (lines.length === 1) {
        return atIndentOnly ? lines[0].replace(/^\s+/, "") : lines[0].replace(/^\s+/, "");
    }

    return lines
        .map((ln, i) => {
            if (i === 0) {
                return ln.replace(/^\s+/, "");
            }
            if (ln.length === 0) {
                return "";
            }
            return baseIndent + ln;
        })
        .join("\n");
}

function toInsert(
    raw: string,
    lineBefore: string,
    document: vscode.TextDocument,
    position: vscode.Position,
    isIntent: boolean,
): PreparedCompletion | null {
    let text = raw;

    if (!isIntent) {
        const nl = text.search(/\r?\n/);
        if (nl !== -1) {
            text = text.slice(0, nl);
        }
    }

    text = text.replace(/[ \t]+$/g, "");
    if (!text || text === "\n" || text === ";" || text === "\\n") {
        return null;
    }

    // Intent multi-line: pure insert at cursor (indent already normalized)
    if (isIntent) {
        // Never full-line rewrite intent bodies to col 0 with a new const
        if (
            lineBefore.trim().length === 0 &&
            /^(const|let|var|function)\b/.test(text.trim())
        ) {
            // Signature already above empty body — strip wrapper again
            text = refineBodyInsert(text, document.getText(), lineBefore);
        }
        text = text.replace(/^\n+/, "");
        if (!text.trim()) {
            return null;
        }
        return {
            text,
            range: new vscode.Range(position, position),
        };
    }

    // --- FIM single-line path ---
    if (lineBefore && text.startsWith(lineBefore)) {
        text = text.slice(lineBefore.length);
        return zero(text, position);
    }

    let lcp = 0;
    while (
        lcp < lineBefore.length &&
        lcp < text.length &&
        lineBefore.charAt(lcp) === text.charAt(lcp)
    ) {
        lcp++;
    }
    if (lcp === lineBefore.length && lineBefore) {
        return zero(text.slice(lcp), position);
    }

    if (
        lcp >= 4 &&
        /^(const|let|var|function|async|export|def|class)\b/.test(text.trim())
    ) {
        // Full-line rewrite only if it keeps a similar declaration shape
        return {
            text: text.replace(/^\s+/, ""),
            range: new vscode.Range(new vscode.Position(position.line, 0), position),
        };
    }

    if (lineBefore) {
        const maxK = Math.min(lineBefore.length, text.length);
        for (let k = maxK; k >= 1; k--) {
            if (lineBefore.endsWith(text.slice(0, k))) {
                text = text.slice(k);
                break;
            }
        }
    }

    if (
        lineBefore.trim().length > 0 &&
        /^(const|let|var|function|async|export)\b/.test(text.trim())
    ) {
        return {
            text: text.replace(/^\s+/, ""),
            range: new vscode.Range(new vscode.Position(position.line, 0), position),
        };
    }

    if (lineBefore.trim().length > 0) {
        const semi = text.indexOf(";");
        if (semi !== -1) {
            text = text.slice(0, semi + 1);
        }
    }

    text = text.replace(/^\s+/, "");
    if (!text) {
        return null;
    }

    if (
        /\b(const|let|var)\b/.test(lineBefore) &&
        /^(const|let|var)\b/.test(text.trim())
    ) {
        const modelLine = raw.split(/\r?\n/)[0].trim();
        if (modelLine && !modelLine.startsWith("```")) {
            return {
                text: modelLine,
                range: new vscode.Range(new vscode.Position(position.line, 0), position),
            };
        }
        return null;
    }

    if (/(?:const|let|var)\s+[\w$]+$/.test(lineBefore) && /^\(/.test(text)) {
        text = " = " + text;
    }

    // Reject trivial noise
    if (text === "const" || text === "let" || text === "var" || text === "function") {
        return null;
    }

    return zero(text, position);
}

function zero(text: string, position: vscode.Position): PreparedCompletion | null {
    if (!text || !text.trim()) {
        return null;
    }
    return { text, range: new vscode.Range(position, position) };
}

function cleanRaw(raw: string): string {
    let text = raw ?? "";
    text = text.replace(/<\|fim_(?:begin|end|pad|prefix|suffix|middle)\|>/g, "");
    text = text.replace(/<\|endoftext\|>/g, "");
    text = text.replace(/<\|im_(?:end|start)\|>/g, "");
    text = text.replace(/```[a-zA-Z0-9_+-]*\s*/g, "");
    text = text.replace(/```/g, "");
    text = text.replace(/^[\r\n]+/, "");
    text = text.replace(/^(javascript|typescript|python|java|go|rust|tsx|jsx)\s*\n/i, "");
    return text;
}

function isLowQuality(text: string, ctx: DocumentContext, isIntent: boolean): boolean {
    const t = text.trim();
    if (!t || t.length < 1) {
        return true;
    }
    // Single punctuation / whitespace
    if (/^[\s;{}()[\],.]*$/.test(t)) {
        return true;
    }
    if (t === "const" || t === "let" || t === "var" || t === "function") {
        return true;
    }
    if (t.startsWith("```") || /^here('s| is)\b/i.test(t)) {
        return true;
    }
    // Generic placeholders / stubs (not spread syntax `...x`)
    if (
        /your code here/i.test(t) ||
        /your prediction logic/i.test(t) ||
        /placeholder for/i.test(t) ||
        /\bTODO\b/.test(t) ||
        /\bFIXME\b/.test(t) ||
        /(^|\n)\s*\.\.\.\s*($|\n)/.test(t) ||
        /\/\/\s*\.\.\./.test(t) ||
        /#\s*For example:/i.test(t)
    ) {
        return true;
    }
    // try without except/finally (incomplete control flow)
    if (/\btry\s*:/.test(t) && !/\bexcept\b/.test(t) && !/\bfinally\b/.test(t)) {
        return true;
    }
    if (/\btry\s*\{/.test(t) && !/\bcatch\b/.test(t) && !/\bfinally\b/.test(t)) {
        return true;
    }
    // Illegal: return ...; else
    if (/return\s+[^;]+;\s*else\b/.test(t)) {
        return true;
    }
    // Trailing unfinished def/async def / function header with no body
    if (/(?:async\s+)?def\s+\w+\s*\([^)]*\)\s*:\s*$/m.test(t)) {
        return true;
    }
    if (/(?:async\s+)?function\s*\w*\s*\([^)]*\)\s*\{\s*$/m.test(t)) {
        return true;
    }
    // Likely pasted another local function from the same file into this body
    if (isIntent && isCrossFunctionContamination(t, ctx.before || "")) {
        return true;
    }
    if (/\b([A-Za-z_$][\w$]*)\s+\1\b/.test(t)) {
        return true;
    }
    if (/\b(const|let|var)\b.+\b(const|let|var)\b/.test(t) && !isIntent) {
        return true;
    }
    // Duplicate return spam on single-line FIM
    if ((t.match(/\breturn\b/g) || []).length >= 3 && !isIntent) {
        return true;
    }
    if (!isIntent && ctx.after) {
        const afterStart = ctx.after.trimStart().slice(0, t.length);
        if (afterStart && t === afterStart) {
            return true;
        }
    }
    // Echo of a nearby existing line (repeat loop)
    if (!isIntent && t.length > 8) {
        const after = ctx.after || "";
        const before = ctx.before || "";
        if (after.includes(t) || before.split("\n").slice(-6).some((l) => l.trim() === t.trim())) {
            return true;
        }
    }
    const lang = (ctx.language || "").toLowerCase();
    if (
        (lang === "javascript" ||
            lang === "typescript" ||
            lang === "javascriptreact" ||
            lang === "typescriptreact") &&
        /^\s*def\s+\w+/.test(t)
    ) {
        return true;
    }
    return false;
}

/**
 * Heuristic: body calls another user-defined function that appears earlier in
 * *this file* but is not the function being written (likely copy-paste from context).
 * Allows recursive self-calls. Does not hardcode algorithm names.
 */
function isCrossFunctionContamination(body: string, before: string): boolean {
    const names = [...before.matchAll(/(?:function|const|let|var|def)\s+([\w$]+)/gi)].map(
        (m) => m[1],
    );
    if (names.length < 2) {
        return false;
    }
    const current = names[names.length - 1];
    const others = new Set(
        names.slice(0, -1).filter((n) => n.toLowerCase() !== current.toLowerCase()),
    );
    if (others.size === 0) {
        return false;
    }
    const builtins = new Set([
        "if", "for", "while", "switch", "catch", "function", "return",
        "parseInt", "parseFloat", "Number", "String", "Boolean", "Array",
        "Object", "Math", "JSON", "console", "setTimeout", "setInterval",
        "Promise", "Error", "Map", "Set", "Date", "RegExp", "Symbol",
        "isNaN", "isFinite", "encodeURIComponent", "decodeURIComponent",
        "require", "fetch", "Buffer", "process",
    ]);
    const callRe = /\b([A-Za-z_$][\w$]*)\s*\(/g;
    let m: RegExpExecArray | null;
    while ((m = callRe.exec(body)) !== null) {
        const callee = m[1];
        if (builtins.has(callee)) {
            continue;
        }
        if (others.has(callee) && callee.toLowerCase() !== current.toLowerCase()) {
            return true;
        }
    }
    return false;
}
