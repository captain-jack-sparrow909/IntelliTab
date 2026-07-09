/**
 * Inline completion provider.
 *
 * Multi-line / factorial-class failures fixed by:
 * - Completing intent generations fully (no mid-stream publish, longer wait)
 * - Not cancelling in-flight intent for the same empty body
 * - Rejecting incomplete / placeholder / unbalanced code
 * - Stripping re-emitted signatures from body inserts
 * - Rejecting junk single-token inserts (`\n`, `;`, `const`)
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
            const res = await waitUntilDone(
                this.activeGen,
                isIntent ? 4500 : 1200,
            );
            return res ? this.toInlineItems(res) : null;
        }

        // Protect in-flight INTENT: do not cancel mid-body generation.
        if (this.activeGen && !this.activeGen.done && this.activeGen.isIntent) {
            if (isIntent && intent && intent === this.activeGen.intentText) {
                const res = await waitUntilDone(this.activeGen, 4500);
                return res ? this.toInlineItems(res) : null;
            }
            // User left the empty-body site — allow cancel.
            if (isIntent) {
                // different intent site
            } else {
                // typing inside incomplete body: don't kill intent if still same line region
                const res = await waitUntilDone(this.activeGen, 4500);
                if (res) {
                    return this.toInlineItems(res);
                }
            }
        }

        if (this.activeGen && this.activeGen.contextKey !== contextKey) {
            // Never cancel an unfinished intent gen for a flapping FIM key.
            if (this.activeGen.isIntent && !this.activeGen.done) {
                const res = await waitUntilDone(this.activeGen, 4500);
                if (res && isIntent) {
                    return this.toInlineItems(res);
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

        const res = await waitUntilDone(gen, isIntent ? 4500 : 1200);
        log(
            `[Provider] mode=${contextData.mode} ` +
                `out=${res ? JSON.stringify(res.text.slice(0, 80)) : "null"} ` +
                `${Date.now() - t0}ms`,
        );
        return res ? this.toInlineItems(res) : null;
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
        const maxTok = isIntent
            ? Math.min(220, Math.max(this.maxTokens, 160))
            : Math.max(20, Math.min(40, this.maxTokens || 32));

        const prepare = (raw: string): PreparedCompletion | null => {
            let cleaned = cleanRaw(raw);
            if (!cleaned) {
                return null;
            }

            if (isIntent) {
                cleaned = refineBodyInsert(cleaned, contextData.before, linePrefix);
                cleaned = finishIncompleteBlock(cleaned);
                cleaned = fixCommonMathBaseCases(cleaned, contextData.before + "\n" + (intentText || ""));
                cleaned = normalizeIndentation(cleaned, document, position);
            }

            const prepared = toInsert(cleaned, linePrefix, document, position, isIntent);
            if (!prepared || !prepared.text) {
                return null;
            }
            if (isLowQuality(prepared.text, contextData, isIntent)) {
                return null;
            }
            if (isIntent && !isStructurallyComplete(prepared.text)) {
                log(
                    `[Provider] reject incomplete structure: ${JSON.stringify(
                        prepared.text.slice(0, 80),
                    )}`,
                );
                return null;
            }
            return prepared;
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
 * If the model re-emits `const factorial = () => { ... }`, keep only the body.
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

/** Drop incomplete tails, flatten useless nesting, fix common factorial base-case bug. */
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

/** Fix well-known wrong base cases when the surrounding name implies them. */
function fixCommonMathBaseCases(text: string, contextHint: string): string {
    let t = text;
    if (!/\bfactorial\b|\bfactorialize\b/i.test(contextHint + "\n" + text)) {
        return t;
    }
    // 0! = 1 — models often emit `if (n === 0) return 0`
    t = t.replace(
        /if\s*\(\s*n\s*===\s*0\s*\)\s*\{[^{}]*return\s+0\s*;[^{}]*\}/g,
        "if (n === 0) {\n    return 1;\n  }",
    );
    t = t.replace(
        /if\s*\(\s*n\s*===\s*0\s*\)\s*return\s+0\s*;/g,
        "if (n === 0) return 1;",
    );
    t = t.replace(
        /if\s*\(\s*n\s*==\s*0\s*\)\s*return\s+0\s*;/g,
        "if (n == 0) return 1;",
    );
    // Prefer unified base: if (n === 0) return 1; else if (n === 1) return 1
    // → if (n <= 1) return 1 when both return 1 (optional tidy — skip if complex)
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
    // Placeholders
    if (
        /your code here/i.test(t) ||
        /\bTODO\b/.test(t) ||
        /\bFIXME\b/.test(t) ||
        t.includes("...")
    ) {
        return true;
    }
    if (/\b([A-Za-z_$][\w$]*)\s+\1\b/.test(t)) {
        return true;
    }
    if (/\b(const|let|var)\b.+\b(const|let|var)\b/.test(t) && !isIntent) {
        return true;
    }
    // Duplicate return spam
    if ((t.match(/\breturn\b/g) || []).length >= 3 && !isIntent) {
        return true;
    }
    if (!isIntent && ctx.after) {
        const afterStart = ctx.after.trimStart().slice(0, t.length);
        if (afterStart && t === afterStart) {
            return true;
        }
    }
    // Echo of a line that already exists just above/below (common with factorial spam)
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
