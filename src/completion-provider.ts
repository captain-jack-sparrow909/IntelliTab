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
        const multiLine = !!contextData.multiLine;
        const linePrefix = document
            .lineAt(position.line)
            .text.substring(0, position.character);

        // Intent keys ignore whitespace-only prefix so indent typing doesn't restart.
        // Multi-line FIM: key on line+depth so typing inside the same block can join.
        const contextKey = isIntent
            ? `i:${intent}:${position.line}`
            : multiLine
              ? `m:${position.line}:${linePrefix.trimEnd()}|${hashShort(contextData.before.slice(-200))}`
              : `f:${linePrefix}|${hashShort(contextData.after)}|${position.line}:${position.character}`;

        const cached = this.completions.get(contextKey);
        if (cached) {
            return this.toInlineItems(cached);
        }

        // Join in-flight work for the same key.
        const waitMs = isIntent || multiLine ? 3500 : 800;
        if (this.activeGen && this.activeGen.contextKey === contextKey) {
            const joined = await waitUntilDone(this.activeGen, waitMs);
            return joined ? this.toInlineItems(joined) : null;
        }

        // Protect in-flight INTENT / multi-line body: do not cancel mid-generation.
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
            } else if (multiLine && !this.activeGen.done) {
                // Don't thrash multi-line FIM on tiny cursor jitter.
                const joined = await waitUntilDone(this.activeGen, 2000);
                if (joined && this.activeGen.contextKey === contextKey) {
                    return this.toInlineItems(joined);
                }
                this.backend.cancelActive();
                this.activeGen = null;
            } else {
                this.backend.cancelActive();
                this.activeGen = null;
            }
        }

        const gen = this.startGeneration(
            contextData,
            isIntent,
            multiLine,
            document,
            position,
            linePrefix,
            contextKey,
            intent,
        );
        this.activeGen = gen;

        const result = await waitUntilDone(gen, waitMs);
        log(
            `[Provider] mode=${contextData.mode}${multiLine ? "+ml" : ""} ` +
                `out=${result ? JSON.stringify(result.text.slice(0, 80)) : "null"} ` +
                `${Date.now() - t0}ms`,
        );
        return result ? this.toInlineItems(result) : null;
    }

    private startGeneration(
        contextData: DocumentContext,
        isIntent: boolean,
        multiLine: boolean,
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

        // Intent / block-continue: multi-line. Mid-expression FIM: one line.
        const stopOnNewline = !isIntent && !multiLine;
        const maxTok = isIntent
            ? Math.min(160, Math.max(this.maxTokens, 96))
            : multiLine
              ? Math.min(96, Math.max(this.maxTokens, 48))
              : Math.max(16, Math.min(28, this.maxTokens || 28));

        const prepare = (raw: string): PreparedCompletion | null => {
            let cleaned = cleanRaw(raw);
            if (!cleaned) {
                return null;
            }

            if (isIntent || multiLine) {
                cleaned = refineBodyInsert(cleaned, contextData.before, linePrefix);
                // Salvage first: cut prose / brace spam / invented next fns
                // before formatting so we don't polish garbage.
                cleaned = salvageCodeCompletion(
                    cleaned,
                    contextData.after || "",
                    multiLine,
                );
                cleaned = finishIncompleteBlock(cleaned);
                cleaned = fixUnreachableElseAfterReturn(cleaned);
            }
            // Cheap whitespace-only formatting (no model cost): break packed statements.
            cleaned = formatStatementNewlines(cleaned);
            if (isIntent || multiLine || cleaned.includes("\n")) {
                cleaned = normalizeIndentation(cleaned, document, position);
            }

            if (!cleaned.trim() || isStructurallyBroken(cleaned)) {
                log(
                    `[Provider] reject broken structure: ${JSON.stringify(
                        cleaned.slice(0, 100),
                    )}`,
                );
                return null;
            }

            const prepared = toInsert(
                cleaned,
                linePrefix,
                document,
                position,
                isIntent,
                multiLine,
            );
            if (!prepared || !prepared.text) {
                return null;
            }
            // Format again after insert shaping (range/body only) if still packed.
            let finalText = formatStatementNewlines(prepared.text);
            if (finalText.includes("\n") || isIntent || multiLine) {
                finalText = normalizeIndentation(finalText, document, position);
            }
            // Final salvage pass after indent normalize
            if (isIntent || multiLine) {
                finalText = salvageCodeCompletion(
                    finalText,
                    contextData.after || "",
                    multiLine,
                );
            }
            const formatted: PreparedCompletion = { text: finalText, range: prepared.range };

            if (!formatted.text.trim() || isStructurallyBroken(formatted.text)) {
                return null;
            }
            if (isLowQuality(formatted.text, contextData, isIntent || multiLine)) {
                return null;
            }
            // Prefer a soft-closed or useful partial over showing nothing.
            // (Strict completeness was rejecting salvage-cut bodies → empty UI.)
            if ((isIntent || multiLine) && !isStructurallyComplete(formatted.text)) {
                const soft = softCloseOpenBlocks(formatted.text);
                if (
                    soft !== formatted.text &&
                    isStructurallyComplete(soft) &&
                    !isStructurallyBroken(soft) &&
                    !isLowQuality(soft, contextData, true)
                ) {
                    log(
                        `[Provider] soft-closed partial: ${JSON.stringify(soft.slice(0, 80))}`,
                    );
                    return { text: soft, range: formatted.range };
                }
                if (isUsefulPartial(formatted.text)) {
                    log(
                        `[Provider] accept useful partial: ${JSON.stringify(
                            formatted.text.slice(0, 80),
                        )}`,
                    );
                    return formatted;
                }
                // Intent comment→code: still require completeness if partial is weak.
                if (isIntent && !multiLine) {
                    log(
                        `[Provider] reject incomplete structure: ${JSON.stringify(
                            formatted.text.slice(0, 80),
                        )}`,
                    );
                    return null;
                }
                // Multi-line FIM: show partial only if not broken (already checked).
                if (formatted.text.trim().length >= 12) {
                    return formatted;
                }
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
                log(
                    `[Provider] TTFT ${firstTokenAt - t0}ms mode=${contextData.mode}${
                        multiLine ? "+ml" : ""
                    }`,
                );
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
                    multiLine: multiLine || undefined,
                },
                onToken,
                {
                    maxTokens: maxTok,
                    stopOnNewline,
                    // Intent / multi-line: never cancel-previous at start of same-key join
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
 * Cut / salvage multi-line completions that derail into prose, brace spam,
 * invented next functions, or runaway indentation.
 */
function salvageCodeCompletion(
    text: string,
    after: string,
    multiLine: boolean,
): string {
    let t = text;
    if (!t) {
        return t;
    }

    t = cutAtProse(t);
    t = cutDuplicateStatementLines(t);
    t = cutRunawayIndent(t);
    t = cutMisindentedClosers(t);
    t = cutBraceCloseSpam(t);

    if (multiLine || after) {
        t = trimMultiLineContinue(t, after);
    }

    // Drop trailing blank lines / half-open tails.
    t = t.replace(/[ \t]+$/gm, "").replace(/\n{3,}/g, "\n\n").replace(/\s+$/, "");
    return t;
}

/** True when a line is natural-language explanation rather than code. */
function isProseLine(line: string): boolean {
    const t = line.trim();
    if (!t || t.length < 12) {
        return false;
    }
    if (
        /^(const|let|var|function|async|await|return|if|else|for|while|switch|case|break|continue|try|catch|finally|throw|class|export|import|from|def|pass|yield|with|except|raise|new|this\.|super\.|public|private|protected|static|interface|type|enum|package|using|fn|func|impl|struct|match|loop|mut|pub)\b/.test(
            t,
        )
    ) {
        return false;
    }
    if ("{}()[];.,/*#`".includes(t[0])) {
        return false;
    }
    if (/[{};=<>]|=>|::|\(\)/.test(t)) {
        if (!/^(this|the|here|note|above|below|we|you|it|in)\b/i.test(t)) {
            return false;
        }
    }
    if (
        /^(this|the|here|note|example|above|below|we |you |it |in this|the following|as you can|explanation|description)\b/i.test(
            t,
        )
    ) {
        return true;
    }
    if (
        /\b(defines?|implements?|takes? a|parameter|method that|function that|class that|as input|as output|is called when)\b/i.test(
            t,
        ) &&
        !/[{};=]/.test(t)
    ) {
        return true;
    }
    const words = t.split(/\s+/);
    if (t.length > 55 && words.length >= 8 && !/[{};=<>()]/.test(t)) {
        return true;
    }
    return false;
}

function cutAtProse(text: string): string {
    const lines = text.split("\n");
    const kept: string[] = [];
    for (const line of lines) {
        if (isProseLine(line)) {
            break;
        }
        // Markdown fence / chat gloss mid-body
        if (/^\s*```/.test(line) && kept.length > 0) {
            break;
        }
        kept.push(line);
    }
    return kept.join("\n");
}

/** Drop consecutive duplicate statement lines (model loops). */
function cutDuplicateStatementLines(text: string): string {
    const lines = text.split("\n");
    const out: string[] = [];
    let prevCode: string | null = null;
    for (const line of lines) {
        const s = line.trim();
        if (
            s &&
            prevCode !== null &&
            s === prevCode &&
            /[A-Za-z]/.test(s) &&
            s.length > 8
        ) {
            // Skip this duplicate; if spam continues, later cuts handle it.
            continue;
        }
        out.push(line);
        if (s) {
            prevCode = s;
        }
    }
    return out.join("\n");
}

/** Stop when indentation explodes (model nesting collapse). */
function cutRunawayIndent(text: string): string {
    const lines = text.split("\n");
    let base: number | null = null;
    const out: string[] = [];
    for (const line of lines) {
        if (!line.trim()) {
            out.push(line);
            continue;
        }
        const ind = (line.match(/^[\t ]*/) || [""])[0].length;
        if (base === null) {
            base = ind;
        }
        // Tight cap: real code rarely jumps >4 levels past the first line.
        if (ind > base + 16 || ind > 32) {
            break;
        }
        out.push(line);
    }
    return out.join("\n");
}

/** Pure `}` / `};` line (not `});` or `} else`). */
function isPureBraceClose(line: string): boolean {
    return /^\s*\}[;,]?\s*$/.test(line);
}

/**
 * Closing braces should not jump to a deeper indent than the previous
 * code line — that's the classic "off the rails" nesting collapse.
 */
function cutMisindentedClosers(text: string): string {
    const lines = text.split("\n");
    const out: string[] = [];
    let prevCodeInd = 0;
    for (const line of lines) {
        if (!line.trim()) {
            out.push(line);
            continue;
        }
        const ind = (line.match(/^[\t ]*/) || [""])[0].length;
        if (isPureBraceClose(line) && out.length > 0 && ind > prevCodeInd) {
            break;
        }
        out.push(line);
        prevCodeInd = ind;
    }
    return out.join("\n");
}

/** Cut cascading pure-`}` lines once we've already closed the useful part. */
function cutBraceCloseSpam(text: string): string {
    const lines = text.split("\n");
    const out: string[] = [];
    let closeRun = 0;
    for (const line of lines) {
        // Only pure `}` / `};` — do NOT treat `});` or `} else` as spam.
        if (isPureBraceClose(line)) {
            closeRun++;
            if (closeRun >= 3) {
                // Keep at most 2 pure closers in a row from the model.
                break;
            }
            out.push(line);
            continue;
        }
        if (line.trim()) {
            closeRun = 0;
        }
        out.push(line);
    }
    return out.join("\n");
}

/**
 * Multi-line FIM cleanup: models often (1) re-emit the FIM suffix that is
 * already in the file, then (2) invent a following function. Keep only the
 * real insert at the cursor.
 */
function trimMultiLineContinue(text: string, after: string): string {
    let t = text;
    if (!t) {
        return t;
    }

    // 1. If the FIM `after` appears inside the generation, cut from there.
    const afterTrim = (after || "").replace(/^\r?\n/, "");
    if (afterTrim.trim().length >= 6) {
        const candidates = [
            afterTrim,
            afterTrim.trimStart(),
            afterTrim
                .split(/\r?\n/)
                .filter((l) => l.trim())
                .slice(0, 3)
                .join("\n"),
        ];
        for (const needle of candidates) {
            if (!needle || needle.length < 6) {
                continue;
            }
            const idx = t.indexOf(needle);
            if (idx >= 8) {
                t = t.slice(0, idx);
                break;
            }
        }
    }

    // 2. Cut before an invented following top-level declaration / class.
    const decl = t.search(
        /\n(?:export\s+)?(?:async\s+)?(?:function\s+\w|class\s+\w|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|def\s+\w)/,
    );
    if (decl > 20) {
        const head = t.slice(0, decl);
        if (braceBalance(head) <= 0) {
            t = head;
        }
    }

    t = t.replace(/[ \t]+$/gm, "").replace(/\n{3,}/g, "\n\n").replace(/\s+$/, "");
    return t;
}

/**
 * True when completion is structurally unusable (prefer show nothing over junk).
 */
function isStructurallyBroken(text: string): boolean {
    const t = text.trim();
    if (!t) {
        return true;
    }
    if (isProseLine(t.split("\n")[0] || "")) {
        return true;
    }
    // Any remaining prose line
    if (t.split("\n").some((ln) => isProseLine(ln))) {
        return true;
    }
    // Wild brace imbalance in the insert alone
    const bal = braceBalance(t);
    if (bal < -2 || bal > 5) {
        return true;
    }
    // Mostly pure closing braces
    const lines = t.split("\n").filter((l) => l.trim());
    if (lines.length >= 3) {
        const closeOnly = lines.filter((l) => isPureBraceClose(l)).length;
        if (closeOnly / lines.length >= 0.4) {
            return true;
        }
    }
    // Runaway indent still present
    const indents = lines.map((l) => (l.match(/^[\t ]*/) || [""])[0].length);
    if (indents.length && Math.max(...indents) > 32) {
        return true;
    }
    // Pure closer deeper than previous line (nesting collapse).
    // Do NOT flag `} else if` or `});` — those continue with code punctuation.
    for (let i = 1; i < lines.length; i++) {
        if (!isPureBraceClose(lines[i])) {
            continue;
        }
        const ind = (lines[i].match(/^[\t ]*/) || [""])[0].length;
        const prev = (lines[i - 1].match(/^[\t ]*/) || [""])[0].length;
        if (ind > prev) {
            return true;
        }
    }
    return false;
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

/**
 * Close a few trailing open braces/parens if the model was cut mid-body
 * (derail-stop / max tokens). Conservative — only small positive balances.
 */
function softCloseOpenBlocks(text: string): string {
    let t = text.replace(/\s+$/, "");
    if (!t) {
        return t;
    }
    // Don't close if clearly mid-expression
    if (/[=+\-*/%,.(]\s*$/.test(t) || /\b(const|let|var|return|if|for|while)\s*$/.test(t)) {
        return t;
    }
    let braces = braceBalance(t);
    let parens = 0;
    for (const ch of t) {
        if (ch === "(") {
            parens++;
        } else if (ch === ")") {
            parens--;
        }
    }
    if (braces < 0 || braces > 3 || parens < 0 || parens > 3) {
        return t;
    }
    if (braces === 0 && parens === 0) {
        return t;
    }
    // Prefer closing after a finished statement
    if (!/[;{})\]]\s*$/.test(t) && !/\breturn\b[^;]*$/.test(t)) {
        return t;
    }
    const indent = (t.match(/\n([ \t]*)\S[^\n]*$/) || ["", "  "])[1];
    const step = indent.startsWith("\t") ? "\t" : "  ";
    while (parens > 0) {
        t += ")";
        parens--;
    }
    while (braces > 0) {
        t += "\n" + step.repeat(Math.max(0, braces - 1)) + "}";
        braces--;
    }
    return t;
}

/**
 * Non-trivial partial body worth showing (better than empty ghost text).
 */
function isUsefulPartial(text: string): boolean {
    const t = text.trim();
    if (t.length < 16) {
        return false;
    }
    if (isStructurallyBroken(t)) {
        return false;
    }
    if (isProseLine(t.split("\n")[0] || "")) {
        return false;
    }
    // At least one real statement keyword
    if (!/\b(return|const|let|var|if|for|while|await|throw|try|switch|async)\b/.test(t)) {
        return false;
    }
    const bal = braceBalance(t);
    if (bal < -1 || bal > 4) {
        return false;
    }
    // Not only a single open brace
    if (/^\{\s*$/.test(t)) {
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
    multiLine: boolean = false,
): PreparedCompletion | null {
    let text = raw;

    // Single-line FIM only: keep first line. Multi-line FIM / intent keep newlines.
    if (!isIntent && !multiLine) {
        const nl = text.search(/\r?\n/);
        if (nl !== -1) {
            text = text.slice(0, nl);
        }
    }

    text = text.replace(/[ \t]+$/g, "");
    if (!text || text === "\n" || text === ";" || text === "\\n") {
        return null;
    }

    // Intent / multi-line FIM: pure insert at cursor (indent already normalized)
    if (isIntent || multiLine) {
        // Never full-line rewrite intent bodies to col 0 with a new const
        if (
            lineBefore.trim().length === 0 &&
            /^(const|let|var|function)\b/.test(text.trim())
        ) {
            // Signature already above empty body — strip wrapper again
            text = refineBodyInsert(text, document.getText(), lineBefore);
        }
        // After `{|` the model often emits a leading newline then the body.
        if (/\{\s*$/.test(lineBefore) && text.startsWith("\n")) {
            // keep the newline so the body lands on the next line
        } else {
            text = text.replace(/^\n+/, "");
        }
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
    // Natural-language leakage (e.g. "This code defines a TaskQueue…")
    if (t.split("\n").some((ln) => isProseLine(ln))) {
        return true;
    }
    if (
        /\b(this code defines|the following code|as you can see)\b/i.test(t)
    ) {
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
