/**
 * Inline completion provider (ghost text) — latency-first.
 *
 * Strategy:
 * 1. Cancel any previous backend job when the cursor context changes
 * 2. Stream tokens; paint partial ghost text as soon as usable
 * 3. Prefer single-line completions (early stop) for the common case
 * 4. Cache final + partial results for re-trigger after VS Code cancels
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

interface GenState {
    promise: Promise<string | null>;
    partial: string;
    done: boolean;
}

export class CompletionProvider implements vscode.InlineCompletionItemProvider {
    private maxTokens: number;
    private linesBefore: number;
    private linesAfter: number;
    private backend: BackendIPC;

    /** Finalized completions. */
    private completions = new Map<string, string>();
    /** In-flight / partial generations by context key. */
    private gens = new Map<string, GenState>();
    /** Only one logical generation at a time (matches single GPU worker). */
    private activeKey: string | null = null;

    private partialRetriggerTimer: ReturnType<typeof setTimeout> | null = null;
    private lastPartialShown = "";

    constructor(
        backend: BackendIPC,
        _debounceMs: number,
        maxTokens: number,
        outputChannel: vscode.OutputChannel | null,
        linesBefore: number = 60,
        linesAfter: number = 15,
    ) {
        this.backend = backend;
        this.maxTokens = maxTokens;
        this.linesBefore = linesBefore;
        this.linesAfter = linesAfter;
        setLogger((msg: string) => outputChannel?.appendLine(msg));
        log("Inline completion provider constructed (latency mode)");
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
        if (!this.backend.isRunning()) {
            return null;
        }
        if (!this.isCompletionValid(document, position)) {
            return null;
        }

        const t0 = Date.now();
        const intent = detectIntent(document, position);
        const contextData: DocumentContext = extractContext(
            document,
            position,
            this.linesBefore,
            this.linesAfter,
        );
        if (intent) {
            contextData.intent = intent;
        }
        const isIntent = !!intent;

        const linePrefix = document
            .lineAt(position.line)
            .text.substring(0, position.character);
        const contextKey = isIntent
            ? `i:${intent}:${position.line}:${linePrefix}`
            : `c:${linePrefix}|${hashShort(contextData.after)}|${position.line}:${position.character}`;

        // 1) Final cache
        const cached = this.completions.get(contextKey);
        if (cached) {
            return this.toInlineItems(cached, position);
        }

        // 2) Partial already streaming — show it immediately (fast path)
        const existing = this.gens.get(contextKey);
        if (existing?.partial) {
            if (existing.done && existing.partial) {
                this.completions.set(contextKey, existing.partial);
                return this.toInlineItems(existing.partial, position);
            }
            // Return current partial; keep generation going.
            if (token.isCancellationRequested) {
                return null;
            }
            return this.toInlineItems(existing.partial, position);
        }

        // 3) Join in-flight without partial yet — wait briefly for first tokens
        if (existing && !existing.done) {
            const first = await waitForPartialOrDone(existing, token, 120);
            if (token.isCancellationRequested) {
                return null;
            }
            if (first) {
                return this.toInlineItems(first, position);
            }
            // Still nothing usable — let re-trigger pick it up
            return null;
        }

        // 4) Start a new generation (cancels any previous backend job)
        if (this.activeKey && this.activeKey !== contextKey) {
            this.gens.delete(this.activeKey);
        }
        this.activeKey = contextKey;

        const gen = this.startGeneration(contextData, isIntent, document, position, contextKey);
        this.gens.set(contextKey, gen);

        // Wait a short window for first paint so one provider call can show ghost text
        // without needing a full re-trigger when the model is warm.
        const first = await waitForPartialOrDone(gen, token, 400);
        log(
            `[Provider] key=${contextKey.slice(0, 60)} first=${first ? first.length : 0}ch ` +
                `in ${Date.now() - t0}ms cancel=${token.isCancellationRequested}`,
        );

        if (token.isCancellationRequested) {
            // Generation continues; partial re-triggers will show ghost text.
            return first ? this.toInlineItems(first, position) : null;
        }
        if (!first) {
            return null;
        }
        return this.toInlineItems(first, position);
    }

    private startGeneration(
        contextData: DocumentContext,
        isIntent: boolean,
        document: vscode.TextDocument,
        position: vscode.Position,
        contextKey: string,
    ): GenState {
        const state: GenState = {
            promise: Promise.resolve(null),
            partial: "",
            done: false,
        };

        const t0 = Date.now();
        let firstTokenAt = 0;
        let accumulated = "";

        // Mid-line / empty-body: stop at first newline for snappy single-line ghost text.
        // Multi-line intent (comment→code) keeps going.
        const stopOnNewline = !isIntent;
        const maxTok = isIntent ? Math.max(this.maxTokens, 128) : this.maxTokens;

        const onToken = (tokenText: string) => {
            if (!tokenText) {
                return;
            }
            if (!firstTokenAt) {
                firstTokenAt = Date.now();
                log(`[Provider] TTFT ${firstTokenAt - t0}ms`);
            }
            accumulated += tokenText;

            let cleaned = cleanCompletion(accumulated, contextData.before, isIntent);
            if (cleaned) {
                cleaned = normalizeIndentation(cleaned, document, position);
            }
            if (cleaned && cleaned.length > 0 && cleaned !== state.partial) {
                state.partial = cleaned;
                // Throttled re-trigger so VS Code paints progressive ghost text
                // even if the original provider call was cancelled.
                this.schedulePartialRetrigger(cleaned);
            }
        };

        state.promise = this.backend
            .complete(contextData, onToken, {
                maxTokens: maxTok,
                stopOnNewline,
            })
            .then(() => {
                state.done = true;
                let cleaned = cleanCompletion(accumulated, contextData.before, isIntent);
                if (cleaned) {
                    cleaned = normalizeIndentation(cleaned, document, position);
                }
                if (cleaned && cleaned.length > 0) {
                    state.partial = cleaned;
                    this.completions.set(contextKey, cleaned);
                    if (this.completions.size > 40) {
                        const first = this.completions.keys().next().value;
                        if (first !== undefined) {
                            this.completions.delete(first);
                        }
                    }
                    log(
                        `[Provider] done ${Date.now() - t0}ms ` +
                            `ttft=${firstTokenAt ? firstTokenAt - t0 : -1}ms ` +
                            `len=${cleaned.length}`,
                    );
                    this.schedulePartialRetrigger(cleaned, true);
                    return cleaned;
                }
                log(`[Provider] empty after ${Date.now() - t0}ms`);
                return null;
            })
            .catch((err: Error) => {
                state.done = true;
                log(`[Provider] error: ${err.message}`);
                return state.partial || null;
            })
            .finally(() => {
                if (this.activeKey === contextKey) {
                    this.activeKey = null;
                }
            });

        return state;
    }

    private schedulePartialRetrigger(text: string, force = false): void {
        if (!force && text === this.lastPartialShown) {
            return;
        }
        if (this.partialRetriggerTimer !== null) {
            // Coalesce rapid tokens (~1 re-trigger per 48ms)
            return;
        }
        this.partialRetriggerTimer = setTimeout(() => {
            this.partialRetriggerTimer = null;
            this.lastPartialShown = text;
            void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
        }, force ? 0 : 48);
    }

    private toInlineItems(
        text: string,
        position: vscode.Position,
    ): vscode.InlineCompletionList {
        const item = new vscode.InlineCompletionItem(text, new vscode.Range(position, position));
        return { items: [item] };
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
        if (this.partialRetriggerTimer !== null) {
            clearTimeout(this.partialRetriggerTimer);
        }
        this.completions.clear();
        this.gens.clear();
        this.backend.cancelActive();
    }
}

// --- helpers -----------------------------------------------------------------

function hashShort(s: string): string {
    // Cheap stable fingerprint for context key (not cryptographic).
    let h = 0;
    const n = Math.min(s.length, 120);
    for (let i = 0; i < n; i++) {
        h = (h * 31 + s.charCodeAt(i)) | 0;
    }
    return `${s.length}:${h}`;
}

/** Wait until partial text exists, gen completes, cancel, or timeout. */
function waitForPartialOrDone(
    state: GenState,
    token: vscode.CancellationToken,
    timeoutMs: number,
): Promise<string | null> {
    if (state.partial) {
        return Promise.resolve(state.partial);
    }
    if (state.done) {
        return Promise.resolve(state.partial || null);
    }
    if (token.isCancellationRequested) {
        return Promise.resolve(null);
    }

    return new Promise((resolve) => {
        const start = Date.now();
        const interval = setInterval(() => {
            if (state.partial) {
                clearInterval(interval);
                sub.dispose();
                resolve(state.partial);
                return;
            }
            if (state.done) {
                clearInterval(interval);
                sub.dispose();
                resolve(state.partial || null);
                return;
            }
            if (token.isCancellationRequested || Date.now() - start >= timeoutMs) {
                clearInterval(interval);
                sub.dispose();
                resolve(state.partial || null);
            }
        }, 16);

        const sub = token.onCancellationRequested(() => {
            clearInterval(interval);
            sub.dispose();
            resolve(state.partial || null);
        });
    });
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
        return lines[0].replace(/^\s+/, "");
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

function cleanCompletion(raw: string, before: string, isIntent = false): string {
    let text = raw;

    const eot = text.indexOf("<|endoftext|>");
    if (eot !== -1) {
        text = text.slice(0, eot);
    }

    text = text.replace(/<\|fim_(?:begin|end|pad|prefix|suffix|middle)\|>/g, "");
    text = text.replace(/<\|endoftext\|>/g, "");
    text = text.replace(/<\|im_(?:end|start)\|>/g, "");
    text = text.replace(/```[a-zA-Z0-9]*\n?/g, "");
    text = text.replace(/```/g, "");
    text = text.replace(/^\s*\n/, "");
    text = text.replace(/^>\s*/, "");

    const beforeLines = before.split("\n");
    const lastBeforeLine = beforeLines[beforeLines.length - 1] ?? "";
    const cursorPrefix = lastBeforeLine.trimStart();
    if (cursorPrefix && text.startsWith(cursorPrefix)) {
        text = text.slice(cursorPrefix.length);
    }
    if (!isIntent && lastBeforeLine.length > 0) {
        const tail = lastBeforeLine.slice(-40);
        if (tail && text.startsWith(tail)) {
            text = text.slice(tail.length);
        }
    }

    if (!isIntent) {
        text = text.replace(/^\s+/, "");
        // Prefer first line only for progressive mid-line completions.
        const nl = text.indexOf("\n");
        if (nl !== -1) {
            text = text.slice(0, nl);
        }
    }

    text = text.replace(/[ \t]+$/gm, "");
    text = text.replace(/\s+$/, "");
    return text;
}
