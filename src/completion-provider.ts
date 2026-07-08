/**
 * Inline completion provider (ghost text).
 */

import * as vscode from "vscode";
import { extractContext, DocumentContext } from "./context-extractor";
import { BackendIPC, TokenCallback } from "./backend-ipc";

let logFn: ((msg: string) => void) | null = null;

export function setLogger(fn: (msg: string) => void): void {
    logFn = fn;
}

function log(msg: string): void {
    logFn?.(msg);
}

export class CompletionProvider implements vscode.InlineCompletionItemProvider {
    private lastContextKey = "";
    private debounceTimer: ReturnType<typeof setTimeout> | null = null;
    private debounceMs: number;
    private maxTokens: number;
    private linesBefore: number;
    private linesAfter: number;
    private backend: BackendIPC;
    // Cache of generated completions keyed by contextKey. Generation runs to
    // completion independent of VS Code's request cancellation, so a result is
    // available when the provider is queried again (or on the next keystroke).
    private completions = new Map<string, string>();
    private inFlight = new Map<string, boolean>();
    // Holds the active request's resolve + position so we can fulfill it with a
    // (partial) completion once generation finishes, even though VS Code may
    // cancel its own token in the meantime.
    private pendingResolve: ((items: vscode.InlineCompletionItem[]) => void) | null = null;
    private pendingPosition: vscode.Position | null = null;
    private pendingContextKey: string | null = null;

    constructor(
        backend: BackendIPC,
        debounceMs: number,
        maxTokens: number,
        outputChannel: vscode.OutputChannel | null,
        linesBefore: number = 150,
        linesAfter: number = 35,
    ) {
        this.backend = backend;
        this.debounceMs = debounceMs;
        this.maxTokens = maxTokens;
        this.linesBefore = linesBefore;
        this.linesAfter = linesAfter;
        setLogger((msg: string) => outputChannel?.appendLine(msg));
        log("Provider constructed");
    }

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken,
    ): Promise<vscode.InlineCompletionItem[]> {
        log(`[Provider] provideInlineCompletionItems called`);
        log(`[Provider] document: ${document.uri.fsPath}`);
        log(`[Provider] scheme: ${document.uri.scheme}`);
        log(`[Provider] position: line=${position.line}, char=${position.character}`);
        log(`[Provider] backend isRunning: ${this.backend.isRunning()}`);

        // Only work with file URIs
        if (document.uri.scheme !== "file") {
            log("[Provider] -> NOT a file URI, returning []");
            return [];
        }

        // Ignore if cursor is at the very start
        if (position.line === 0 && position.character === 0) {
            log("[Provider] -> cursor at start, returning []");
            return [];
        }

        // Check if cursor is at a valid position
        if (!this.isCompletionValid(document, position)) {
            log("[Provider] -> invalid position, returning []");
            return [];
        }

        // Build a context key
        const contextData = extractContext(
            document,
            position,
            this.linesBefore,
            this.linesAfter,
        );
        const contextKey = `${contextData.before.length}:${contextData.after.length}:${position.line}:${position.character}`;

        log(`[Provider] context key: ${contextKey}`);
        log(`[Provider] last context key: ${this.lastContextKey}`);

        // Return a cached completion if we already generated one for this context.
        const cached = this.completions.get(contextKey);
        if (cached && cached.trim().length > 0) {
            log(`[Provider] -> returning cached completion (${cached.length} chars)`);
            const item = new vscode.InlineCompletionItem(cached, new vscode.Range(position, position));
            return [item];
        }

        // If a generation for this context is already running, don't start another.
        if (this.inFlight.get(contextKey)) {
            log("[Provider] -> generation already in flight, returning []");
            return [];
        }

        // Don't re-request if context hasn't changed since last keystroke-triggered gen
        if (contextKey === this.lastContextKey) {
            log("[Provider] -> context unchanged, returning []");
            return [];
        }
        this.lastContextKey = contextKey;

        // Kick off generation. This runs to completion independent of VS Code's
        // request cancellation — VS Code re-queries the provider (or the next
        // keystroke does) and we return the cached result once ready.
        this.inFlight.set(contextKey, true);
        let accumulated = "";
        const streamCb: TokenCallback = (tokenText: string) => {
            if (tokenText) {
                accumulated += tokenText;
                log(`[Provider] -> token: ${JSON.stringify(tokenText)}`);
            }
        };

        log("[Provider] -> starting background generation");
        this.backend
            .complete(contextData, undefined as unknown as vscode.CancellationToken, streamCb)
            .then(() => {
                this.inFlight.delete(contextKey);
                const cleaned = cleanCompletion(accumulated, contextData.before);
                if (cleaned) {
                    this.completions.set(contextKey, cleaned);
                    log(`[Provider] -> cached completion (${cleaned.length} chars)`);
                } else {
                    log("[Provider] -> empty completion, not caching");
                }
                // If the request that started this generation is still waiting,
                // fulfill it now with the result.
                if (this.pendingResolve && this.pendingContextKey === contextKey) {
                    const pos = this.pendingPosition!;
                    const text = this.completions.get(contextKey);
                    this.pendingResolve(text ? [new vscode.InlineCompletionItem(text, new vscode.Range(pos, pos))] : []);
                    this.pendingResolve = null;
                    this.pendingPosition = null;
                    this.pendingContextKey = null;
                } else if (cleaned) {
                    // VS Code didn't keep our request open (it cancelled it, which
                    // is common). Re-trigger inline suggestions so the provider is
                    // invoked again and can serve the now-cached completion.
                    log("[Provider] -> re-triggering inline suggestions");
                    vscode.commands.executeCommand("editor.action.inlineSuggest.trigger").then(
                        () => {},
                        (e: any) => log(`[Provider] -> trigger failed: ${e}`),
                    );
                }
            })
            .catch((err) => {
                this.inFlight.delete(contextKey);
                log(`[Provider] -> backend error: ${err.message}`);
            });

        // Hold this request open until generation completes, so VS Code shows the
        // ghost text as soon as it's ready (instead of waiting for the next keystroke).
        return new Promise<vscode.InlineCompletionItem[]>((resolve) => {
            this.pendingResolve = resolve;
            this.pendingPosition = position;
            this.pendingContextKey = contextKey;
        });
    }

    private isCompletionValid(document: vscode.TextDocument, position: vscode.Position): boolean {
        const line = document.lineAt(position.line);
        const beforeCursor = line.text.substring(0, position.character);

        if (beforeCursor.match(/\/\*/)) {
            return false;
        }

        return true;
    }

    dispose(): void {
        if (this.debounceTimer !== null) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    }
}

/**
 * Clean a raw model completion into a usable inline suggestion.
 *
 * The model tends to wrap output in markdown code fences (```javascript ... ```)
 * and to echo the code already present before the cursor. We strip the fences
 * and remove any leading text that duplicates what's already at the cursor.
 */
function cleanCompletion(raw: string, before: string): string {
    let text = raw;

    // Stop generation at the end-of-text marker.
    const eot = text.indexOf("<|endoftext|>");
    if (eot !== -1) {
        text = text.slice(0, eot);
    }

    // Remove model control tokens (FIM markers, etc.).
    text = text.replace(/<\|fim_begin\|>/g, "");
    text = text.replace(/<\|fim_end\|>/g, "");
    text = text.replace(/<\|fim_pad\|>/g, "");
    text = text.replace(/<\|endoftext\|>/g, "");

    // Remove markdown code fences anywhere (handles tokens split across fences).
    text = text.replace(/```[a-zA-Z0-9]*\n?/g, "");
    text = text.replace(/```/g, "");

    // Collapse the leading blank lines the model often emits.
    text = text.replace(/^\s*\n/, "");
    // Remove a stray leading '>' left over from FIM markers.
    text = text.replace(/^>\s*/, "");

    // The model echoes the line already before the cursor. Strip a leading
    // prefix that matches the tail of `before` (typically the current line).
    const beforeLines = before.split("\n");
    const lastBeforeLine = beforeLines[beforeLines.length - 1];
    if (lastBeforeLine && text.startsWith(lastBeforeLine)) {
        text = text.slice(lastBeforeLine.length);
    }

    // If it still starts by repeating the whole `before` tail more broadly,
    // remove a leading segment equal to the cursor-line prefix.
    const cursorPrefix = lastBeforeLine.trimStart();
    if (cursorPrefix && text.startsWith(cursorPrefix)) {
        text = text.slice(cursorPrefix.length);
    }

    // Stop at the first blank-line-delimited repetition (model loops).
    const lines = text.split("\n");
    const seen = new Set<string>();
    const out: string[] = [];
    for (const ln of lines) {
        const key = ln.trim();
        if (key && seen.has(key) && out.length > 0) {
            break;
        }
        if (key) {
            seen.add(key);
        }
        out.push(ln);
    }
    text = out.join("\n");

    text = text.replace(/\s+$/, "");
    return text;
}
