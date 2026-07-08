/**
 * Inline completion provider (ghost text).
 */

import * as vscode from "vscode";
import { extractContext, detectIntent, DocumentContext } from "./context-extractor";
import { BackendIPC, TokenCallback } from "./backend-ipc";

let logFn: ((msg: string) => void) | null = null;

export function setLogger(fn: (msg: string) => void): void {
    logFn = fn;
}

function log(msg: string): void {
    logFn?.(msg);
}

export class CompletionProvider implements vscode.CompletionItemProvider {
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

    private toCompletionItem(text: string, document: vscode.TextDocument, position: vscode.Position): vscode.CompletionItem {
        const firstLine = text.split("\n")[0] || "completion";
        const label = firstLine.length > 60 ? firstLine.slice(0, 60) + "…" : firstLine;
        const item = new vscode.CompletionItem(label, vscode.CompletionItemKind.Snippet);
        const range = document.getWordRangeAtPosition(position) ?? new vscode.Range(position, position);
        item.range = range;
        item.insertText = new vscode.SnippetString(text);
        item.detail = "MLX Code Completion";
        item.documentation = new vscode.MarkdownString("```\n" + text + "\n```");
        item.sortText = " ";
        item.preselect = true;
        return item;
    }


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

    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        context: vscode.CompletionContext,
    ): Promise<vscode.CompletionItem[]> {
        log(`[Provider] provideCompletionItems called`);
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
        // Intent generations use a larger budget and aren't truncated to one line.
        const isIntent = !!intent;
        const contextKey = isIntent
            ? `intent:${intent}:${position.line}:${position.character}`
            : `${contextData.before.length}:${contextData.after.length}:${position.line}:${position.character}`;

        log(`[Provider] context key: ${contextKey}`);
        log(`[Provider] last context key: ${this.lastContextKey}`);

        // Return a cached completion if we already generated one for this context.
        const cached = this.completions.get(contextKey);
        if (cached && cached.trim().length > 0) {
            log(`[Provider] -> returning cached completion (${cached.length} chars)`);
            return [this.toCompletionItem(cached, document, position)];
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
        // request cancellation. We return a promise that resolves with the
        // completion as soon as generation finishes, so whichever caller invoked
        // us (inline provider or fallback dropdown) gets the result directly.
        this.inFlight.set(contextKey, true);
        let accumulated = "";
        const streamCb: TokenCallback = (tokenText: string) => {
            if (tokenText) {
                accumulated += tokenText;
                log(`[Provider] -> token: ${JSON.stringify(tokenText)}`);
            }
        };

        log("[Provider] -> starting background generation" + (isIntent ? " (intent mode)" : ""));
        return new Promise<vscode.CompletionItem[]>((resolve) => {
            this.backend
                .complete(
                    contextData,
                    undefined as unknown as vscode.CancellationToken,
                    streamCb,
                    isIntent ? 512 : undefined,
                )
                .then(() => {
                    this.inFlight.delete(contextKey);
                    const cleaned = cleanCompletion(accumulated, contextData.before, isIntent);
                    if (cleaned) {
                        this.completions.set(contextKey, cleaned);
                        log(`[Provider] -> cached completion (${cleaned.length} chars)`);
                        log(`[Provider] -> CLEANED TEXT: ${JSON.stringify(cleaned)}`);
                        resolve([this.toCompletionItem(cleaned, document, position)]);
                    } else {
                        log("[Provider] -> empty completion, not caching");
                        resolve([]);
                    }
                })
                .catch((err) => {
                    this.inFlight.delete(contextKey);
                    log(`[Provider] -> backend error: ${err.message}`);
                    resolve([]);
                });
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
function cleanCompletion(raw: string, before: string, isIntent = false): string {
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

    // The model echoes the code already before the cursor. Strip any leading
    // text that duplicates the current line's prefix (e.g. you typed
    // "const c" and the model emits "const d = ..." -> drop the echoed "const ").
    const beforeLines = before.split("\n");
    const lastBeforeLine = beforeLines[beforeLines.length - 1];
    const cursorPrefix = lastBeforeLine.trimStart();
    if (cursorPrefix && text.startsWith(cursorPrefix)) {
        text = text.slice(cursorPrefix.length);
    }
    // Drop any leftover indentation from the stripped prefix.
    if (!isIntent) {
        text = text.replace(/^\s+/, "");
    }

    // Stop at the first blank line: for inline (mid-line) completion we only
    // want the immediate continuation, not a whole function body + examples.
    // For intent mode we keep the full generated implementation.
    if (!isIntent) {
        const firstBlank = text.search(/\n\s*\n/);
        if (firstBlank !== -1) {
            text = text.slice(0, firstBlank);
        }

        // Drop runaway repetition (model loops on example outputs).
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
    }

    text = text.replace(/\s+$/, "");
    return text;
}
