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

        // Don't re-request if context hasn't changed
        if (contextKey === this.lastContextKey) {
            log("[Provider] -> context unchanged, returning []");
            return [];
        }
        this.lastContextKey = contextKey;

        // Cancel any pending debounce
        if (this.debounceTimer !== null) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
            log("[Provider] -> cancelled previous debounce");
        }

        // Debounce
        return new Promise<vscode.InlineCompletionItem[]>((resolve) => {
            log(`[Provider] -> scheduling request after ${this.debounceMs}ms`);
            this.debounceTimer = setTimeout(async () => {
                this.debounceTimer = null;
                log("[Provider] -> debounce fired, sending request to backend");

                // Check if cancelled during debounce
                if (token.isCancellationRequested) {
                    log("[Provider] -> cancelled during debounce");
                    resolve([]);
                    return;
                }

                try {
                    const controller = new vscode.CancellationTokenSource();
                    const requestToken = controller.token;

                    // Listen for cancellation
                    token.onCancellationRequested(() => {
                        log("[Provider] -> cancelled by user");
                        controller.cancel();
                        controller.dispose();
                        resolve([]);
                    });

                    // Stream callback — accumulate tokens; the result is
                    // resolved by backend.complete() once the stream ends.
                    let accumulated = "";
                    const onToken: TokenCallback = (tokenText: string) => {
                        if (requestToken.isCancellationRequested || token.isCancellationRequested) {
                            return;
                        }
                        if (tokenText) {
                            accumulated += tokenText;
                            log(`[Provider] -> token: ${JSON.stringify(tokenText)}`);
                        }
                    };

                    log("[Provider] -> calling backend.complete()");
                    this.backend
                        .complete(contextData, requestToken, onToken)
                        .then(() => {
                            controller.dispose();
                            log(`[Provider] -> backend complete. accumulated: ${JSON.stringify(accumulated)}`);

                            if (requestToken.isCancellationRequested || token.isCancellationRequested) {
                                log("[Provider] -> cancelled during generation");
                                resolve([]);
                                return;
                            }

                            if (!accumulated || accumulated.trim().length === 0) {
                                log("[Provider] -> no tokens generated, returning []");
                                resolve([]);
                                return;
                            }

                            const range = new vscode.Range(position, position);
                            const item = new vscode.InlineCompletionItem(
                                accumulated,
                                range,
                            );
                            item.command = {
                                title: "MLX Code Completion",
                                command: "",
                            };

                            log(`[Provider] -> returning ${accumulated.length} chars`);
                            resolve([item]);
                        })
                        .catch((err) => {
                            controller.dispose();
                            log(`[Provider] -> backend error: ${err.message}`);
                            resolve([]);
                        });
                } catch (err) {
                    log(`[Provider] -> exception: ${err}`);
                    resolve([]);
                }
            }, this.debounceMs);
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
