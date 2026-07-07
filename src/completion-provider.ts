/**
 * Completion provider.
 *
 * Uses the traditional CompletionItemProvider. Returns a Thenable that
 * resolves with completion items after the MLX backend finishes generating.
 * VS Code waits for the Thenable and displays the results.
 */

import * as vscode from "vscode";
import { extractContext, DocumentContext } from "./context-extractor";
import { BackendIPC, TokenCallback } from "./backend-ipc";

export class CompletionProvider implements vscode.CompletionItemProvider {
    private lastContextKey = "";
    private debounceTimer: ReturnType<typeof setTimeout> | null = null;
    private debounceMs: number;
    private maxTokens: number;
    private backend: BackendIPC;

    constructor(backend: BackendIPC, debounceMs: number, maxTokens: number) {
        this.backend = backend;
        this.debounceMs = debounceMs;
        this.maxTokens = maxTokens;
    }

    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken,
        _context: vscode.CompletionContext,
    ): Promise<vscode.CompletionItem[]> {
        // Only work with file URIs
        if (document.uri.scheme !== "file") {
            return [];
        }

        // Ignore if cursor is at the very start
        if (position.line === 0 && position.character === 0) {
            return [];
        }

        // Check if cursor is at a valid position
        if (!this.isCompletionValid(document, position)) {
            return [];
        }

        // Build a context key to detect meaningful changes
        const contextData = extractContext(document, position, 150, 35);
        const contextKey = `${contextData.before.length}:${contextData.after.length}:${position.line}:${position.character}`;

        // Don't re-request if context hasn't changed
        if (contextKey === this.lastContextKey) {
            return [];
        }
        this.lastContextKey = contextKey;

        // Cancel any pending debounce
        if (this.debounceTimer !== null) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }

        // Debounce: wait for user to stop typing
        return new Promise<vscode.CompletionItem[]>((resolve) => {
            this.debounceTimer = setTimeout(async () => {
                this.debounceTimer = null;

                // Check if cancelled during debounce
                if (token.isCancellationRequested) {
                    resolve([]);
                    return;
                }

                try {
                    const controller = new vscode.CancellationTokenSource();
                    const requestToken = controller.token;

                    // Listen for cancellation (when user types again during generation)
                    token.onCancellationRequested(() => {
                        controller.cancel();
                        controller.dispose();
                        resolve([]);
                    });

                    // Stream callback — accumulate tokens
                    let accumulated = "";
                    const onToken: TokenCallback = (tokenText: string) => {
                        if (requestToken.isCancellationRequested || token.isCancellationRequested) {
                            return;
                        }
                        if (tokenText) {
                            accumulated += tokenText;
                        }
                        // Empty token signals end of stream
                    };

                    // Send request to backend
                    this.backend
                        .complete(contextData, requestToken, onToken)
                        .then(() => {
                            controller.dispose();

                            // Check if cancelled during generation
                            if (requestToken.isCancellationRequested || token.isCancellationRequested) {
                                resolve([]);
                                return;
                            }

                            // If no tokens were streamed (backend returned empty), return empty
                            if (!accumulated || accumulated.trim().length === 0) {
                                resolve([]);
                                return;
                            }

                            // Create completion item
                            const wordRange = this.getWordRange(document, position);
                            const item = new vscode.CompletionItem(
                                accumulated.trim(),
                                vscode.CompletionItemKind.Snippet,
                            );
                            item.insertText = accumulated;
                            item.range = wordRange;
                            item.detail = "MLX Code Completion";
                            item.sortText = "\0"; // Put at the top

                            resolve([item]);
                        })
                        .catch((err) => {
                            controller.dispose();
                            resolve([]);
                        });
                } catch (err) {
                    resolve([]);
                }
            }, this.debounceMs);
        });
    }

    private isCompletionValid(document: vscode.TextDocument, position: vscode.Position): boolean {
        const line = document.lineAt(position.line);
        const beforeCursor = line.text.substring(0, position.character);

        // Don't provide completions inside multi-line comments
        if (beforeCursor.match(/\/\*/)) {
            return false;
        }

        return true;
    }

    private getWordRange(document: vscode.TextDocument, position: vscode.Position): vscode.Range {
        const word = document.getWordRangeAtPosition(position);
        return word || new vscode.Range(position, position);
    }

    dispose(): void {
        if (this.debounceTimer !== null) {
            clearTimeout(this.debounceTimer);
            this.debounceTimer = null;
        }
    }
}
