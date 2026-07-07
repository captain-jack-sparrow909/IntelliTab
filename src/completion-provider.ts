/**
 * Inline completion provider.
 *
 * Uses VS Code's InlineCompletionItemProvider API (like GitHub Copilot)
 * to show streaming, token-by-token completions as inline ghost text.
 * Listens for cursor position changes (keyboard only), debounces input,
 * extracts document context, and streams completions from the MLX backend.
 */

import * as vscode from "vscode";
import { extractContext } from "./context-extractor";
import { BackendIPC, TokenCallback } from "./backend-ipc";

/**
 * A running completion session. Tracks the current request so we can
 * cancel it when the user types again.
 */
interface CompletionSession {
    controller: vscode.CancellationTokenSource;
    item: vscode.InlineCompletionItem;
    accumulated: string;
}

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private currentSession: CompletionSession | null = null;
    private lastRequestKey = "";
    private debounceTimer: ReturnType<typeof setTimeout> | null = null;
    private debounceMs: number;
    private maxTokens: number;

    constructor(
        private backend: BackendIPC,
        debounceMs: number,
        maxTokens: number,
    ) {
        this.debounceMs = debounceMs;
        this.maxTokens = maxTokens;
    }

    /**
     * Called by VS Code when inline completions are requested.
     * We return null immediately and stream completions via the callback.
     */
    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken,
    ): Promise<vscode.InlineCompletionItem[] | null> {
        // Only work with file URIs
        if (document.uri.scheme !== "file") {
            return null;
        }

        // Ignore if cursor is at the very start
        if (position.line === 0 && position.character === 0) {
            return null;
        }

        // Build a request key to detect meaningful changes
        const requestKey = `${position.line}:${position.character}:${document.languageId}`;
        if (requestKey === this.lastRequestKey) {
            return null;
        }
        this.lastRequestKey = requestKey;

        // Cancel any in-flight session
        this.cancelSession();

        // Check if cursor is at a valid position (not inside a multi-line comment)
        if (!this.isCompletionValid(document, position)) {
            return null;
        }

        // Extract context window
        const docContext = extractContext(document, position, 150, 35);

        // Create a cancellation token for this request
        const controller = new vscode.CancellationTokenSource();
        const requestToken = controller.token;

        // Listen for top-level cancellation (e.g., user presses Escape)
        token.onCancellationRequested(() => {
            controller.cancel();
            controller.dispose();
            this.cancelSession();
        });

        // Create the inline completion item (empty initially, filled by streaming)
        const wordRange = this.getWordRange(document, position);
        const item = new vscode.InlineCompletionItem("", wordRange);

        // Track the session
        this.currentSession = {
            controller,
            item,
            accumulated: "",
        };

        // Streaming callback — called for each token from the backend
        const onToken: TokenCallback = (tokenText: string) => {
            if (requestToken.isCancellationRequested || token.isCancellationRequested) {
                return;
            }

            if (tokenText) {
                this.currentSession!.accumulated += tokenText;
                this.currentSession!.item.insertText = this.currentSession!.accumulated;

                // Notify VS Code that the inline item has been updated
                // This triggers a re-render with the new text
                vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
            } else {
                // Empty token signals end of stream
                this.cancelSession();
            }
        };

        // Send request to backend
        try {
            await this.backend.complete(docContext, requestToken, onToken);
        } catch {
            // Request was cancelled or errored — session already cleaned up
        }

        // Return null because we're streaming updates, not returning items upfront
        // VS Code will see the updated item via the inline suggest commands
        return null;
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

    private cancelSession(): void {
        if (this.currentSession) {
            this.currentSession.controller.cancel();
            this.currentSession.controller.dispose();
            this.currentSession = null;
        }
    }
}
