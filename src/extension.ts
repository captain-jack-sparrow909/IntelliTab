/**
 * MLX Code Completion — VS Code Extension Entry Point
 *
 * Spawns the Python MLX server and registers the completion provider.
 */

import * as vscode from "vscode";
import { BackendIPC } from "./backend-ipc";
import { CompletionProvider } from "./completion-provider";
import * as path from "path";

let backend: BackendIPC | null = null;

export function activate(context: vscode.ExtensionContext): void {
    const config = vscode.workspace.getConfiguration("mlxCompletion");
    const modelPath = config.get<string>("modelPath") || "";
    const quantization = config.get<string>("quantization") || "4bit";
    const debounceMs = config.get<number>("debounceMs") || 50;
    const maxTokens = config.get<number>("maxTokens") || 64;

    // Find the Python server script relative to this extension
    const extensionRoot = context.extensionPath;
    const serverScript = path.join(extensionRoot, "python-server", "server.py");

    // Check if server script exists
    if (!require("fs").existsSync(serverScript)) {
        vscode.window.showErrorMessage(
            `MLX Code Completion: Server script not found at ${serverScript}. ` +
            "Please ensure the python-server directory is present."
        );
        return;
    }

    // Initialize backend IPC
    backend = new BackendIPC(
        serverScript,
        modelPath,
        quantization,
        maxTokens,
        config.get<number>("temperature") || 0.0,
    );

    // Start the backend (loads the model)
    backend.start().then(() => {
        vscode.window.showInformationMessage("MLX Code Completion: Model loaded and ready.");
    }).catch((err) => {
        vscode.window.showErrorMessage(
            `MLX Code Completion: Failed to start backend: ${err.message}. ` +
            "Please configure the model path in settings."
        );
    });

    // Register completion provider for all languages
    const provider = new CompletionProvider(
        backend,
        debounceMs,
        maxTokens,
    );

    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            [{ pattern: "*" }],
            provider,
            ".",
            "(",
            '"',
            "'",
            "`",
            "[",
            "{",
            " ",
            ":",
            "=",
            ",",
        ),
    );
}

export function deactivate(): void {
    backend?.stop();
    backend = null;
}
