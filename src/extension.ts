/**
 * MLX Code Completion — VS Code Extension Entry Point
 */

import * as vscode from "vscode";
import { BackendIPC } from "./backend-ipc";
import { CompletionProvider, setLogger } from "./completion-provider";
import * as path from "path";

let backend: BackendIPC | null = null;
let outputChannel: vscode.OutputChannel | null = null;

export function activate(context: vscode.ExtensionContext): void {
    // Create output channel for debug logs
    outputChannel = vscode.window.createOutputChannel("MLX Code Completion");
    setLogger((msg: string) => outputChannel!.appendLine(msg));
    outputChannel.appendLine("[MLX] Extension activated");
    try {
        require("fs").writeFileSync("/tmp/mlx_activated.log", new Date().toISOString() + "\n");
    } catch {}

    // Ensure inline suggestions and quick suggestions are enabled in this window.
    vscode.workspace
        .getConfiguration("editor")
        .update("inlineSuggest.enabled", true, vscode.ConfigurationTarget.Global)
        .then(
            () => outputChannel!.appendLine("[MLX] Enabled editor.inlineSuggest.enabled"),
            (e: any) => outputChannel!.appendLine(`[MLX] Could not enable inlineSuggest: ${e}`),
        );
    vscode.workspace
        .getConfiguration("editor")
        .update("quickSuggestions", { other: true, comments: false, strings: false }, vscode.ConfigurationTarget.Global)
        .then(
            () => outputChannel!.appendLine("[MLX] Enabled editor.quickSuggestions"),
            (e: any) => outputChannel!.appendLine(`[MLX] Could not enable quickSuggestions: ${e}`),
        );
    vscode.window.showInformationMessage("MLX Code Completion: extension activated.");

     const config = vscode.workspace.getConfiguration("mlxCompletion");
    const modelPath = config.get<string>("modelPath") || "";
    const quantization = config.get<string>("quantization") || "4bit";
    const debounceMs = config.get<number>("debounceMs") || 50;
    const maxTokens = config.get<number>("maxTokens") || 64;
    const contextLinesBefore = config.get<number>("contextLinesBefore") || 150;
    const contextLinesAfter = config.get<number>("contextLinesAfter") || 35;

    outputChannel.appendLine(`[MLX] Model path: ${modelPath}`);
    outputChannel.appendLine(`[MLX] Quantization: ${quantization}`);
    outputChannel.appendLine(`[MLX] Debounce: ${debounceMs}ms`);
    outputChannel.appendLine(`[MLX] Max tokens: ${maxTokens}`);

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
        outputChannel,
    );

     // Start the backend (loads the model)
    backend.start().then(() => {
        outputChannel!.appendLine("[MLX] Backend started successfully");
        vscode.window.showInformationMessage("MLX Code Completion: Model loaded and ready.");
    }).catch((err) => {
        outputChannel!.appendLine(`[MLX] Backend start failed: ${err.message}`);
        vscode.window.showErrorMessage(
              `MLX Code Completion: Failed to start backend: ${err.message}. ` +
              "Please configure the model path in settings."
          );
    });

     // Register inline completion provider (ghost text) for all languages.
     // The universal "*" selector is unreliable for inline providers in some
     // VS Code versions, so register an explicit broad set of languages.
    const provider = new CompletionProvider(
        backend,
        debounceMs,
        maxTokens,
        outputChannel,
        contextLinesBefore,
        contextLinesAfter,
    );

    const languages = [
        "javascript", "typescript", "python", "json", "html", "css",
        "java", "c", "cpp", "csharp", "go", "rust", "ruby", "php",
        "swift", "kotlin", "shellscript", "sql", "markdown", "yaml",
        "plaintext", "xml", "scss", "less", "vue", "jsx", "tsx",
    ];
    const selector = languages.map((l) => ({ language: l }));

    const reg = vscode.languages.registerCompletionItemProvider(
        selector,
        provider,
        // Trigger characters that prompt a completion request.
        ".", "(", "\"", "'", "`", "[", "{", " ", ":", "=", ",", "\n",
        ";", "!", ">", "<", "-", "_", "#", "@", "$", "%", "&", "*", "+", "/", "|", "~",
    );
    context.subscriptions.push(reg);
    outputChannel.appendLine(`[MLX] Completion provider registered for ${languages.length} languages`);

    // Diagnostic: confirm the extension observes document changes in the host.
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((e) => {
            if (e.document.uri.scheme === "file") {
                outputChannel!.appendLine(
                    `[MLX][diag] text changed in ${e.document.uri.fsPath} ` +
                    `(lang=${e.document.languageId}, lines=${e.document.lineCount})`,
                );
            }
        }),
    );
}

export function deactivate(): void {
    backend?.stop();
    backend = null;
    outputChannel?.dispose();
    outputChannel = null;
}
