/**
 * MLX Code Completion — VS Code Extension Entry Point
 */

import * as vscode from "vscode";
import { BackendIPC } from "./backend-ipc";
import { CompletionProvider, setLogger } from "./completion-provider";
import * as path from "path";
import * as fs from "fs";

let backend: BackendIPC | null = null;
let outputChannel: vscode.OutputChannel | null = null;

export function activate(context: vscode.ExtensionContext): void {
    outputChannel = vscode.window.createOutputChannel("MLX Code Completion");
    setLogger((msg: string) => outputChannel!.appendLine(msg));
    outputChannel.appendLine("[MLX] Extension activated (latency mode)");

    const editorConfig = vscode.workspace.getConfiguration("editor");
    void editorConfig.update("inlineSuggest.enabled", true, vscode.ConfigurationTarget.Global);

    const config = vscode.workspace.getConfiguration("mlxCompletion");
    const modelPath = config.get<string>("modelPath") || "";
    const quantization = config.get<string>("quantization") || "4bit";
    const debounceMs = Math.max(20, config.get<number>("debounceMs") || 40);
    const maxTokens = config.get<number>("maxTokens") || 32;
    const contextLinesBefore = config.get<number>("contextLinesBefore") || 60;
    const contextLinesAfter = config.get<number>("contextLinesAfter") || 15;

    outputChannel.appendLine(
        `[MLX] model=${modelPath || "(default)"} quant=${quantization} ` +
            `debounce=${debounceMs}ms maxTok=${maxTokens} ` +
            `ctx=${contextLinesBefore}/${contextLinesAfter}`,
    );

    const serverScript = path.join(context.extensionPath, "python-server", "server.py");
    if (!fs.existsSync(serverScript)) {
        vscode.window.showErrorMessage(
            `MLX Code Completion: Server script not found at ${serverScript}.`,
        );
        return;
    }

    backend = new BackendIPC(
        serverScript,
        modelPath,
        quantization,
        maxTokens,
        config.get<number>("temperature") || 0.0,
        outputChannel,
    );

    let backendReady = false;
    let triggerTimer: ReturnType<typeof setTimeout> | null = null;

    const scheduleInlineTrigger = (delay = debounceMs) => {
        if (triggerTimer !== null) {
            clearTimeout(triggerTimer);
        }
        triggerTimer = setTimeout(() => {
            triggerTimer = null;
            if (!backendReady) {
                return;
            }
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return;
            }
            const scheme = editor.document.uri.scheme;
            if (scheme !== "file" && scheme !== "untitled") {
                return;
            }
            void vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
        }, delay);
    };

    backend
        .start()
        .then(() => {
            backendReady = true;
            outputChannel!.appendLine("[MLX] Backend ready");
            vscode.window.showInformationMessage("MLX Code Completion: ready.");
            scheduleInlineTrigger(0);
        })
        .catch((err: Error) => {
            outputChannel!.appendLine(`[MLX] Backend failed: ${err.message}`);
            vscode.window.showErrorMessage(
                `MLX Code Completion: Failed to start backend: ${err.message}`,
            );
        });

    const provider = new CompletionProvider(
        backend,
        debounceMs,
        maxTokens,
        outputChannel,
        contextLinesBefore,
        contextLinesAfter,
    );

    const reg = vscode.languages.registerInlineCompletionItemProvider(
        [{ scheme: "file" }, { scheme: "untitled" }],
        provider,
    );
    context.subscriptions.push(reg);
    context.subscriptions.push({ dispose: () => provider.dispose() });

    // Trigger ghost text after typing settles (VS Code often won't auto-invoke).
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument((e) => {
            if (e.contentChanges.length === 0) {
                return;
            }
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.document.uri.toString() !== e.document.uri.toString()) {
                return;
            }
            const scheme = e.document.uri.scheme;
            if (scheme !== "file" && scheme !== "untitled") {
                return;
            }
            scheduleInlineTrigger(debounceMs);
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand("mlxCompletion.trigger", () => {
            scheduleInlineTrigger(0);
        }),
    );

    context.subscriptions.push({
        dispose: () => {
            if (triggerTimer !== null) {
                clearTimeout(triggerTimer);
            }
        },
    });
}

export function deactivate(): void {
    backend?.stop();
    backend = null;
    outputChannel?.dispose();
    outputChannel = null;
}
