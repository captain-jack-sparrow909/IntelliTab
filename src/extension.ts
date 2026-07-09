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
    outputChannel.appendLine("[MLX] Extension activated (Phase A: FIM + adaptive context)");

    const editorConfig = vscode.workspace.getConfiguration("editor");
    void editorConfig.update("inlineSuggest.enabled", true, vscode.ConfigurationTarget.Global);

    const config = vscode.workspace.getConfiguration("mlxCompletion");
    const modelPath = resolveModelPath(config.get<string>("modelPath") || "");
    const quantization = config.get<string>("quantization") || "4bit";
    // 50ms is enough to coalesce keystrokes without feeling laggy.
    const debounceMs = Math.max(35, config.get<number>("debounceMs") || 50);
    const maxTokens = config.get<number>("maxTokens") || 32;
    // Upper bounds — adaptive extractor uses less for FIM, more for intent.
    const contextLinesBefore = config.get<number>("contextLinesBefore") || 60;
    const contextLinesAfter = config.get<number>("contextLinesAfter") || 15;

    outputChannel.appendLine(
        `[MLX] model=${modelPath || "(server default)"} quant=${quantization} ` +
            `debounce=${debounceMs}ms maxTok=${maxTokens} ` +
            `ctxCap=${contextLinesBefore}/${contextLinesAfter}`,
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

/**
 * Resolve model path: user setting → Phase A defaults on disk.
 * Empty string lets the Python server run its own fallback chain.
 */
function resolveModelPath(configured: string): string {
    const home = process.env.HOME || process.env.USERPROFILE || "";
    // Quality first: 7B Instruct (if present) > 3B base FIM > 3B Instruct.
    // 3B base alone often maps `sub` → `a + b` without enough signal; 7B is better.
    const candidates = [
        configured,
        path.join(home, ".mlx-models", "Qwen2.5-Coder-7B-Instruct-MLX-4bit"),
        path.join(home, ".mlx-models", "Qwen2.5-Coder-3B-4bit"),
        path.join(home, ".mlx-models", "Qwen2.5-Coder-3B-Instruct-MLX-4bit"),
    ].filter(Boolean);

    for (const c of candidates) {
        const expanded = c.startsWith("~")
            ? path.join(home, c.slice(1))
            : c;
        if (fs.existsSync(path.join(expanded, "config.json"))) {
            return expanded;
        }
    }
    return configured || path.join(home, ".mlx-models", "Qwen2.5-Coder-3B-4bit");
}
