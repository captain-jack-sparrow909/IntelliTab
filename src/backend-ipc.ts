/**
 * Backend IPC — length-prefixed JSON over stdin/stdout pipes.
 *
 * Spawns the Python MLX server as a child process and provides
 * a typed API for sending requests and receiving responses/streaming tokens.
 *
 * Protocol: each message is [4 bytes big-endian uint32 length][JSON body].
 */

import { spawn, ChildProcess } from "child_process";
import * as vscode from "vscode";
import * as fs from "fs";

// --- Message types ---

export interface CompleteRequest {
    type: "complete";
    id: number;
    context: {
        before: string;
        after: string;
        language: string;
     };
    maxTokens?: number;
    streaming?: boolean;
}

export interface CompleteResponse {
    type: "complete";
    id: number;
    completion: string;
}

export interface StreamToken {
    type: "stream";
    id: number;
    token: string;
}

export interface Cancelled {
    type: "cancelled";
    id: number;
}

export interface ErrorMsg {
    type: "error";
    id?: number;
    error: string;
}

export interface ReadyMsg {
    type: "ready";
}

export interface PongMsg {
    type: "pong";
    id?: number;
}

export type ServerMessage =
     | CompleteResponse
     | StreamToken
     | Cancelled
     | ErrorMsg
     | ReadyMsg
     | PongMsg;

export type TokenCallback = (token: string) => void;

// --- Length-prefixed protocol ---

/**
 * Encode a JSON object into a length-prefixed message.
 * [4 bytes big-endian uint32][JSON body]
 */
export function encodeMessage(data: Record<string, unknown>): Buffer {
    const body = JSON.stringify(data);
    const bodyBytes = Buffer.from(body, "utf-8");
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32BE(bodyBytes.length, 0);
    return Buffer.concat([prefix, bodyBytes]);
}

// --- Backend IPC Manager ---

export class BackendIPC {
    private process: ChildProcess | null = null;
    private nextId = 1;
    private pendingRequests = new Map<number, {
        resolve: (completion: string) => void;
        reject: (err: Error) => void;
        token: vscode.CancellationToken;
     }>();
    private streamTokens = new Map<number, TokenCallback>();
    private buffer = Buffer.alloc(0);
    private readableClosed = false;
    private resolveReady: (() => void) | null = null;
    private readyPromise: Promise<void> | null = null;
    private outputChannel: vscode.OutputChannel | null;

    constructor(
        private serverScript: string,
        private modelPath: string,
        private quantization: string,
        private maxTokens: number,
        private temperature: number,
        outputChannel: vscode.OutputChannel | null = null,
     ) {
        this.outputChannel = outputChannel;
     }

    private log(msg: string): void {
        this.outputChannel?.appendLine(msg);
     }

     /** Start the backend process and wait for it to be ready. */
    async start(): Promise<void> {
        this.log(`[IPC] Spawning server: ${this.serverScript}`);
        const pythonPath = this.findPython();
        this.log(`[IPC] Python: ${pythonPath}`);

        const args = ["-u", this.serverScript, "--streaming"];
        if (this.modelPath) {
            args.push("--model", this.modelPath);
        }
        if (this.quantization) {
            args.push("--quantization", this.quantization);
        }
        if (this.maxTokens) {
            args.push("--max-tokens", String(this.maxTokens));
        }
        if (this.temperature !== undefined) {
            args.push("--temperature", String(this.temperature));
        }

        this.log(`[IPC] Spawn args: ${args.join(" ")}`);
        this.process = spawn(pythonPath, args, {
            stdio: ["pipe", "pipe", "pipe"],
            env: {
                ...process.env,
                PYTHONUNBUFFERED: "1",
            },
        });

        const proc = this.process!;
        if (proc.stdout) {
            proc.stdout.on("data", (data: Buffer) => this.handleStdout(data));
        }
        if (proc.stderr) {
            proc.stderr.on("data", (data: Buffer) => {
                const line = data.toString().trim();
                this.log(`[IPC] Server: ${line}`);
            });
        }
        this.process.on("exit", (code) => {
            this.log(`[IPC] Process exited with code ${code}`);
            this.readableClosed = true;
            for (const [id, pending] of this.pendingRequests) {
                pending.reject(new Error(`Backend process exited (code ${code})`));
                this.pendingRequests.delete(id);
            }
        });
        this.process.on("error", (err) => {
            this.log(`[IPC] Process error: ${err.message}`);
            this.readableClosed = true;
            for (const [id, pending] of this.pendingRequests) {
                pending.reject(err);
                this.pendingRequests.delete(id);
            }
        });

        // Wait for ready signal
        this.readyPromise = new Promise((resolve) => {
            this.resolveReady = resolve;
        });
        await this.readyPromise;
        this.log("[IPC] Backend is ready");
     }

    private findPython(): string {
        const candidates = ["python3", "python"];
        for (const candidate of candidates) {
            try {
                const { execSync } = require("child_process");
                execSync(`${candidate} --version`, { stdio: "ignore" });
                return candidate;
             } catch {
                try {
                    const which = require("child_process").execSync(
                         `which ${candidate}`, { stdio: "ignore" }
                     ).toString().trim();
                    if (which) return which;
                 } catch {
                     // skip
                 }
             }
         }
        const commonPaths = [
             "/opt/homebrew/bin/python3",
             "/usr/bin/python3",
             "/usr/bin/python",
         ];
        for (const p of commonPaths) {
            if (fs.existsSync(p)) return p;
         }
        throw new Error(
             "Python 3 not found. Please install Python 3.10+ and ensure it's on PATH."
         );
     }

    private handleStdout(data: Buffer): void {
        if (this.readableClosed) return;
        this.buffer = Buffer.concat([this.buffer, data]);
        this.parseBuffer();
     }

    private parseBuffer(): void {
        while (this.buffer.length >= 4) {
             // Read length prefix
            const msgLength = this.buffer.readUInt32BE(0);
             // Need at least 4 (prefix) + msgLength (body)
            const totalNeeded = 4 + msgLength;
            if (this.buffer.length < totalNeeded) {
                break;
             }
             // Extract the message body
            const body = this.buffer.slice(4, totalNeeded);
            this.buffer = this.buffer.slice(totalNeeded);
             // Parse and dispatch
            try {
                const msg = JSON.parse(body.toString("utf-8")) as ServerMessage;
                this.dispatchMessage(msg);
             } catch {
                this.log("[IPC] Failed to parse message");
             }
         }
     }

    private dispatchMessage(msg: ServerMessage): void {
        switch (msg.type) {
            case "ready":
                if (this.resolveReady) {
                    this.resolveReady();
                    this.resolveReady = null;
                 }
                break;

            case "stream": {
                const streamMsg = msg as StreamToken;
                const callback = this.streamTokens.get(streamMsg.id);
                // Empty token signals end of stream
                if (streamMsg.token === "") {
                    this.streamTokens.delete(streamMsg.id);
                    const pending = this.pendingRequests.get(streamMsg.id);
                    if (pending) {
                        this.pendingRequests.delete(streamMsg.id);
                        pending.resolve("");
                    }
                    break;
                }
                if (callback) {
                    callback(streamMsg.token);
                }
                break;
             }

            case "complete": {
                const completeMsg = msg as CompleteResponse;
                const pending = this.pendingRequests.get(completeMsg.id);
                if (pending) {
                    this.pendingRequests.delete(completeMsg.id);
                    this.streamTokens.delete(completeMsg.id);
                    pending.resolve(completeMsg.completion);
                }
                break;
             }

            case "cancelled": {
                 // Backend acknowledged cancellation — no action needed
                break;
             }

            case "error": {
                const errorMsg = msg as ErrorMsg;
                const pending = this.pendingRequests.get(errorMsg.id ?? -1);
                if (pending) {
                    this.pendingRequests.delete(errorMsg.id ?? -1);
                    this.streamTokens.delete(errorMsg.id ?? -1);
                    pending.reject(new Error(errorMsg.error));
                }
                break;
             }

            case "pong":
                 // Handled by ping() via a different mechanism
                break;
         }
     }

     /** Send a completion request to the backend. */
    complete(
        context: { before: string; after: string; language: string },
        token: vscode.CancellationToken,
        onToken?: TokenCallback,
    ): Promise<string> {
        if (!this.process?.stdin) {
            throw new Error("Backend process not started");
        }

        const id = this.nextId++;
        const request: CompleteRequest = {
            type: "complete",
            id,
            context,
            maxTokens: this.maxTokens,
            streaming: true,
        };

        return new Promise<string>((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject, token });

            if (onToken) {
                this.streamTokens.set(id, onToken);
            }

             // Listen for cancellation only if a token was provided. When no
             // token is given the generation runs to completion independently of
             // any single request's lifecycle (used by the inline provider, which
             // is cancelled aggressively by VS Code).
            if (token) {
                const onCancel = token.onCancellationRequested(() => {
                    onCancel.dispose();
                    this.send({ type: "cancel", id });
                    const pending = this.pendingRequests.get(id);
                    if (pending) {
                        pending.reject(new Error("Cancelled by user"));
                        this.pendingRequests.delete(id);
                    }
                    this.streamTokens.delete(id);
                });
            }

             // Send the request
            this.send(request as unknown as Record<string, unknown>);
        });
    }

    private send(msg: Record<string, unknown>): void {
        if (!this.process?.stdin) return;
        const encoded = encodeMessage(msg);
        this.process.stdin.write(encoded);
     }

     /** Health check ping. */
    ping(): Promise<boolean> {
        return new Promise((resolve) => {
            const id = this.nextId++;
            this.send({ type: "ping", id });
            const timer = setTimeout(() => {
                resolve(false);
            }, 1000);
             // One-shot: intercept pong in dispatchMessage
            const origDispatch = this.dispatchMessage.bind(this);
            const checkPong = (msg: ServerMessage) => {
                if (msg.type === "pong" && (msg as PongMsg).id === id) {
                    clearTimeout(timer);
                    resolve(true);
                } else {
                    origDispatch(msg);
                }
            };
             // Replace dispatchMessage temporarily
            this.dispatchMessage = checkPong;
            setTimeout(() => {
                 // Restore original dispatchMessage after timeout
                if (this.dispatchMessage === checkPong) {
                    this.dispatchMessage = origDispatch;
                }
            }, 1000);
        });
    }

     /** Stop the backend process. */
    stop(): void {
        this.process?.kill();
        this.process = null;
        for (const [, pending] of this.pendingRequests) {
            pending.reject(new Error("Backend stopped"));
        }
        this.pendingRequests.clear();
        this.streamTokens.clear();
        this.buffer = Buffer.alloc(0);
        this.readableClosed = true;
     }

     /** Check if the backend is running. */
    isRunning(): boolean {
        return this.process !== null;
     }
}
