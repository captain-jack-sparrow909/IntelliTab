/**
 * Backend IPC — length-prefixed JSON over stdin/stdout pipes.
 *
 * Optimized for low latency:
 * - Cancels the previous in-flight completion when a new one starts
 * - Streams tokens to the caller as they arrive
 * - Correct max_tokens field for the Python server
 */

import { spawn, ChildProcess } from "child_process";
import * as vscode from "vscode";
import * as fs from "fs";

export interface CompleteRequest {
    type: "complete";
    id: number;
    context: {
        before: string;
        after: string;
        language: string;
        intent?: string;
        mode?: "fim" | "intent";
        multiLine?: boolean;
    };
    max_tokens?: number;
    streaming?: boolean;
    stop_on_newline?: boolean;
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

export function encodeMessage(data: Record<string, unknown>): Buffer {
    const body = JSON.stringify(data);
    const bodyBytes = Buffer.from(body, "utf-8");
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32BE(bodyBytes.length, 0);
    return Buffer.concat([prefix, bodyBytes]);
}

export interface SpeculativeOptions {
    /** Enable speculative decoding (default true). */
    enabled?: boolean;
    /** Path to draft model; empty string disables; omit for auto-pick. */
    draftModelPath?: string;
    /** Draft tokens per verification step (1–8). */
    numDraftTokens?: number;
}

/** Phase E: dual-model routing options. */
export interface DualModelOptions {
    /** Enable fast mid-line model (default true). */
    enabled?: boolean;
    /** Path to fast model; omit for auto 3B; empty disables via enabled=false. */
    fastModelPath?: string;
}

export class BackendIPC {
    private process: ChildProcess | null = null;
    private nextId = 1;
    private pendingRequests = new Map<
        number,
        {
            resolve: (completion: string) => void;
            reject: (err: Error) => void;
        }
    >();
    private streamTokens = new Map<number, TokenCallback>();
    private buffer = Buffer.alloc(0);
    private readableClosed = false;
    private resolveReady: (() => void) | null = null;
    private readyPromise: Promise<void> | null = null;
    private outputChannel: vscode.OutputChannel | null;
    /** Most recent completion request id — cancelled when a newer one starts. */
    private activeCompleteId: number | null = null;
    private speculative: SpeculativeOptions;
    private dualModel: DualModelOptions;

    constructor(
        private serverScript: string,
        private modelPath: string,
        private quantization: string,
        private maxTokens: number,
        private temperature: number,
        outputChannel: vscode.OutputChannel | null = null,
        speculative: SpeculativeOptions = {},
        dualModel: DualModelOptions = {},
    ) {
        this.outputChannel = outputChannel;
        this.speculative = speculative;
        this.dualModel = dualModel;
    }

    private log(msg: string): void {
        this.outputChannel?.appendLine(msg);
    }

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

        // Phase D: speculative decoding
        if (this.speculative.enabled === false) {
            args.push("--no-speculative");
        } else if (
            this.speculative.draftModelPath !== undefined &&
            this.speculative.draftModelPath !== ""
        ) {
            args.push("--draft-model", this.speculative.draftModelPath);
        }
        // Omit --draft-model entirely for auto-pick when enabled.
        if (
            this.speculative.numDraftTokens !== undefined &&
            this.speculative.numDraftTokens > 0
        ) {
            args.push("--num-draft-tokens", String(this.speculative.numDraftTokens));
        }

        // Phase E: dual-model routing
        if (this.dualModel.enabled === false) {
            args.push("--no-dual-model");
        } else if (
            this.dualModel.fastModelPath !== undefined &&
            this.dualModel.fastModelPath !== ""
        ) {
            args.push("--fast-model", this.dualModel.fastModelPath);
        }

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
                if (line) {
                    this.log(`[IPC] ${line}`);
                }
            });
        }
        this.process.on("exit", (code) => {
            this.log(`[IPC] Process exited with code ${code}`);
            this.readableClosed = true;
            for (const [, pending] of this.pendingRequests) {
                pending.reject(new Error(`Backend process exited (code ${code})`));
            }
            this.pendingRequests.clear();
        });
        this.process.on("error", (err) => {
            this.log(`[IPC] Process error: ${err.message}`);
            this.readableClosed = true;
            for (const [, pending] of this.pendingRequests) {
                pending.reject(err);
            }
            this.pendingRequests.clear();
        });

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
                    const which = require("child_process")
                        .execSync(`which ${candidate}`, { stdio: "ignore" })
                        .toString()
                        .trim();
                    if (which) {
                        return which;
                    }
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
            if (fs.existsSync(p)) {
                return p;
            }
        }
        throw new Error(
            "Python 3 not found. Please install Python 3.10+ and ensure it's on PATH.",
        );
    }

    private handleStdout(data: Buffer): void {
        if (this.readableClosed) {
            return;
        }
        this.buffer = Buffer.concat([this.buffer, data]);
        this.parseBuffer();
    }

    private parseBuffer(): void {
        while (this.buffer.length >= 4) {
            const msgLength = this.buffer.readUInt32BE(0);
            const totalNeeded = 4 + msgLength;
            if (this.buffer.length < totalNeeded) {
                break;
            }
            const body = this.buffer.subarray(4, totalNeeded);
            this.buffer = this.buffer.subarray(totalNeeded);
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
                if (streamMsg.token === "") {
                    // End of stream
                    this.streamTokens.delete(streamMsg.id);
                    const pending = this.pendingRequests.get(streamMsg.id);
                    if (pending) {
                        this.pendingRequests.delete(streamMsg.id);
                        if (this.activeCompleteId === streamMsg.id) {
                            this.activeCompleteId = null;
                        }
                        pending.resolve("");
                    }
                    break;
                }
                const callback = this.streamTokens.get(streamMsg.id);
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
                    if (this.activeCompleteId === completeMsg.id) {
                        this.activeCompleteId = null;
                    }
                    pending.resolve(completeMsg.completion);
                }
                break;
            }

            case "cancelled":
                break;

            case "error": {
                const errorMsg = msg as ErrorMsg;
                const id = errorMsg.id ?? -1;
                const pending = this.pendingRequests.get(id);
                if (pending) {
                    this.pendingRequests.delete(id);
                    this.streamTokens.delete(id);
                    if (this.activeCompleteId === id) {
                        this.activeCompleteId = null;
                    }
                    pending.reject(new Error(errorMsg.error));
                }
                break;
            }

            case "pong":
                break;
        }
    }

    /**
     * Cancel any in-flight completion so the GPU can start the new one ASAP.
     */
    cancelActive(): void {
        if (this.activeCompleteId === null) {
            return;
        }
        const id = this.activeCompleteId;
        this.send({ type: "cancel", id });
        this.streamTokens.delete(id);
        const pending = this.pendingRequests.get(id);
        if (pending) {
            this.pendingRequests.delete(id);
            pending.resolve(""); // soft-cancel: resolve empty, caller uses partials
        }
        this.activeCompleteId = null;
    }

    complete(
        context: {
            before: string;
            after: string;
            language: string;
            intent?: string;
            mode?: "fim" | "intent";
            multiLine?: boolean;
        },
        onToken?: TokenCallback,
        options?: {
            maxTokens?: number;
            stopOnNewline?: boolean;
            /** Cancel in-flight job first (default true). Set false when joining. */
            cancelPrevious?: boolean;
        },
    ): Promise<string> {
        if (!this.process?.stdin) {
            throw new Error("Backend process not started");
        }

        // Only cancel when the caller is starting a *new* context.
        // Blind cancel-on-every-call caused thrash + multi-second delays.
        if (options?.cancelPrevious !== false) {
            this.cancelActive();
        }

        const id = this.nextId++;
        this.activeCompleteId = id;

        const request: CompleteRequest = {
            type: "complete",
            id,
            context,
            max_tokens: options?.maxTokens ?? this.maxTokens,
            streaming: true,
            stop_on_newline: options?.stopOnNewline ?? false,
        };

        return new Promise<string>((resolve, reject) => {
            this.pendingRequests.set(id, { resolve, reject });
            if (onToken) {
                this.streamTokens.set(id, onToken);
            }
            this.send(request as unknown as Record<string, unknown>);
        });
    }

    private send(msg: Record<string, unknown>): void {
        if (!this.process?.stdin) {
            return;
        }
        this.process.stdin.write(encodeMessage(msg));
    }

    ping(): Promise<boolean> {
        return new Promise((resolve) => {
            const id = this.nextId++;
            this.send({ type: "ping", id });
            const timer = setTimeout(() => resolve(false), 1000);
            const origDispatch = this.dispatchMessage.bind(this);
            const checkPong = (msg: ServerMessage) => {
                if (msg.type === "pong" && (msg as PongMsg).id === id) {
                    clearTimeout(timer);
                    this.dispatchMessage = origDispatch;
                    resolve(true);
                } else {
                    origDispatch(msg);
                }
            };
            this.dispatchMessage = checkPong;
            setTimeout(() => {
                if (this.dispatchMessage === checkPong) {
                    this.dispatchMessage = origDispatch;
                }
            }, 1000);
        });
    }

    stop(): void {
        this.cancelActive();
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

    isRunning(): boolean {
        return this.process !== null;
    }
}
