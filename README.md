# IntelliTab

Low-latency AI code completion for VS Code powered by a local MLX model on Apple Silicon.

## Architecture

```
VS Code Extension (TypeScript)
    ── stdin/stdout pipes (length-prefixed JSON) ──
Python MLX Server (persistent process)
    ── Metal (GPU) ──
   Qwen2.5-Coder-3B-4bit (base FIM — default)
```

No REST server, no Ollama, no OpenAI-compatible API. Just native IPC.

## Phase A design (speed + accuracy)

| Path | When | Model prompt | Context | Decode |
|------|------|--------------|---------|--------|
| **FIM** | Mid-line typing | `<\|fim_prefix\|>…<\|fim_suffix\|>…<\|fim_middle\|>` | Imports + enclosing scope + tight window | ≤32 tok, stop at `\n` |
| **Intent** | Comment / signature / empty body | FIM-framed request (base) or short chat (Instruct) | Richer window + imports/scope | 96–192 tok, multi-line |

Adaptive context always prefers **imports + enclosing function/class** over raw N lines, so prefill stays small without dropping the high-value tokens.

## Features

- **Copilot-style ghost text** — dimmed inline suggestions; accept with `Tab`.
- **True FIM completion** — Qwen2.5-Coder base special tokens (not chat-template mid-fill).
- **Intent / comment-to-code** — comments, bare signatures, empty function bodies.
- **Adaptive structured context** — imports + scope + mode-scaled window.
- **Streaming + progressive paint** — first tokens show ASAP; stale jobs cancel on type.
- **Persistent model** — loaded once, stays warm.
- **Dual policy** — fast single-line FIM vs richer multi-line intent.

## Performance

| Metric | Value |
|--------|-------|
| Default model | Qwen2.5-Coder-**3B base** 4-bit (~2 GB) |
| First token target | ~150–250 ms mid-line (hardware-dependent) |
| Max tokens | 32 FIM / 96–192 intent |
| Context | Adaptive (imports + scope; caps 80/20 configurable) |

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with MLX support
- **Python 3.10+** on PATH
- **~4 GB RAM** free for the default 3B model (~6–8 GB if you switch to 7B)
- The `code` CLI on PATH for F5 Extension Development Host

## Setup

### 1. Install Python dependencies

```bash
pip install -r python-server/requirements.txt
```

### 2. Download the default model (3B base FIM)

```bash
python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download('mlx-community/Qwen2.5-Coder-3B-4bit', local_dir=str(Path.home()/'.mlx-models'/'Qwen2.5-Coder-3B-4bit'))"
```

Fallbacks (auto-detected if the 3B base is missing):

```bash
# Instruct 3B (still has FIM tokens; slightly different prompt path)
python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download('lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit', local_dir=str(Path.home()/'.mlx-models'/'Qwen2.5-Coder-3B-Instruct-MLX-4bit'))"

# Heavier / higher quality (set mlxCompletion.modelPath)
python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download('lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit', local_dir=str(Path.home()/'.mlx-models'/'Qwen2.5-Coder-7B-Instruct-MLX-4bit'))"
```

### 3. Configure the extension

Open VS Code Settings (`Cmd+,`) and search for "MLX Code Completion". Set:

| Setting | Default | Description |
|---------|---------|-------------|
| `mlxCompletion.modelPath` | `""` | Empty = auto (`~/.mlx-models/Qwen2.5-Coder-3B-4bit` preferred). |
| `mlxCompletion.quantization` | `4bit` | Only applied if checkpoint is not already quantized. |
| `mlxCompletion.debounceMs` | `40` | Debounce delay (20–200 ms). |
| `mlxCompletion.maxTokens` | `32` | FIM max tokens (intent uses 96–192). |
| `mlxCompletion.temperature` | `0.0` | Greedy decode for stable completions. |
| `mlxCompletion.contextLinesBefore` | `80` | **Upper bound** for adaptive context. |
| `mlxCompletion.contextLinesAfter` | `20` | **Upper bound** for adaptive context. |

### 4. Build and run

```bash
npm install
npm run compile
```

Then open the extension in VS Code:
1. Open this folder in VS Code.
2. Press `F5` (Debug: Start Debugging). A new **Extension Development Host** window opens with the extension loaded and the project folder open.
3. In the host window, open a code file and start typing. Ghost text appears at the cursor; accept with `Tab`.
4. For comment-to-code: write a comment describing a function, press Enter to a new line, then type to get the implementation as ghost text.

> **Tip:** Ghost text requires `editor.inlineSuggest.enabled` (the extension turns this on at activation). If nothing appears, open **Output → "MLX Code Completion"** for logs, or run **Trigger Inline Suggestion** from the command palette.

## How It Works

### VS Code Extension (`src/`)

1. **`extension.ts`** — Spawns the Python backend, registers `InlineCompletionItemProvider`, triggers ghost text on typing, resolves the Phase A model path.
2. **`completion-provider.ts`** — Ghost text provider with dual policy (FIM vs intent), progressive streaming paint, cancel-on-type, light quality filters.
3. **`context-extractor.ts`** — Adaptive context: imports + enclosing scope + mode-scaled window; `detectIntent()` for comments, signatures, empty bodies.
4. **`backend-ipc.ts`** — Child process + length-prefixed JSON; cancels in-flight work when a new request starts.
5. **`debounce.ts`** — Debounce utility.

### Python MLX Server (`python-server/`)

1. **`server.py`** — Reader-thread IPC loop (cancel mid-generation), dual-policy dispatch, TTFT metrics.
2. **`model.py`** — Loads Qwen2.5-Coder (prefers **3B base FIM**). Builds native FIM prompts for mid-line; intent framing for comment→code. Cooperative cancel + newline early-stop.
3. **`protocol.py`** — Length-prefixed JSON: `[4 bytes big-endian uint32][JSON body]`.

### Protocol

```
VS Code ──→ [4-byte length][JSON: {type:"complete", id:1, context:{before, after, language, intent?}}] ──→ Python
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"a "}] ──→ VS Code
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"+ "}] ──→ VS Code
...
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:""}] ──→ VS Code   (empty token = end of stream)
```

`mode` is `"fim"` or `"intent"`. The `intent` field is set only for comment/signature/empty-body requests.

## File Structure

```
ide-extension/
├── package.json               # Extension manifest with settings
├── tsconfig.json              # TypeScript config
├── .gitignore
├── .vscode/
│   ├── settings.json          # Dev settings (enables inline + quick suggestions)
│   ├── launch.json            # Debug launch (uses the `code` binary)
│   └── tasks.json             # npm: compile task
├── README.md
├── src/
│   ├── extension.ts           # Entry point — spawns backend, registers provider
│   ├── completion-provider.ts # Ghost text + dual policy (FIM / intent)
│   ├── context-extractor.ts   # Adaptive imports + scope + intent detect
│   ├── backend-ipc.ts         # IPC + cancel-in-flight
│   └── debounce.ts            # Debounce utility
└── python-server/
    ├── server.py              # Cancel-aware IPC loop, dual policy
    ├── model.py               # FIM-first MLX engine (3B base default)
    ├── protocol.py            # Length-prefixed JSON
    └── requirements.txt       # mlx, mlx-lm
```

## Troubleshooting

- **F5 doesn't open the Extension Development Host** — the `code` binary must be on PATH. Run **Shell Command: Install 'code' command in PATH** from the command palette, or set `runtime` in `.vscode/launch.json` to the absolute path of `Visual Studio Code.app/Contents/Resources/app/bin/code`.
- **"Python 3 not found"** — Install Python 3.10+ and ensure it's on PATH (`which python3`).
- **"Model not found"** — Download the default 3B base model (see Setup), or set `mlxCompletion.modelPath`. Fallbacks: Instruct 3B, then 7B under `~/.mlx-models/`.
- **"Failed to start backend"** — Check that the model dir has `config.json` and weight files; look at Output → "MLX Code Completion".
- **No ghost text appears** — Output panel should show `Backend ready` and `[Provider] mode=fim|intent …`. Ensure `editor.inlineSuggest.enabled` is true, or run **MLX: Trigger Inline Completion**.
- **Wrong language generated from a comment** — The intent prompt uses the active file's language id; make sure the file has the correct language (e.g. a `.js` file is detected as JavaScript).
- **Completion echoes what I already typed** — The cleanup strips a leading prefix that duplicates the current line; if it still echoes, the model is mid-identifier — type a space or finish the token and it will complete the rest.
- **High latency / OOM** — Use the 3B model (`mlxCompletion.modelPath` → 3B path) or 4-bit quantization.

## Future Work

- [x] Inline ghost-text mode (`InlineCompletionItemProvider`)
- [x] Phase A: FIM base 3B + adaptive context + dual policy
- [ ] Phase B: tighter dual decode policies / accept-rate metrics
- [ ] Phase C: Prefix / token caching
- [ ] Phase D: Speculative decoding (draft 1.5B + target 3B/7B)
- [ ] Phase E: Optional 7B only for low-confidence intent
- [ ] Support for JetBrains IDEs (PyCharm, etc.)
- [ ] Richer UI (accept/reject indicators, partial acceptance)
