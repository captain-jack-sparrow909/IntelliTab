# IntelliTab

Low-latency AI code completion for VS Code powered by a local MLX model on Apple Silicon.

## Architecture

```
VS Code Extension (TypeScript)
    ── stdin/stdout pipes (length-prefixed JSON) ──
Python MLX Server (persistent process)
    ── Metal (GPU) ──
   Qwen2.5-Coder-7B-Instruct-MLX-4bit (default)
```

No REST server, no Ollama, no OpenAI-compatible API. Just native IPC.

## Features

- **Copilot-style ghost text** — completions appear as dimmed inline suggestions at the cursor as you type. Accept with `Tab` (or the configured inline-suggest accept key).
- **Intent / comment-to-code** — write a comment like `// write a function that calculates factorial` and the model generates the full implementation, in the file's language and matching the surrounding style.
- **Instruct-based continuation** — uses the model's chat/instruct template with the code before *and* after the cursor, so it fills in the middle correctly (the base model has no true FIM training).
- **Streaming** — tokens stream from the model and the suggestion updates as it generates.
- **Persistent model** — loaded once on extension activation, never unloaded.
- **Debounce** (40–200ms configurable, default 50ms) to avoid spamming the model.
- **150-line context window** — only the relevant surrounding code is sent (150 lines before cursor, 35 after).

## Performance

| Metric | Value |
|--------|-------|
| First token latency | ~210ms (3B) / ~300–400ms (7B) — target; depends on context size |
| Model size | ~2 GB RAM (3B, 4-bit) / ~4.3 GB RAM (7B, 4-bit) |
| Max tokens per completion | 32 default (early-stop on newline for normal mode; ~128 for intent) |
| Context window (default) | 60 lines before / 15 after (smaller = faster prefill) |

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with MLX support
- **Python 3.10+** on PATH
- **~6–8 GB RAM** free for the default 7B model (4-bit); ~4 GB for the 3B model
- The `code` CLI on PATH (run **Shell Command: Install 'code' command in PATH** from the VS Code command palette, or the F5 debug launch may fail to open the Extension Development Host)

## Setup

### 1. Install Python dependencies

```bash
pip install -r python-server/requirements.txt
```

### 2. Download the model

The default model is **Qwen2.5-Coder-7B-Instruct-MLX-4bit** and should be at
`~/.mlx-models/Qwen2.5-Coder-7B-Instruct-MLX-4bit`. If it is missing, download it:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit', local_dir='~/.mlx-models/Qwen2.5-Coder-7B-Instruct-MLX-4bit')"
```

For a lighter/faster option, the 3B model also works (set `mlxCompletion.modelPath`
accordingly):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit', local_dir='~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit')"
```

### 3. Configure the extension

Open VS Code Settings (`Cmd+,`) and search for "MLX Code Completion". Set:

| Setting | Default | Description |
|---------|---------|-------------|
| `mlxCompletion.modelPath` | `""` | Path to MLX model directory. Empty = the built-in default (7B). |
| `mlxCompletion.quantization` | `4bit` | Quantization level (`4bit`, `8bit`, `BF16`) — only applied if the model isn't already quantized. |
| `mlxCompletion.debounceMs` | `40` | Debounce delay (20–200 ms). Lower = snappier. |
| `mlxCompletion.maxTokens` | `32` | Max tokens for normal completion (intent uses more). |
| `mlxCompletion.temperature` | `0.0` | Generation temperature (0.0 = deterministic). |
| `mlxCompletion.contextLinesBefore` | `60` | Lines before cursor (lower = faster). |
| `mlxCompletion.contextLinesAfter` | `15` | Lines after cursor (lower = faster). |

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

1. **`extension.ts`** — Entry point. Spawns the Python process, registers the completion provider (`CompletionItemProvider`) with trigger characters, and force-enables inline/quick suggestions. On `activate()` the model loads immediately; on `deactivate()` the process is killed.
2. **`completion-provider.ts`** — Implements `InlineCompletionItemProvider` (ghost text). Detects Copilot-style *intent* (a comment or function signature at/above the cursor), extracts document context, and returns an `InlineCompletionItem` with correct insert range and indentation. Caches completions per context and re-triggers ghost text when VS Code cancels a slow generation.
3. **`context-extractor.ts`** — Reads the document around the cursor (150 lines before, 35 after). Also implements `detectIntent()` for comment/signature → code, and strips a leading bracket from the suffix so the model completes a body instead of echoing a signature.
4. **`backend-ipc.ts`** — Manages the child process lifecycle and the length-prefixed JSON IPC protocol. Handles streaming token callbacks and resolves the request when the stream ends.
5. **`debounce.ts`** — Debounce utility with `cancel()` and `flush()` methods.

### Python MLX Server (`python-server/`)

1. **`server.py`** — Main server loop. Reads messages from stdin, dispatches to the model engine. Never unloads the model. Passes `--model`, `--quantization`, `--max-tokens`, and `--temperature` from the extension settings.
2. **`model.py`** — Loads Qwen2.5-Coder via MLX. Builds either an **instruct continuation prompt** (code before + after the cursor, asking the model to output only the inserted code) or an **intent prompt** (comment/description → full implementation in the file's language, matching surrounding style). Both use the model's chat template. Auto-detects whether the model is already quantized.
3. **`protocol.py`** — Length-prefixed JSON encoding/decoding: `[4 bytes big-endian uint32][JSON body]`.

### Protocol

```
VS Code ──→ [4-byte length][JSON: {type:"complete", id:1, context:{before, after, language, intent?}}] ──→ Python
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"a "}] ──→ VS Code
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"+ "}] ──→ VS Code
...
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:""}] ──→ VS Code   (empty token = end of stream)
```

The `intent` field is present only for comment-to-code requests.

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
│   ├── completion-provider.ts # InlineCompletionItemProvider (ghost text) + intent mode
│   ├── context-extractor.ts   # Extracts context window + detects intent
│   ├── backend-ipc.ts         # Child process + length-prefixed JSON IPC
│   └── debounce.ts            # Debounce utility
└── python-server/
    ├── server.py              # Main server loop
    ├── model.py               # MLX model loading, instruct/intent prompting, streaming
    ├── protocol.py            # Length-prefixed JSON encoding/decoding
    └── requirements.txt       # mlx, mlx-lm
```

## Troubleshooting

- **F5 doesn't open the Extension Development Host** — the `code` binary must be on PATH. Run **Shell Command: Install 'code' command in PATH** from the command palette, or set `runtime` in `.vscode/launch.json` to the absolute path of `Visual Studio Code.app/Contents/Resources/app/bin/code`.
- **"Python 3 not found"** — Install Python 3.10+ and ensure it's on PATH (`which python3`).
- **"Model not found"** — Set `mlxCompletion.modelPath` to your downloaded model directory, or download the default 7B model (see Setup).
- **"Failed to start backend"** — Check that the model files exist at the configured path (look for `config.json` and `model.safetensors`).
- **No ghost text appears** — Open the **Output** panel (`Cmd+Shift+U`), select **"MLX Code Completion"**, and check the logs. Confirm the backend printed "Model loaded successfully" and that `provideInlineCompletionItems called` appears when you type. Ensure `editor.inlineSuggest.enabled` is true, then try **Trigger Inline Suggestion** from the command palette.
- **Wrong language generated from a comment** — The intent prompt uses the active file's language id; make sure the file has the correct language (e.g. a `.js` file is detected as JavaScript).
- **Completion echoes what I already typed** — The cleanup strips a leading prefix that duplicates the current line; if it still echoes, the model is mid-identifier — type a space or finish the token and it will complete the rest.
- **High latency / OOM** — Use the 3B model (`mlxCompletion.modelPath` → 3B path) or 4-bit quantization.

## Future Work

- [ ] Speculative decoding with a 1B–2B draft model (faster first token)
- [ ] Prefix caching of the system + file-context prompt
- [ ] Completion ranking / re-ranking across multiple candidates
- [x] Inline ghost-text mode (`InlineCompletionItemProvider`)
- [ ] Support for JetBrains IDEs (PyCharm, etc.)
- [ ] Richer UI (accept/reject indicators, partial acceptance)
