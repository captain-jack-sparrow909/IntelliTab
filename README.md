# IntelliTab

Low-latency AI code completion for VS Code powered by a local MLX model on Apple Silicon.

## Architecture

```
VS Code Extension (TypeScript)
    ── stdin/stdout pipes (length-prefixed JSON) ──
Python MLX Server (persistent process)
    ── Metal (GPU) ──
   Qwen2.5-Coder-3B-Instruct-MLX-4bit
```

No REST server, no Ollama, no OpenAI-compatible API. Just native IPC.

## Features

- **Fill-in-the-Middle (FIM)** prompting — sends `<fim_prefix>{before}<fim_suffix>{after}<fim_middle>` for precise completions
- **Streaming ghost text** — completions appear token-by-token as inline suggestions (like GitHub Copilot), not a dropdown list
- **Persistent model** — loaded once on extension activation, never unloaded
- **Debounce** (40–200ms configurable, default 50ms) to avoid spamming the model
- **Cancellation** — in-flight completions are cancelled when you type again
- **150-line context window** — only the relevant surrounding code is sent (150 lines before cursor, 35 after)

## Performance

| Metric | Value |
|--------|-------|
| First token latency | ~210ms |
| Model size | ~2 GB RAM (4-bit quantized) |
| Max tokens per completion | 64 (configurable) |

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with MLX support
- **Python 3.10+** on PATH
- **~4–8 GB RAM** free (model uses ~2 GB for 4-bit)

## Setup

### 1. Install Python dependencies

```bash
pip install -r python-server/requirements.txt
```

### 2. Download the model

The model should be downloaded at `~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit`.

If you need to re-download:

```bash
huggingface-cli download lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit \
    --local-dir ~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit
```

Or via Python:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit', local_dir='~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit')"
```

### 3. Configure the extension

Open VS Code Settings (`Cmd+,`) and search for "MLX Code Completion". Set:

| Setting | Default | Description |
|---------|---------|-------------|
| `mlxCompletion.modelPath` | `""` | Path to MLX model directory (auto-detects quantization from config.json) |
| `mlxCompletion.quantization` | `4bit` | Quantization level (`4bit`, `6bit`, `8bit`, `BF16`) — only applied if model isn't already quantized |
| `mlxCompletion.debounceMs` | `50` | Debounce delay (40–200 ms) |
| `mlxCompletion.maxTokens` | `64` | Maximum tokens to generate per completion |
| `mlxCompletion.temperature` | `0.0` | Generation temperature (0.0 = deterministic) |
| `mlxCompletion.contextLinesBefore` | `150` | Lines before cursor to include |
| `mlxCompletion.contextLinesAfter` | `35` | Lines after cursor to include |

Or set in `settings.json`:

```json
{
    "mlxCompletion.modelPath": "~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit",
    "mlxCompletion.quantization": "4bit",
    "mlxCompletion.debounceMs": 50,
    "mlxCompletion.maxTokens": 64,
    "mlxCompletion.temperature": 0.0,
    "mlxCompletion.contextLinesBefore": 150,
    "mlxCompletion.contextLinesAfter": 35
}
```

### 4. Build and run

```bash
npm install
npm run compile
```

Then open the extension in VS Code:
1. Open this folder in VS Code
2. Press `F5` (Debug: Start Debugging)
3. A new VS Code window opens with the extension loaded
4. Open any code file and start typing — ghost text appears as you type

## How It Works

### VS Code Extension (`src/`)

1. **`extension.ts`** — Entry point. Spawns the Python process, registers the inline completion provider. On `activate()`, the model loads immediately. On `deactivate()`, the process is killed.
2. **`completion-provider.ts`** — Implements `InlineCompletionItemProvider`. Listens for cursor position changes (keyboard only), debounces input, extracts document context, and streams completions back as inline ghost text.
3. **`context-extractor.ts`** — Reads the document around the cursor: 150 lines before, 35 lines after. Returns a structured context object.
4. **`backend-ipc.ts`** — Manages the child process lifecycle and the length-prefixed JSON IPC protocol. Handles streaming token callbacks, cancellation, and health checks.
5. **`debounce.ts`** — Debounce utility with `cancel()` and `flush()` methods.

### Python MLX Server (`python-server/`)

1. **`server.py`** — Main server loop. Reads messages from stdin, dispatches to the model engine. Never unloads the model.
2. **`model.py`** — Loads Qwen2.5-Coder via MLX. Auto-detects whether the model is already quantized (by checking `config.json` for a `quantization` key) to avoid double-quantizing. Handles FIM prompt construction and both sync/streaming generation.
3. **`protocol.py`** — Length-prefixed JSON encoding/decoding: `[4 bytes big-endian uint32][JSON body]`.

### Protocol

```
VS Code ──→ [4-byte length][JSON: {type:"complete", id:1, context:{...}}] ──→ Python
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"def "}] ──→ VS Code
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"add"}] ──→ VS Code
...
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:""}] ──→ VS Code   (empty token = done)
```

## File Structure

```
ide-extension/
├── package.json               # Extension manifest with settings
├── tsconfig.json              # TypeScript config
├── .gitignore
├── .vscode/settings.json      # Dev settings
├── README.md
├── src/
│   ├── extension.ts           # Entry point — spawns backend, registers provider
│   ├── completion-provider.ts # InlineCompletionItemProvider (streaming ghost text)
│   ├── context-extractor.ts   # Extracts context window from document
│   ├── backend-ipc.ts         # Child process + length-prefixed JSON IPC
│   └── debounce.ts            # Debounce utility
└── python-server/
    ├── server.py              # Main server loop
    ├── model.py               # MLX model loading, FIM prompting, streaming
    ├── protocol.py            # Length-prefixed JSON encoding/decoding
    └── requirements.txt       # mlx, mlx-lm
```

## Troubleshooting

- **"Python 3 not found"** — Install Python 3.10+ and ensure it's on PATH (`which python3`)
- **"Model not found"** — Set `mlxCompletion.modelPath` to your downloaded model directory. Default is `~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit`
- **"Failed to start backend"** — Check that the model files exist at the configured path (look for `config.json` and `model.safetensors`)
- **High latency** — Try a lower quantization (4bit vs 8bit) or reduce `contextLinesBefore`
- **Model OOM** — Use 4bit quantization (~2 GB) instead of BF16 (~6 GB)
- **Model loads but produces no completions** — Make sure you're not at the very start of a document (position 0,0), and not inside a multi-line comment (`/* ... */`)

## Future Work

- [x] Speculative decoding with a 1B–2B draft model
- [ ] Support for JetBrains IDEs (PyCharm, etc.)
- [ ] Language-specific prompt templates
- [ ] Completion ranking / re-ranking
- [ ] Configurable trigger characters (currently triggers on `.`, `(`, `"`, `'`, `` ` ``, `[`, `{`, ` `, `:`, `=`, `,`)
- [ ] Inline completion widget with richer UI (accept/reject indicators)
