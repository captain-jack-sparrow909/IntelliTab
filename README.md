# MLX Code Completion

Low-latency AI code completion for VS Code powered by a local MLX model on Apple Silicon.

## Architecture

```
VS Code Extension (TypeScript)
   ── stdin/stdout pipes (length-prefixed JSON) ──
Python MLX Server (persistent process)
   ── Metal (GPU) ──
   Qwen2.5-Coder-3B-Instruct-MLX
```

No REST server, no Ollama, no OpenAI-compatible API. Just native IPC.

## Features

- **Fill-in-the-Middle (FIM)** prompting for precise completions
- **Streaming** completions token-by-token for low perceived latency
- **Persistent model** — loaded once, never unloaded
- **Debounce** (40–80ms configurable) to avoid spamming the model
- **Cancellation** — in-flight completions are cancelled when you type again
- **150-line context window** — only the relevant surrounding code is sent

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with MLX support
- **Python 3.10+** on PATH
- **~4–8 GB RAM** free (depends on quantization)

## Setup

### 1. Install Python dependencies

```bash
pip install -r python-server/requirements.txt
```

### 2. Download the model

```bash
huggingface-cli download lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit --local-dir ~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit
```

Or use the Hugging Face API:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('lmstudio-community/Qwen2.5-Coder-3B-Instruct-MLX-4bit', local_dir='~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit')"
```

### 3. Configure the extension

Open VS Code Settings (`Cmd+,`) and search for "MLX Code Completion". Set:

| Setting | Value |
|---------|-------|
| `mlxCompletion.modelPath` | Path to the model directory (e.g., `~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit`) |
| `mlxCompletion.quantization` | `4bit` (default), `6bit`, `8bit`, or `BF16` |
| `mlxCompletion.debounceMs` | `50` (default, range 40–200) |
| `mlxCompletion.maxTokens` | `64` (default) |
| `mlxCompletion.temperature` | `0.0` (deterministic) |

Or set in `settings.json`:

```json
{
    "mlxCompletion.modelPath": "~/.mlx-models/Qwen2.5-Coder-3B-Instruct-MLX-4bit",
    "mlxCompletion.quantization": "4bit",
    "mlxCompletion.debounceMs": 50,
    "mlxCompletion.maxTokens": 64
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
4. Open any code file and start typing

## How It Works

### VS Code Extension (`src/`)

1. **`extension.ts`** — Entry point. Spawns the Python process, registers the completion provider.
2. **`completion-provider.ts`** — Listens for cursor changes, debounces input, sends requests to the backend.
3. **`context-extractor.ts`** — Reads the document around the cursor (150 lines before, 35 lines after).
4. **`backend-ipc.ts`** — Manages the child process and length-prefixed JSON protocol.
5. **`debounce.ts`** — Debounce utility for input events.

### Python MLX Server (`python-server/`)

1. **`server.py`** — Main server loop. Reads messages from stdin, dispatches to the model engine.
2. **`model.py`** — Loads Qwen2.5-Coder via MLX, handles FIM prompting and streaming generation.
3. **`protocol.py`** — Length-prefixed JSON encoding/decoding for IPC.

### Protocol

```
VS Code ──→ [4-byte length][JSON: {type:"complete", id:1, ...}] ──→ Python
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"def "}] ──→ VS Code
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:"add"}] ──→ VS Code
...
Python ──→ [4-byte length][JSON: {type:"stream", id:1, token:""}] ──→ VS Code  (done)
```

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `mlxCompletion.modelPath` | `""` | Path to MLX model directory |
| `mlxCompletion.quantization` | `4bit` | Quantization level |
| `mlxCompletion.debounceMs` | `50` | Debounce delay (40–200ms) |
| `mlxCompletion.maxTokens` | `64` | Max tokens per completion |
| `mlxCompletion.temperature` | `0.0` | Generation temperature |
| `mlxCompletion.contextLinesBefore` | `150` | Lines before cursor |
| `mlxCompletion.contextLinesAfter` | `35` | Lines after cursor |

## Troubleshooting

- **"Python 3 not found"** — Install Python 3.10+ and ensure it's on PATH (`which python3`)
- **"Model not found"** — Set `mlxCompletion.modelPath` to your downloaded model directory
- **High latency** — Try a lower quantization (Q4 vs Q8) or reduce `contextLinesBefore`
- **Model OOM** — Use a lower quantization level (4bit uses ~2GB vs ~6GB for BF16)

## Future Work

- [ ] Speculative decoding with a 1B–2B draft model
- [ ] Support for JetBrains IDEs (PyCharm, etc.)
- [ ] Inline completion widget (ghost text preview)
- [ ] Language-specific prompt templates
- [ ] Completion ranking / re-ranking
