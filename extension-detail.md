VS Code
      │
      ▼
Extension
      │
      ▼
MLX Python API
      │
      ▼
Qwen3-Coder 4B

No REST server.

No Ollama.

No OpenAI-compatible API.

No JSON serialization.

Just:

VS Code
↓

Native IPC

↓

Python MLX process

↓

Metal

That alone removes a surprising amount of latency.

Even better

I'd keep the model always loaded.

launch VS Code

↓

load model once

↓

keep KV cache alive

↓

serve completions

Never unload it.

Never recreate the tokenizer.

Never recreate the prompt template.

Use Fill-in-the-Middle (FIM)

Most code models support FIM.

Instead of prompting:

def add(a, b):

you send:

<PRE>

def add(

<SUF>

    return a + b

<MID>

This is dramatically better for IDE completion than standard left-to-right prompting.

Many generic extensions don't take full advantage of this.

Keep the context tiny

Don't send:

entire file
entire workspace
every open tab

Send only:

100–200 lines before the cursor
20–50 lines after the cursor
current imports

That's usually enough.

Use speculative decoding

This is one of the biggest improvements available.

Have:

a tiny 1B–2B draft model
an 8B target model

The small model predicts ahead, and the larger model verifies. On compatible implementations, this can significantly increase effective throughput.

My dream stack for Apple Silicon
VS Code Extension (TypeScript)

↓

Native IPC

↓

Python MLX Server

↓

Qwen3-Coder 4B

Features:

FIM prompting
streaming after the first token
64-token maximum generation
150-line context window
speculative decoding (if supported)
persistent KV cache
debounce around 40–80 ms
cancellation as soon as the user types again

That should feel very close to "instant."
