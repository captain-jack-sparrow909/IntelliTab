"""
MLX model engine for code completion.

Loads Qwen2.5-Coder-3B-Instruct and provides:
- FIM (Fill-in-the-Middle) prompt construction
- Streaming text generation
- Persistent KV cache across requests
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler


# FIM tokens for Qwen2.5-Coder (uses the <|fim_*|> special tokens)
FIM_BEGIN = "<|fim_begin|>"
FIM_END = "<|fim_end|>"
FIM_PAD = "<|fim_pad|>"

# Mapping from quantization name to MLX model_config
QUANT_CONFIG = {
    "4bit": {"quant_group_size": 32, "bits": 4},
    "6bit": {"quant_group_size": 32, "bits": 6},
    "8bit": {"quant_group_size": 32, "bits": 8},
    "BF16": {},
    "IQ4_XS": {"quant_group_size": 32, "bits": 4},
}


def _is_already_quantized(model_path: str) -> bool:
    """Check if the model is already an MLX-quantized model.

    MLX quantized models have a config.json with a 'quantization' key,
    or a quantized.safetensors.index.json file.
    """
    p = Path(model_path)
    # Check for quantized safetensors index
    if p.joinpath("quantized.safetensors.index.json").exists():
        return True
    # Check config.json for quantization settings
    config_path = p.joinpath("config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if "quantization" in config:
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


class ModelEngine:
    """Manages the MLX model and generates completions."""

    def __init__(
        self,
        model_path: str,
        quantization: str = "4bit",
        max_tokens: int = 64,
        temperature: float = 0.0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature

        sys.stderr.write(f"[mlx] Loading model from: {model_path}\n")
        sys.stderr.flush()

        # Determine if we need to re-quantize
        model_config = {}
        if not _is_already_quantized(model_path):
            quant_cfg = QUANT_CONFIG.get(quantization, QUANT_CONFIG["4bit"])
            if quant_cfg:
                model_config["quantization"] = quant_cfg
                sys.stderr.write(f"[mlx] Applying quantization: {quantization}\n")
                sys.stderr.flush()

        # Load model and tokenizer (done once at startup)
        self.model, self.tokenizer = load(
            model_path,
            model_config=model_config if model_config else None,
        )

        sys.stderr.write(f"[mlx] Model loaded successfully\n")
        sys.stderr.flush()

        # Build sampler
        self.sampler = make_sampler(temp=temperature, top_p=1.0)

    def build_fim_prompt(self, before: str, after: str) -> str:
        """Build a code-continuation prompt for Qwen2.5-Coder.

        This model is an instruct model without true FIM training, so we use an
        instruct prompt that asks it to output only the code to insert at the
        cursor. Both the prefix (before cursor) and suffix (after cursor) are
        provided so it can fill in the middle.
        """
        system = (
            "You are a code completion engine. Output ONLY the code that should "
            "be inserted at the <cursor> position, continuing naturally from the "
            "code before it and consistent with the code after it. Do not repeat "
            "the existing code. No explanation, no markdown fences."
        )
        user = (
            "Complete the code by outputting only what goes at <cursor>.\n\n"
            "```javascript\n"
            f"{before}\n<cursor>\n{after}\n```\n\n"
            "Code to insert at <cursor>:"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def build_intent_prompt(self, intent: str, language: str = "javascript", context: str = "") -> str:
        """Build an instruct prompt for generating code from a comment/description.

        Used when the user writes a comment (e.g. "// fetch user data") and we
        want the model to generate the whole implementation, Copilot-style.
        `language` is the editor language id (e.g. "javascript") and `context`
        is the surrounding code so the generated style/indentation matches.
        """
        intent = intent.strip()
        language = (language or "javascript").strip()
        lang_name = {
            "js": "JavaScript", "javascript": "JavaScript", "ts": "TypeScript",
            "typescript": "TypeScript", "py": "Python", "python": "Python",
            "java": "Java", "go": "Go", "rust": "Rust", "cpp": "C++", "c": "C",
            "csharp": "C#", "rb": "Ruby", "php": "PHP", "sh": "Shell",
            "shellscript": "Shell", "html": "HTML", "css": "CSS",
        }.get(language.lower(), language)
        surrounding = (context or "").strip()
        system = (
            f"You are a code completion engine. Write code in {lang_name} only. "
            "Match the existing code style and indentation of the surrounding "
            "code. Respond with ONLY the code implementation, no explanation, "
            "no markdown fences."
        )
        user_parts = [f"Write {lang_name} code for the following request:\n{intent}"]
        if surrounding:
            user_parts.append(
                "The code should be placed in this file. Match its style and "
                "indentation:\n```\n" + surrounding + "\n```"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        msg_id: Optional[int] = None,
    ) -> str:
        """Generate a completion synchronously.

        Returns the full completion string.
        """
        max_tok = max_tokens or self.max_tokens

        try:
            result = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tok,
                sampler=self.sampler,

            )
            return result
        except Exception as e:
            sys.stderr.write(f"[mlx] Generation error: {e}\n")
            sys.stderr.flush()
            return ""

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        msg_id: Optional[int] = None,
    ):
        """Generate a completion with streaming tokens.

        Yields (token: str, is_final: bool) tuples.
        The last yield has is_final=True.
        """
        max_tok = max_tokens or self.max_tokens

        try:
            count = 0
            for item in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tok,
                sampler=self.sampler,

            ):
                token = item.text
                is_final = count >= max_tok - 1
                yield (token, is_final)
                count += 1
        except Exception as e:
            sys.stderr.write(f"[mlx] Stream generation error: {e}\n")
            sys.stderr.flush()
            yield ("", True)
