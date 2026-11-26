
"""LLM answer generation via Ollama (local, fast) with transformers fallback.

Ollama runs quantized GGUF models on CPU efficiently (~9–15 tok/s on Intel).
The transformers backend is kept as a fallback for teammates with CUDA/MPS.

Ollama must be running: ``ollama serve``
Default model: llama3.2:latest  (3.2B Q4_K_M, already downloaded)
Override with the OLLAMA_MODEL env var or the OLLAMA_MODEL module constant.
"""

import gc
import os
import urllib.error
import urllib.request
import json
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ollama settings — override with environment variables if needed.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

# Transformers fallback model (used only when Ollama is unreachable).
_TRANSFORMERS_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Public name shown in the UI / orchestrator output.
MODEL_NAME = OLLAMA_MODEL

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def _ollama_generate(
    system: str,
    user: str,
    max_new_tokens: int,
    *,
    json_mode: bool = False,
) -> str:
    """Call the Ollama /api/chat endpoint and return the assistant reply."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0,       # deterministic — matches do_sample=False
        },
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    text = result.get("message", {}).get("content", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# Transformers fallback (lazy-loaded, cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_transformers_model():
    print(f"[llm] Ollama unavailable — loading {_TRANSFORMERS_MODEL_NAME} via transformers…")

    tokenizer = AutoTokenizer.from_pretrained(_TRANSFORMERS_MODEL_NAME)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            _TRANSFORMERS_MODEL_NAME,
            torch_dtype=torch.float16,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            _TRANSFORMERS_MODEL_NAME,
            torch_dtype=torch.float16,
        )
    else:
        device = torch.device("cpu")
        # float16 halves memory on CPU — still slow but uses ~6 GB vs 12 GB.
        model = AutoModelForCausalLM.from_pretrained(
            _TRANSFORMERS_MODEL_NAME,
            torch_dtype=torch.float16,
        )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[llm] Transformers model loaded on {device}.")
    return tokenizer, model, device


def _transformers_generate(
    system: str,
    user: str,
    max_new_tokens: int,
) -> str:
    tokenizer, model, device = _load_transformers_model()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Public API — unchanged signature, drop-in replacement
# ---------------------------------------------------------------------------

def generate_answer(
    prompt: str,
    max_new_tokens: int = 150,
    language: str = "en",
) -> str:
    """Generate an answer to *prompt*.

    Uses Ollama when it is running (fast); falls back to the transformers
    model otherwise (slow on CPU — start Ollama for best performance).
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")

    language = (language or "en").lower().strip()
    if language not in LANGUAGE_NAMES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: "
            f"{', '.join(LANGUAGE_NAMES)}"
        )

    language_name = LANGUAGE_NAMES[language]
    system_msg = (
        f"You are a factual assistant. Respond entirely in {language_name}. "
        "Preserve names, dates, numbers and factual details. "
        "Do not switch languages unnecessarily."
    )

    return _generate(system_msg, prompt, max_new_tokens)


def generate_claim_json(system: str, prompt: str, max_new_tokens: int) -> str:
    """Generate structured extraction metadata with the configured model."""
    return _generate(system, prompt, max_new_tokens, json_mode=True)


def _generate(system: str, prompt: str, max_new_tokens: int, *, json_mode: bool = False) -> str:
    if _ollama_available():
        answer = _ollama_generate(system, prompt.strip(), max_new_tokens, json_mode=json_mode)
    else:
        print(
            "[llm] WARNING: Ollama server not found at "
            f"{OLLAMA_BASE_URL}. Run `ollama serve` for fast generation. "
            "Falling back to transformers (slow on CPU)."
        )
        answer = _transformers_generate(system, prompt.strip(), max_new_tokens)

    if not answer:
        raise RuntimeError("The model returned an empty answer.")
    return answer


def unload_model() -> None:
    """Release any cached transformers model and free memory.

    Not needed when using Ollama (the server manages its own memory).
    """
    _load_transformers_model.cache_clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()
