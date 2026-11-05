"""Shared helpers for the entropy-based thinking detector.

Every script talks to a local Ollama server through its OpenAI-compatible
endpoint, because plain ollama.chat() does not return token probabilities.
"""

import math

from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2:1b"
TOP_LOGPROBS = 10  # entropy is computed over the 10 most likely candidate tokens
WINDOW = 8         # tokens in the rolling average (the "window of observation")

# Verdict thresholds, calibrated on the labelled prompts in build_trial_bank.py.
CALM = 0.6   # late entropy below this: the model has settled
HIGH = 0.9   # overall entropy above this, and late not calm: it never settled
RISE = 0.25  # late above early by more than this: uncertainty is growing

# Ollama ignores the API key, but the OpenAI client requires one.
client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")


def entropy_bits(top_logprobs) -> float:
    """Return the Shannon entropy, in bits, of one token position."""
    probs = [math.exp(candidate.logprob) for candidate in top_logprobs]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def token_entropies(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 200,
    temperature: float | None = None,
) -> list[float]:
    """Generate a reply to *prompt* and return the entropy of every token."""
    options = {} if temperature is None else {"temperature": temperature}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        max_tokens=max_tokens,
        **options,
    )
    logprobs = response.choices[0].logprobs
    if not logprobs or not logprobs.content:
        raise RuntimeError(f"{model} returned no token probabilities.")
    return [entropy_bits(token.top_logprobs) for token in logprobs.content]


def rolling_mean(values: list[float], window: int = WINDOW) -> list[float]:
    """Smooth *values* with a trailing average over the last *window* points."""
    smoothed = []
    for i in range(len(values)):
        recent = values[max(0, i - window + 1): i + 1]
        smoothed.append(sum(recent) / len(recent))
    return smoothed


def summarize(entropies: list[float]) -> tuple[float, float, float]:
    """Return the mean entropy of the first third, the last third, and overall."""
    if not entropies:
        raise ValueError("Cannot summarize an empty entropy curve.")
    third = max(1, len(entropies) // 3)
    early = sum(entropies[:third]) / third
    late = sum(entropies[-third:]) / third
    overall = sum(entropies) / len(entropies)
    return early, late, overall


def verdict(early: float, late: float, overall: float) -> str:
    """Classify an entropy curve from its early, late, and overall averages."""
    if late < CALM:
        return "healthy (converged/calm)"
    if overall > HIGH:
        return "OVERTHINKING (stayed high)"
    if late > early + RISE:
        return "OVERTHINKING (rising)"
    return "unclear"