"""Check that the local model returns token probabilities.

The detector needs the probability of each generated token to compute
entropy. This script asks one question and prints every token with its
entropy. If no probabilities come back, the detector cannot work with
this model or server. The other scripts do not need this one.
"""

from common import DEFAULT_MODEL, TOP_LOGPROBS, client, entropy_bits

PROMPT = "What is 17 + 26? Think step by step."


def main() -> None:
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        max_tokens=80,
    )
    logprobs = response.choices[0].logprobs
    if not logprobs or not logprobs.content:
        print(f"No token probabilities returned: {DEFAULT_MODEL} on this "
              "server cannot be used for entropy.")
        return

    print("Token probabilities work. Each token and its entropy "
          "(higher = more uncertain):\n")
    for token in logprobs.content:
        print(f"{token.token!r:15} entropy={entropy_bits(token.top_logprobs):.3f}")


if __name__ == "__main__":
    main()