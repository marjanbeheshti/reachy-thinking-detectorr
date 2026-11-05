"""Watch whether a reasoning model settles on an answer while it thinks.

Every PROBE_EVERY tokens, the script takes the reasoning so far, closes the
<think> block itself, and asks the model to finish "The final answer is:".
This "peek" shows which answer the model would give if it stopped now. If
the same answer keeps coming back, the model has converged.

Converged means committed, not correct: on this bat-and-ball question the
model often locks onto $0.10, the classic wrong answer (the right one is $0.05).
wizard_brain.py uses this same peek method with the robot.
"""

from common import TOP_LOGPROBS, client, entropy_bits

MODEL = "deepseek-r1:1.5b"  # a reasoning model, which thinks inside <think> tags
PROBE_EVERY = 40            # streamed tokens between peeks ("k" in the paper)
MAX_TOKENS = 400            # thinking budget
PROMPT = (
    "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. "
    "How much is the ball? Think step by step."
)


def peek(reasoning: str) -> tuple[str, float]:
    """Return the model's current answer and the average entropy of its tokens."""
    # Reasoning models answer only after </think>, so close the thinking
    # block ourselves and let the model complete the final answer.
    forced = reasoning + "\n</think>\nThe final answer is:"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": forced},
        ],
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        max_tokens=12,
        temperature=0.0,
    )
    tokens = response.choices[0].logprobs.content or []
    answer = "".join(token.token for token in tokens).strip()
    entropies = [entropy_bits(token.top_logprobs) for token in tokens]
    return answer, (sum(entropies) / len(entropies) if entropies else 0.0)


def judge(answers: list[str]) -> str:
    """Describe whether the answers across peeks have settled."""
    last = answers[-1]
    if last and answers[-3:].count(last) >= 2:
        return (f"answer locked in ('{last[:30]}') across peeks -> CONVERGED\n"
                "         (converged = committed, NOT necessarily correct)")
    if len(set(answers)) == len(answers):
        return "answer changes every peek -> still exploring / possible overthinking"
    return "partially settling -> still working"


def main() -> None:
    print(f"Model: {MODEL}\nPrompt: {PROMPT}\n")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )

    reasoning = ""
    tokens_since_peek = 0
    answers: list[str] = []
    for chunk in stream:
        piece = chunk.choices[0].delta.content if chunk.choices else None
        if not piece:
            continue
        reasoning += piece
        tokens_since_peek += 1  # Ollama streams about one token per chunk
        if tokens_since_peek >= PROBE_EVERY:
            tokens_since_peek = 0
            answer, entropy = peek(reasoning)
            answers.append(answer)
            print(f"[peek {len(answers)}] answer='{answer[:30]}'  "
                  f"answer-entropy={entropy:.2f}")

    print("\n--- reading the trajectory ---")
    if len(answers) < 2:
        print(f"Only {len(answers)} peek(s): the model finished too quickly to judge. "
              "Try a lower PROBE_EVERY.")
        return
    print(f"VERDICT: {judge(answers)}")


if __name__ == "__main__":
    main()