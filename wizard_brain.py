"""Answer the robot's questions with an LLM, and report whether it settled.

This program joins the Wizard-of-Oz server as a dashboard client. Whenever
the robot sends a transcribed question, it lets a reasoning model think,
periodically "peeks" at the answer the model would give so far, and replies
once that answer stops changing. The robot then speaks the reply.

States:
    CONVERGED  two peeks in a row gave the same answer
    UNSETTLED  the answer kept changing; the most frequent one is used
    STUCK      no answer could be read at all

Run order (each in its own terminal):
    1. ollama serve                      (with deepseek-r1:1.5b pulled)
    2. node wizardOfOz/index.js          (message server)
    3. python experiment/experiment.py   (robot: microphone and speaker)
    4. python wizard_brain.py            (this file)
"""

import asyncio
import json
import re

import websockets
from openai import OpenAI

SERVER_URL = "ws://localhost:8080/?dashboard=true"
MODEL = "deepseek-r1:1.5b"
PROBE_EVERY = 25   # streamed tokens between peeks (lower = more peeks)
MAX_TOKENS = 400   # thinking budget per question

# Ollama's OpenAI-compatible endpoint; the API key is required but ignored.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def peek_answer(question: str, reasoning: str) -> str:
    """Return the answer the model would give if it stopped thinking now."""
    # Reasoning models answer only after </think>, so close the thinking
    # block ourselves and let the model complete the final answer.
    forced = reasoning + "\n</think>\nThe final answer is:"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": forced},
        ],
        max_tokens=12,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def clean_answer(answer: str) -> str:
    r"""Turn LaTeX-style output such as '\boxed{43}' into speakable text."""
    answer = re.sub(r"\\[a-zA-Z]+", " ", answer)  # drop \boxed, \text, ...
    for symbol in "\\[](){}*$#":
        answer = answer.replace(symbol, " ")
    return re.sub(r"\s+", " ", answer).strip()


def has_converged(answers: list[str]) -> bool:
    """True when the last two peeks returned the same non-empty answer."""
    return len(answers) >= 2 and answers[-1] == answers[-2] != ""


def run_detector(question: str) -> tuple[str, str]:
    """Let the model think about *question*; return (state, answer)."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
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
            answers.append(peek_answer(question, reasoning))
            print(f"   [THINKING] -> {answers[-1][:40]!r}")
            if has_converged(answers):
                return "CONVERGED", answers[-1]

    # The model stopped on its own: take one last peek, so short replies
    # that finished before the first peek are not mistaken for STUCK.
    if reasoning and tokens_since_peek > 0:
        answers.append(peek_answer(question, reasoning))
        print(f"   [FINAL]    -> {answers[-1][:40]!r}")
        if has_converged(answers):
            return "CONVERGED", answers[-1]

    answers = [a for a in answers if a]
    if not answers:
        return "STUCK", ""
    if len(answers) == 1:
        return "CONVERGED", answers[0]
    # Never settled: fall back to the most frequent answer, not the last one.
    return "UNSETTLED", max(set(answers), key=answers.count)


def make_reply(state: str, answer: str) -> str:
    """Turn a detector result into the sentence the robot will speak."""
    if state == "CONVERGED":
        return f"The answer is {clean_answer(answer)}."
    if state == "UNSETTLED":
        return f"I'm not sure, but my best guess is {clean_answer(answer)}."
    return "I'm not sure. I seem to be stuck."


async def main() -> None:
    async with websockets.connect(SERVER_URL) as ws:
        print("Brain connected. Waiting for questions...\n")
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # The server echoes our own replies as "sent"; only answer the robot.
            if message.get("direction") != "received":
                continue
            question = (message.get("text") or "").strip()
            if not question:
                continue

            print(f"\nQ: {question}")
            # Thinking is slow and blocking, so run it off the event loop.
            state, answer = await asyncio.to_thread(run_detector, question)
            reply = make_reply(state, answer)
            print(f"A: {reply}  [{state}]")
            await ws.send(json.dumps({"text": reply}))


if __name__ == "__main__":
    asyncio.run(main())