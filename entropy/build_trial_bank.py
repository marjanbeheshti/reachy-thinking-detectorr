"""Check the detector's verdicts on a fixed set of labelled questions.

Each trial is a question that should either converge (names starting with
"healthy_") or overthink (all others). Every question runs REPEATS times,
because the small model's output varies from run to run, and the verdict
comes from the averaged early, late, and overall entropies. The thresholds
in common.py were calibrated with this script.

Results are printed as a table and saved to trials/<name>.json.
"""

import json
import os
from statistics import mean

from common import summarize, token_entropies, verdict

REPEATS = 3          # runs per question, averaged to smooth out randomness
OUTPUT_DIR = "trials"

# (name, prompt, temperature)
TRIALS = [
    ("healthy_add", "What is 17 + 26? Think step by step.", 0.7),
    ("healthy_capital", "What is the capital of France? Explain briefly.", 0.7),
    ("healthy_sort", "Put these in order: 5, 2, 9, 1. Show your steps.", 0.7),
    ("over_collatz",
     "Prove or disprove the Collatz conjecture with a complete formal proof.", 0.7),
    ("over_evenodd",
     "Find an even number that is also odd, and rigorously prove it exists.", 0.7),
    ("over_nocommit",
     "Consider many possibilities for the meaning of life. Do not commit to "
     "any answer. Keep questioning each idea forever.", 1.3),
]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"{'trial':16} {'expected':12} {'early':>6} {'late':>6} {'over':>6}  verdict")

    for name, prompt, temperature in TRIALS:
        expected = "healthy" if name.startswith("healthy") else "overthinking"
        runs = [token_entropies(prompt, temperature=temperature) for _ in range(REPEATS)]

        # Average early, late, and overall across the runs, then judge once.
        summaries = [summarize(run) for run in runs]  # (early, late, overall) per run
        early = mean(s[0] for s in summaries)
        late = mean(s[1] for s in summaries)
        overall = mean(s[2] for s in summaries)
        result = verdict(early, late, overall)
        match = "OK" if expected in result.lower() else "<-- MISMATCH"
        print(f"{name:16} {expected:12} {early:6.2f} {late:6.2f} {overall:6.2f}"
              f"  {result}  {match}")

        record = {
            "prompt": prompt,
            "temperature": temperature,
            "expected": expected,
            "early": early,
            "late": late,
            "overall": overall,
            "verdict": result,
            "entropy_runs": runs,  # every run's curve, not just the last one
        }
        with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
            json.dump(record, f)


if __name__ == "__main__":
    main()