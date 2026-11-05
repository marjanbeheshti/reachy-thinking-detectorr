"""Run one question and draw how the model's uncertainty changes over time.

This is the basic detector. It asks the model one question, computes the
entropy of every generated token, smooths the curve with a rolling average,
prints a verdict, and saves the plot as entropy_trajectory.png.
"""

import matplotlib.pyplot as plt

from common import WINDOW, rolling_mean, summarize, token_entropies, verdict

PROMPT = "What is 17 + 26? Think step by step."
PLOT_FILE = "entropy_trajectory.png"


def main() -> None:
    entropies = token_entropies(PROMPT, max_tokens=120)
    early, late, overall = summarize(entropies)
    result = verdict(early, late, overall)
    print(f"Early avg: {early:.2f}   Late avg: {late:.2f}   Overall: {overall:.2f}")
    print(f"VERDICT: {result}")

    # Grey: raw entropy of each token. Red: the smoothed trajectory.
    plt.figure(figsize=(8, 4))
    plt.plot(entropies, color="#cccccc", lw=1, label="each token")
    plt.plot(rolling_mean(entropies), color="#c0392b", lw=2,
             label=f"rolling average ({WINDOW} tokens)")
    plt.title(f"{PROMPT}\n{result}")
    plt.xlabel("token position (time ->)")
    plt.ylabel("entropy (bits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=120)
    print(f"Saved {PLOT_FILE}")


if __name__ == "__main__":
    main()