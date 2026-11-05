"""Compare the entropy curve of an easy question with a hard one.

This is the key result: an easy question (17 + 26) and an unsolved problem
(the Collatz conjecture) produce differently shaped entropy curves. The
script plots the two side by side and saves them as compare_cases.png.
"""

import matplotlib.pyplot as plt

from common import WINDOW, rolling_mean, summarize, token_entropies, verdict

CASES = {
    "Easy (should converge)": "What is 17 + 26? Think step by step.",
    "Hard (may not converge)": (
        "Prove or disprove the Collatz conjecture rigorously, "
        "with a complete formal proof."
    ),
}
PLOT_FILE = "compare_cases.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (title, prompt) in zip(axes, CASES.items()):
        entropies = token_entropies(prompt)
        early, late, overall = summarize(entropies)
        result = verdict(early, late, overall)
        print(f"{title}: early={early:.2f} late={late:.2f} "
              f"overall={overall:.2f} -> {result}")

        # Grey: raw entropy of each token. Red: the smoothed trajectory.
        ax.plot(entropies, color="#cccccc", lw=1, label="each token")
        ax.plot(rolling_mean(entropies), color="#c0392b", lw=2,
                label=f"rolling average ({WINDOW} tokens)")
        ax.set_title(f"{title}\n{result}")
        ax.set_xlabel("token position (time ->)")

    axes[0].set_ylabel("entropy (bits)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=120)
    print(f"\nSaved {PLOT_FILE}")


if __name__ == "__main__":
    main()