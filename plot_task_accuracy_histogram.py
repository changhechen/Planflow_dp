import matplotlib.pyplot as plt
import numpy as np


def main():
    tasks = [
        "Sort table",
        "Sort table Reverse",
        "Tool Usage & Pick Cube",
    ]

    model_scores = {
        "DP": [0.10, 0.60, 0.50],
        "Pi0": [0.54, 0.50, 0.08],
        "PlanFlow-DP": [0.70, 0.72, 0.62],
    }

    colors = {
        "DP": "#4C78A8",
        "Pi0": "#F58518",
        "PlanFlow-DP": "#54A24B",
    }

    x = np.arange(len(tasks))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, scores) in enumerate(model_scores.items()):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=model_name,
            color=colors[model_name],
        )

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Tasks")
    ax.set_ylabel("Accuracy")
    ax.set_title("Task Accuracy by Model")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=10)
    ax.set_ylim(0, 0.85)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("task_accuracy_histogram.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
