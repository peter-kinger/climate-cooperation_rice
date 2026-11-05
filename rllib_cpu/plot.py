"""Plot exported evaluation tables; no Ray runtime is required."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot RICE-N evaluation and training results")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    df = pd.read_csv(run / "evaluation.csv")
    mean = df.groupby("year").mean(numeric_only=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, [("temperature", "Atmospheric temperature anomaly (C)"),
            ("carbon_atmosphere", "Atmospheric carbon (model units)"), ("gross_output", "Global gross output (model units)"),
            ("consumption", "Global consumption (model units)"), ("mean_mitigation", "Mean mitigation rate"),
            ("social_welfare", "Global discounted welfare")]):
        for _, group in df.groupby("episode"):
            ax.plot(group.year, group[key], alpha=0.2, color="steelblue")
        ax.plot(mean.index, mean[key], color="steelblue")
        ax.set(xlabel="Year", ylabel=label)
        ax.grid(alpha=0.25)
    fig.savefig(run / "analysis.png", dpi=160)
    plt.close(fig)
    regions = pd.read_csv(run / "regions.csv")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for region, group in regions.groupby("region_id"):
        values = group.groupby("year").mitigation_rate.mean()
        ax.plot(values.index, values, label=str(region), linewidth=1)
    ax.set(xlabel="Year", ylabel="Mitigation rate")
    ax.legend(title="Region ID", ncol=5, fontsize=7)
    fig.savefig(run / "regional_mitigation.png", dpi=160)
    plt.close(fig)
    training = pd.read_csv(run / "training.csv")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(training.iteration, training.episode_reward_mean, marker="o")
    ax.set(xlabel="Training iteration", ylabel="Mean episode reward (sum over regions)")
    fig.savefig(run / "training.png", dpi=160)
    plt.close(fig)
    print(f"Charts saved in {run.resolve()}")


if __name__ == "__main__":
    main()
