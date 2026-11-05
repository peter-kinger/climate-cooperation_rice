"""uv run python -m rllib_cpu.train --output runs/demo"""
import argparse
import csv
import json
from pathlib import Path
import ray
import yaml
from rllib_cpu.runtime import setup, build


def main():
    parser = argparse.ArgumentParser(description="Train shared CPU PPO policies for RICE-N")
    parser.add_argument("--config", type=Path, default=Path("configs/cpu.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if config["iterations"] < 1:
        parser.error("iterations must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    setup(config["num_cpus"])
    algo = None
    try:
        algo = build(config)
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["iteration", "env_steps", "episode_reward_mean"])
            writer.writeheader()
            for iteration in range(1, config["iterations"] + 1):
                result = algo.train()
                metrics = result.get("env_runners", result)
                row = {"iteration": iteration, "env_steps": result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0)),
                       "episode_reward_mean": metrics.get("episode_reward_mean", result.get("episode_reward_mean"))}
                writer.writerow(row)
                fp.flush()
                print(json.dumps(row), flush=True)
        checkpoint = algo.save(str(output / "checkpoint"))
        (output / "checkpoint_path.txt").write_text(str(checkpoint.checkpoint.path), encoding="utf-8")
        print(f"Saved run: {output}")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
