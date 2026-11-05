"""Reload checkpoint in a separate process and export complete trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import ray
import torch
import yaml
from ray.rllib.algorithms.algorithm import Algorithm
from rllib_cpu.env import RiceEnv
from rllib_cpu.runtime import setup


KEYS = ["global_temperature", "global_carbon_mass", "gross_output_all_regions",
        "consumption_all_regions", "mitigation_rate_all_regions", "utility_all_regions",
        "social_welfare_all_regions", "capital_all_regions", "labor_all_regions"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--explore", action="store_true", help="Sample policy actions instead of deterministic inference")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    run = args.run.resolve()
    for stale in run.glob("trajectory_*.npz"):
        stale.unlink()
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    setup(config["num_cpus"])
    algo = None
    try:
        # Relative checkpoint location allows moving the entire run directory.
        algo = Algorithm.from_checkpoint(str(run / "checkpoint"))
        rows, region_rows, rewards = [], [], []
        for episode in range(args.episodes):
            torch.manual_seed(config["seed"] + episode)
            np.random.seed(config["seed"] + episode)
            env = RiceEnv(config["env"])
            obs, _ = env.reset(seed=config["seed"] + episode)
            total = 0.0
            for _ in range(env.env.episode_length):
                actions = {i: algo.compute_single_action(o, policy_id="regions", explore=args.explore)
                           for i, o in obs.items()}
                obs, rew, terminated, _, _ = env.step(actions)
                total += sum(rew.values())
                if terminated["__all__"]:
                    break
            sim = env.env
            states = {key: sim.global_state[key]["value"][:sim.timestep + 1] for key in KEYS}
            if not all(np.isfinite(value).all() for value in states.values()):
                raise ValueError("Non-finite simulation outputs")
            np.savez_compressed(run / f"trajectory_{episode:03d}.npz", **states)
            steps = sim.global_state["activity_timestep"]["value"][:sim.timestep + 1].reshape(-1)
            # Negotiation adds two zero-time steps. Only retain the economic state each period.
            selected = [0] + [i for i in range(1, len(steps)) if steps[i] > steps[i - 1]]
            for t in selected:
                year = sim.start_year + int(steps[t]) * sim.dice_constant["xDelta"]
                row = {"episode": episode, "year": year,
                       "temperature": float(states["global_temperature"][t].reshape(-1)[0]),
                       "carbon_atmosphere": float(states["global_carbon_mass"][t].reshape(-1)[0])}
                for field, key in [("gross_output", "gross_output_all_regions"), ("consumption", "consumption_all_regions"),
                                   ("utility", "utility_all_regions"), ("social_welfare", "social_welfare_all_regions")]:
                    row[field] = float(states[key][t].sum())
                row["mean_mitigation"] = float(states["mitigation_rate_all_regions"][t].mean())
                rows.append(row)
                for i in range(sim.num_regions):
                    region_rows.append({"episode": episode, "year": year, "region_id": i,
                        **{key.removesuffix("_all_regions"): float(states[key][t, i]) for key in KEYS if key.endswith("_all_regions")}})
            rewards.append(total)
        pd.DataFrame(rows).to_csv(run / "evaluation.csv", index=False)
        pd.DataFrame(region_rows).to_csv(run / "regions.csv", index=False)
        summary = {"episodes": args.episodes, "explore": args.explore, "mean_total_reward": float(np.mean(rewards)),
                   "final_temperature_mean": float(np.mean([r["temperature"] for r in rows if r["year"] == sim.end_year])),
                   "region_files_in_agent_order": sorted(f.name for f in (Path(__file__).resolve().parents[1] / "region_yamls").glob("*.yml") if f.name != "default.yml")}
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
