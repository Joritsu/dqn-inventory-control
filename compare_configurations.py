from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from config import ENVIRONMENT_CONFIGS, TRAINING_CONFIG_NAMES
from inventory_env import InventoryEnv


ARTIFACT_DIR = Path("models/report_artifacts")
TRAINING_RUNS_DIR = Path("models/training_runs")
EVALUATION_EPISODES = 100
BASE_STOCK_TARGET = 110


def random_policy(observation, env):
    return env.action_space.sample()


def base_stock_policy(target_inventory):
    def policy(observation, env):
        current_inventory = int(round(observation[0] * env.max_inventory))
        pending_orders = env._pending_orders_total()
        inventory_position = current_inventory + pending_orders
        return max(0, min(env.max_order, target_inventory - inventory_position))

    return policy


def dqn_policy(model):
    def policy(observation, env):
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    return policy


def checkpoint_timesteps(checkpoint_path):
    name_parts = checkpoint_path.stem.split("_")

    if len(name_parts) < 2 or name_parts[-1] != "steps":
        return -1

    try:
        return int(name_parts[-2])
    except ValueError:
        return -1


def training_run_dir(config_name):
    return TRAINING_RUNS_DIR / config_name


def load_compatible_dqn(config_name, env_config):
    run_dir = training_run_dir(config_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = sorted(
        checkpoint_dir.glob("dqn_inventory_*_steps.zip"),
        key=checkpoint_timesteps,
        reverse=True,
    )
    candidate_paths = [
        run_dir / "best_model" / "best_model",
        run_dir / "dqn_inventory",
    ] + checkpoint_paths
    expected_shape = InventoryEnv(**env_config).observation_space.shape

    for model_path in candidate_paths:
        path = Path(model_path)
        model_file = path if path.suffix == ".zip" else Path(f"{model_path}.zip")
        if not model_file.exists():
            continue

        model = DQN.load(str(model_path))
        if model.observation_space.shape == expected_shape:
            return model, str(model_path)

    return None, None


def run_episode(policy, env_config, seed):
    env = InventoryEnv(**env_config)
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    total_cost = 0.0
    total_demand = 0
    total_sold = 0
    total_unmet_demand = 0
    total_inventory = 0
    total_order = 0
    done = False

    while not done:
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += info["ordering_cost"] + info["holding_cost"] + info["shortage_cost"]
        total_demand += info["demand"]
        total_sold += info["sold"]
        total_unmet_demand += info["unmet_demand"]
        total_inventory += info["inventory"]
        total_order += info["order_quantity"]
        done = terminated or truncated

    service_level = total_sold / total_demand if total_demand > 0 else 0.0

    return {
        "reward": total_reward,
        "total_cost": total_cost,
        "service_level": service_level,
        "unmet_demand": total_unmet_demand,
        "average_inventory": total_inventory / env.episode_length,
        "average_order": total_order / env.episode_length,
    }


def summarize_policy(config_name, policy_name, policy, env_config):
    episodes = [
        run_episode(policy, env_config, seed)
        for seed in range(EVALUATION_EPISODES)
    ]

    return {
        "config": config_name,
        "policy": policy_name,
        "episodes": EVALUATION_EPISODES,
        "mean_reward": np.mean([episode["reward"] for episode in episodes]),
        "best_reward": np.max([episode["reward"] for episode in episodes]),
        "worst_reward": np.min([episode["reward"] for episode in episodes]),
        "mean_total_cost": np.mean([episode["total_cost"] for episode in episodes]),
        "mean_service_level": np.mean([episode["service_level"] for episode in episodes]),
        "mean_unmet_demand": np.mean([episode["unmet_demand"] for episode in episodes]),
        "mean_inventory": np.mean([episode["average_inventory"] for episode in episodes]),
        "mean_order": np.mean([episode["average_order"] for episode in episodes]),
    }


def load_policies(config_name, env_config):
    policies = [
        ("Random policy", random_policy),
        (f"Base-stock S={BASE_STOCK_TARGET}", base_stock_policy(BASE_STOCK_TARGET)),
    ]

    model, model_path = load_compatible_dqn(config_name, env_config)
    if model is None:
        print(f"No compatible DQN model found for {config_name}. The experiment will use only baseline policies.")
    else:
        policies.insert(0, (f"Best DQN trained on {config_name}", dqn_policy(model)))

    return policies


def write_config_settings(configs):
    rows = []

    for config in configs:
        env = InventoryEnv(**config["env_config"])
        rows.append(
            {
                "config": config["name"],
                "description": config["description"],
                "episode_length": env.episode_length,
                "max_inventory": env.max_inventory,
                "initial_inventory": env.initial_inventory,
                "max_order": env.max_order,
                "lead_time": env.lead_time,
                "mean_demand": env.mean_demand,
                "demand_std": env.demand_std,
                "demand_trend_strength": env.demand_trend_strength,
                "demand_seasonal_amplitude": env.demand_seasonal_amplitude,
                "demand_spike_chance": env.demand_spike_chance,
                "demand_spike_multiplier": env.demand_spike_multiplier,
                "demand_regimes": config["env_config"].get("demand_regimes", []),
                "order_cost": env.order_cost,
                "holding_cost": env.holding_cost,
                "shortage_cost": env.shortage_cost,
            }
        )

    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "config_experiment_settings.csv", index=False)


def plot_metric(results, column, title, ylabel, filename, percent=False):
    pivot = results.pivot(index="config", columns="policy", values=column)
    ax = pivot.plot(kind="bar", figsize=(9, 5), width=0.75)

    ax.set_title(title)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Policy")

    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / filename, dpi=160)
    plt.close()


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows = []

    experiment_configs = [
        {"name": config_name, **ENVIRONMENT_CONFIGS[config_name]}
        for config_name in TRAINING_CONFIG_NAMES
    ]

    for config in experiment_configs:
        print(f"Running configuration: {config['name']}")
        policies = load_policies(config["name"], config["env_config"])
        for policy_name, policy in policies:
            print(f"  Evaluating {policy_name}")
            result_rows.append(
                summarize_policy(
                    config["name"],
                    policy_name,
                    policy,
                    config["env_config"],
                )
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(ARTIFACT_DIR / "config_experiment_results.csv", index=False)
    write_config_settings(experiment_configs)

    plot_metric(
        results,
        "mean_total_cost",
        "Mean total cost by configuration",
        "Mean total cost",
        "config_mean_total_cost.png",
    )
    plot_metric(
        results,
        "mean_service_level",
        "Mean service level by configuration",
        "Mean service level",
        "config_mean_service_level.png",
        percent=True,
    )
    plot_metric(
        results,
        "mean_reward",
        "Mean reward by configuration",
        "Mean reward",
        "config_mean_reward.png",
    )

    print(f"Configuration experiment outputs saved in {ARTIFACT_DIR}.")


if __name__ == "__main__":
    main()
