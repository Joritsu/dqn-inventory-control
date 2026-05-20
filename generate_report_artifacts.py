import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from config import ENVIRONMENT_CONFIGS
from inventory_env import InventoryEnv


ARTIFACT_DIR = Path("models/report_artifacts")
REPORT_CONFIG_NAME = "baseline"
TRAINING_RUN_DIR = Path("models/training_runs") / REPORT_CONFIG_NAME
TRAINING_METRICS_PATH = TRAINING_RUN_DIR / "training_metrics.csv"
INITIAL_MODEL_PATH = str(TRAINING_RUN_DIR / "initial_dqn_inventory")
BEST_MODEL_PATH = str(TRAINING_RUN_DIR / "best_model" / "best_model")
FINAL_MODEL_PATH = str(TRAINING_RUN_DIR / "dqn_inventory")
ORDER_COMPARISON_PATH = ARTIFACT_DIR / "initial_vs_best_orders.csv"
ORDER_COMPARISON_PLOT_PATH = ARTIFACT_DIR / "initial_vs_best_orders.png"
HEATMAP_PLOT_PATH = ARTIFACT_DIR / "inventory_heatmap.png"
TRAJECTORY_PLOT_PATH = ARTIFACT_DIR / "inventory_trajectories.png"
GRADIENT_COMPARISON_PATH = ARTIFACT_DIR / "gradient_comparison.png"


def report_env_config():
    return ENVIRONMENT_CONFIGS[REPORT_CONFIG_NAME]["env_config"]


def plot_training_metrics():
    if not TRAINING_METRICS_PATH.exists():
        print(f"No training metrics found at {TRAINING_METRICS_PATH}. Run `python train_dqn.py` first.")
        return

    metrics = pd.read_csv(TRAINING_METRICS_PATH)
    plots = [
        ("mean_reward", "Mean reward", "training_mean_reward.png"),
        ("mean_total_cost", "Mean total cost", "training_total_cost.png"),
        ("mean_service_level", "Mean service level", "training_service_level.png"),
    ]

    for column, title, filename in plots:
        plt.figure(figsize=(8, 4.5))
        plt.plot(metrics["timesteps"], metrics[column], marker="o", linewidth=2)
        plt.title(title)
        plt.xlabel("Training timesteps")
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / filename, dpi=160)
        plt.close()


def model_file_exists(model_path):
    path = Path(model_path)
    model_file = path if path.suffix == ".zip" else Path(f"{model_path}.zip")
    return model_file.exists()


def load_compatible_model(model_paths):
    expected_shape = InventoryEnv(**report_env_config()).observation_space.shape

    for model_path in model_paths:
        if not model_file_exists(model_path):
            continue

        model = DQN.load(str(model_path))
        if model.observation_space.shape == expected_shape:
            return model, str(model_path)

    return None, None


def run_model_episode(model, seed):
    env = InventoryEnv(**report_env_config())
    observation, info = env.reset(seed=seed)
    rows = []
    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": info["step"],
                "demand": info["demand"],
                "expected_demand": info["expected_demand"],
                "order": info["order_quantity"],
                "inventory": info["inventory"],
                "pending_orders": info["pending_orders_total"],
                "unmet_demand": info["unmet_demand"],
                "reward": reward,
            }
        )
        done = terminated or truncated

    return rows


def run_multiple_episodes(model, n_episodes=10, base_seed=0):
    all_episodes = []
    for i in range(n_episodes):
        rows = run_model_episode(model, seed=base_seed + i)
        for r in rows:
            r["episode"] = i
        all_episodes.extend(rows)
    return all_episodes


def write_order_comparison(seed=123):
    initial_model, initial_model_path = load_compatible_model([INITIAL_MODEL_PATH])
    trained_model, trained_model_path = load_compatible_model([BEST_MODEL_PATH, FINAL_MODEL_PATH])

    if initial_model is None or trained_model is None:
        print("Initial or trained DQN model is missing or incompatible. Run `python train_dqn.py` first.")
        return

    print(f"Comparing {initial_model_path} against {trained_model_path}.")

    initial_rows = run_model_episode(initial_model, seed)
    trained_rows = run_model_episode(trained_model, seed)

    with open(ORDER_COMPARISON_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "demand",
                "expected_demand",
                "initial_order",
                "best_order",
                "initial_inventory",
                "best_inventory",
                "initial_pending_orders",
                "best_pending_orders",
                "initial_unmet_demand",
                "best_unmet_demand",
            ],
        )
        writer.writeheader()

        for initial, trained in zip(initial_rows, trained_rows):
            writer.writerow(
                {
                    "step": initial["step"],
                    "demand": initial["demand"],
                    "expected_demand": initial["expected_demand"],
                    "initial_order": initial["order"],
                    "best_order": trained["order"],
                    "initial_inventory": initial["inventory"],
                    "best_inventory": trained["inventory"],
                    "initial_pending_orders": initial["pending_orders"],
                    "best_pending_orders": trained["pending_orders"],
                    "initial_unmet_demand": initial["unmet_demand"],
                    "best_unmet_demand": trained["unmet_demand"],
                }
            )

    comparison = pd.read_csv(ORDER_COMPARISON_PATH)
    plt.figure(figsize=(9, 5))
    plt.plot(comparison["step"], comparison["demand"], label="Demand", linewidth=2)
    plt.plot(comparison["step"], comparison["initial_order"], label="Initial policy order", alpha=0.8)
    plt.plot(comparison["step"], comparison["best_order"], label="Best policy order", alpha=0.8)
    plt.title("Order policy before and after training")
    plt.xlabel("Episode step")
    plt.ylabel("Units")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ORDER_COMPARISON_PLOT_PATH, dpi=160)
    plt.close()

    return initial_rows, trained_rows

def plot_inventory_heatmap(initial_rows, trained_rows, n_seeds=8):
    initial_model, _ = load_compatible_model([INITIAL_MODEL_PATH])
    trained_model, _ = load_compatible_model([BEST_MODEL_PATH, FINAL_MODEL_PATH])

    if initial_model is None or trained_model is None:
        print("Models not found – skipping heatmap generation.")
        return

    def collect_inventory_matrix(model, n_seeds):
        matrix = []
        for s in range(n_seeds):
            rows = run_model_episode(model, seed=s * 7)
            matrix.append([r["inventory"] for r in rows])
        max_len = max(len(r) for r in matrix)
        padded = [r + [np.nan] * (max_len - len(r)) for r in matrix]
        return np.array(padded, dtype=float)

    init_matrix = collect_inventory_matrix(initial_model, n_seeds)
    best_matrix = collect_inventory_matrix(trained_model, n_seeds)

    vmin = np.nanmin([init_matrix, best_matrix])
    vmax = np.nanmax([init_matrix, best_matrix])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    fig.suptitle("Inventory level heatmap across episodes")

    for ax, mat, title in zip(
        axes,
        [init_matrix, best_matrix],
        ["Initial policy (untrained model)", "Best model (after training)"],
    ):
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Episode step")
        ax.set_ylabel("Episode (seed)")
        ax.set_yticks(range(n_seeds))
        ax.set_yticklabels([f"#{i}" for i in range(n_seeds)])
        plt.colorbar(im, ax=ax, label="Inventory (units)", fraction=0.03, pad=0.02)

    plt.tight_layout()
    plt.savefig(HEATMAP_PLOT_PATH, dpi=160)
    plt.close()


def plot_inventory_trajectories(initial_rows, trained_rows):
    if initial_rows is None or trained_rows is None:
        print("No data available for inventory comparison plot.")
        return

    init_df = pd.DataFrame(initial_rows)
    best_df = pd.DataFrame(trained_rows)
    steps = init_df["step"].values

    plt.figure(figsize=(9, 5))
    plt.plot(steps, init_df["inventory"], label="Initial policy inventory", alpha=0.8)
    plt.plot(steps, best_df["inventory"], label="Best policy inventory", alpha=0.8)
    plt.title("Inventory comparison before and after training")
    plt.xlabel("Episode step")
    plt.ylabel("Inventory (units)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAJECTORY_PLOT_PATH, dpi=160)
    plt.close()


def plot_gradient_comparison(initial_rows, trained_rows):
    if initial_rows is None or trained_rows is None:
        print("No data available for cumulative unmet demand plot.")
        return

    init_df = pd.DataFrame(initial_rows)
    best_df = pd.DataFrame(trained_rows)
    steps = init_df["step"].values

    plt.figure(figsize=(9, 5))
    plt.plot(steps, init_df["unmet_demand"].cumsum(), label="Initial policy", alpha=0.8)
    plt.plot(steps, best_df["unmet_demand"].cumsum(), label="Best policy", alpha=0.8)
    plt.title("Cumulative unmet demand")
    plt.xlabel("Episode step")
    plt.ylabel("Units (cumulative)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRADIENT_COMPARISON_PATH, dpi=160)
    plt.close()


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    plot_training_metrics()
    result = write_order_comparison()
    initial_rows, trained_rows = result if result else (None, None)

    plot_inventory_heatmap(initial_rows, trained_rows)
    plot_inventory_trajectories(initial_rows, trained_rows)
    plot_gradient_comparison(initial_rows, trained_rows)
    print(f"Report artifacts saved in {ARTIFACT_DIR}.")


if __name__ == "__main__":
    main()
