import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_metrics()
    write_order_comparison()
    print(f"Report artifacts saved in {ARTIFACT_DIR}.")


if __name__ == "__main__":
    main()
