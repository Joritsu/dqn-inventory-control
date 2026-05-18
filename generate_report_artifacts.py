import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN

from inventory_env import InventoryEnv


ARTIFACT_DIR = Path("models/report_artifacts")
REPORT_CONFIG_NAME = "baseline"
TRAINING_RUN_DIR = Path("models/training_runs") / REPORT_CONFIG_NAME
TRAINING_METRICS_PATH = TRAINING_RUN_DIR / "training_metrics.csv"
INITIAL_MODEL_PATH = str(TRAINING_RUN_DIR / "initial_dqn_inventory")
FINAL_MODEL_PATH = str(TRAINING_RUN_DIR / "dqn_inventory")
ORDER_COMPARISON_PATH = ARTIFACT_DIR / "initial_vs_final_orders.csv"


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


def run_model_episode(model, seed):
    env = InventoryEnv()
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
    initial_path = Path(f"{INITIAL_MODEL_PATH}.zip")
    final_path = Path(f"{FINAL_MODEL_PATH}.zip")

    if not initial_path.exists() or not final_path.exists():
        print("Initial or final DQN model is missing. Run `python train_dqn.py` first.")
        return

    initial_model = DQN.load(INITIAL_MODEL_PATH)
    final_model = DQN.load(FINAL_MODEL_PATH)
    expected_shape = InventoryEnv().observation_space.shape

    if initial_model.observation_space.shape != expected_shape or final_model.observation_space.shape != expected_shape:
        print("Initial or final DQN model is incompatible with the current environment.")
        print("Run `python train_dqn.py` before regenerating the order-comparison artifacts.")
        return

    initial_rows = run_model_episode(initial_model, seed)
    final_rows = run_model_episode(final_model, seed)

    with open(ORDER_COMPARISON_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "demand",
                "expected_demand",
                "initial_order",
                "final_order",
                "initial_inventory",
                "final_inventory",
                "initial_pending_orders",
                "final_pending_orders",
                "initial_unmet_demand",
                "final_unmet_demand",
            ],
        )
        writer.writeheader()

        for initial, final in zip(initial_rows, final_rows):
            writer.writerow(
                {
                    "step": initial["step"],
                    "demand": initial["demand"],
                    "expected_demand": initial["expected_demand"],
                    "initial_order": initial["order"],
                    "final_order": final["order"],
                    "initial_inventory": initial["inventory"],
                    "final_inventory": final["inventory"],
                    "initial_pending_orders": initial["pending_orders"],
                    "final_pending_orders": final["pending_orders"],
                    "initial_unmet_demand": initial["unmet_demand"],
                    "final_unmet_demand": final["unmet_demand"],
                }
            )

    comparison = pd.read_csv(ORDER_COMPARISON_PATH)
    plt.figure(figsize=(9, 5))
    plt.plot(comparison["step"], comparison["demand"], label="Demand", linewidth=2)
    plt.plot(comparison["step"], comparison["initial_order"], label="Initial policy order", alpha=0.8)
    plt.plot(comparison["step"], comparison["final_order"], label="Final policy order", alpha=0.8)
    plt.title("Order policy before and after training")
    plt.xlabel("Episode step")
    plt.ylabel("Units")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "initial_vs_final_orders.png", dpi=160)
    plt.close()


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_metrics()
    write_order_comparison()
    print(f"Report artifacts saved in {ARTIFACT_DIR}.")


if __name__ == "__main__":
    main()
