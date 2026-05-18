import argparse
import csv
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env

from config import ENVIRONMENT_CONFIGS, TRAINING_CONFIG_NAMES
from inventory_env import InventoryEnv


TRAINING_RUNS_DIR = Path("models/training_runs")
MODEL_NAME = "dqn_inventory"
INITIAL_MODEL_NAME = "initial_dqn_inventory"
BEST_MODEL_DIR_NAME = "best_model"
CHECKPOINT_DIR_NAME = "checkpoints"
TRAINING_METRICS_NAME = "training_metrics.csv"
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 10
PATIENCE_EVALUATIONS = 15


def run_dir(config_name):
    return TRAINING_RUNS_DIR / config_name


def run_paths(config_name):
    directory = run_dir(config_name)
    return {
        "model": directory / MODEL_NAME,
        "initial_model": directory / INITIAL_MODEL_NAME,
        "best_model_dir": directory / BEST_MODEL_DIR_NAME,
        "checkpoints": directory / CHECKPOINT_DIR_NAME,
        "evaluations": directory / "evaluations",
        "metrics": directory / TRAINING_METRICS_NAME,
    }


def evaluate_policy_metrics(model, env_config, n_eval_episodes=EVAL_EPISODES):
    results = []

    for seed in range(n_eval_episodes):
        env = InventoryEnv(**env_config)
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        total_cost = 0.0
        total_demand = 0
        total_sold = 0
        done = False

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += info["ordering_cost"] + info["holding_cost"] + info["shortage_cost"]
            total_demand += info["demand"]
            total_sold += info["sold"]
            done = terminated or truncated

        service_level = total_sold / total_demand if total_demand > 0 else 0.0
        results.append((total_reward, total_cost, service_level))

    return {
        "mean_reward": sum(result[0] for result in results) / len(results),
        "mean_total_cost": sum(result[1] for result in results) / len(results),
        "mean_service_level": sum(result[2] for result in results) / len(results),
    }


def write_training_metrics_header(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timesteps", "mean_reward", "mean_total_cost", "mean_service_level"],
        )
        writer.writeheader()


def append_training_metrics(path, timesteps, metrics):
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["timesteps", "mean_reward", "mean_total_cost", "mean_service_level"],
        )
        writer.writerow(
            {
                "timesteps": timesteps,
                "mean_reward": metrics["mean_reward"],
                "mean_total_cost": metrics["mean_total_cost"],
                "mean_service_level": metrics["mean_service_level"],
            }
        )


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, metrics_path, env_config, eval_freq, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self.metrics_path = metrics_path
        self.env_config = env_config
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_policy_metrics(self.model, self.env_config, self.n_eval_episodes)
        append_training_metrics(self.metrics_path, self.num_timesteps, metrics)
        return True


def train_config(config_name, config):
    env_config = config["env_config"]
    paths = run_paths(config_name)

    print(f"Training DQN for configuration: {config_name}")
    print(config["description"])

    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    paths["best_model_dir"].mkdir(parents=True, exist_ok=True)

    env = InventoryEnv(**env_config)
    eval_env = InventoryEnv(**env_config)
    check_env(env, warn=True)

    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE_EVALUATIONS,
        min_evals=5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best_model_dir"]),
        log_path=str(paths["evaluations"]),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        callback_after_eval=early_stop_callback,
        verbose=1,
    )
    metrics_callback = TrainingMetricsCallback(
        paths["metrics"],
        env_config,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=str(paths["checkpoints"]),
        name_prefix="dqn_inventory",
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.97,
        exploration_fraction=0.70,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        seed=123,
    )

    model.save(str(paths["initial_model"]))
    write_training_metrics_header(paths["metrics"])
    append_training_metrics(paths["metrics"], 0, evaluate_policy_metrics(model, env_config))

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([eval_callback, metrics_callback, checkpoint_callback]),
    )
    model.save(str(paths["model"]))
    append_training_metrics(paths["metrics"], model.num_timesteps, evaluate_policy_metrics(model, env_config))

    print(f"Saved initial model to {paths['initial_model']}.zip")
    print(f"Saved model to {paths['model']}.zip")
    print(f"Best evaluation model saved in {paths['best_model_dir']}.")
    print(f"Training checkpoints saved in {paths['checkpoints']}.")
    print(f"Training metrics saved to {paths['metrics']}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN inventory policies for named environment configs.")
    parser.add_argument(
        "configs",
        nargs="*",
        choices=sorted(ENVIRONMENT_CONFIGS.keys()),
        help="Optional config names to train. If omitted, trains the report configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_names = args.configs or TRAINING_CONFIG_NAMES

    for config_name in config_names:
        train_config(config_name, ENVIRONMENT_CONFIGS[config_name])


if __name__ == "__main__":
    main()
