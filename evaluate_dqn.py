from pathlib import Path

from stable_baselines3 import DQN

from config import ENVIRONMENT_CONFIGS
from inventory_env import InventoryEnv


EVALUATION_CONFIG_NAME = "baseline"
TRAINING_RUNS_DIR = Path("models/training_runs")
EVALUATION_EPISODES = 100
BASE_STOCK_TARGETS = [60, 70, 80, 90, 100, 110, 120, 130]


def checkpoint_timesteps(checkpoint_path):
    name_parts = checkpoint_path.stem.split("_")

    if len(name_parts) < 2 or name_parts[-1] != "steps":
        return -1

    try:
        return int(name_parts[-2])
    except ValueError:
        return -1


def evaluation_env_config():
    return ENVIRONMENT_CONFIGS[EVALUATION_CONFIG_NAME]["env_config"]


def load_compatible_dqn():
    run_dir = TRAINING_RUNS_DIR / EVALUATION_CONFIG_NAME
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = sorted(
        checkpoint_dir.glob("dqn_inventory_*_steps.zip"),
        key=checkpoint_timesteps,
        reverse=True,
    )
    candidate_paths = [
        run_dir / "dqn_inventory",
        run_dir / "best_model" / "best_model",
    ] + checkpoint_paths
    expected_shape = InventoryEnv(**evaluation_env_config()).observation_space.shape

    for model_path in candidate_paths:
        path = Path(model_path)
        model_file = path if path.suffix == ".zip" else Path(f"{model_path}.zip")
        if not model_file.exists():
            continue

        model = DQN.load(str(model_path))
        if model.observation_space.shape == expected_shape:
            return model, str(model_path)

    return None, None


def run_episode(policy, seed):
    env = InventoryEnv(**evaluation_env_config())
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    unmet_demand = 0
    total_demand = 0
    total_sold = 0
    total_inventory = 0
    total_order = 0
    total_cost = 0.0
    done = False

    while not done:
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_demand += info["demand"]
        total_sold += info["sold"]
        unmet_demand += info["unmet_demand"]
        total_inventory += info["inventory"]
        total_order += info["order_quantity"]
        total_cost += info["ordering_cost"] + info["holding_cost"] + info["shortage_cost"]
        done = terminated or truncated

    service_level = total_sold / total_demand if total_demand > 0 else 0.0
    return (
        total_reward,
        unmet_demand,
        total_inventory / env.episode_length,
        total_order / env.episode_length,
        total_cost,
        service_level,
    )


def summarize_policy(name, policy):
    results = [run_episode(policy, seed) for seed in range(EVALUATION_EPISODES)]

    rewards = [reward for reward, _, _, _, _, _ in results]
    unmet_demands = [unmet for _, unmet, _, _, _, _ in results]
    inventories = [inventory for _, _, inventory, _, _, _ in results]
    orders = [order for _, _, _, order, _, _ in results]
    costs = [cost for _, _, _, _, cost, _ in results]
    service_levels = [service_level for _, _, _, _, _, service_level in results]

    print(f"{name}")
    print(f"  Mean reward: {sum(rewards) / len(rewards):.2f}")
    print(f"  Best reward: {max(rewards):.2f}")
    print(f"  Worst reward: {min(rewards):.2f}")
    print(f"  Mean total cost: {sum(costs) / len(costs):.2f}")
    print(f"  Mean service level: {sum(service_levels) / len(service_levels):.2%}")
    print(f"  Mean unmet demand: {sum(unmet_demands) / len(unmet_demands):.2f}")
    print(f"  Mean inventory: {sum(inventories) / len(inventories):.2f}")
    print(f"  Mean order: {sum(orders) / len(orders):.2f}")
    print()


def base_stock_policy(target_inventory):
    def policy(observation, env):
        current_inventory = int(round(observation[0] * env.max_inventory))
        pending_orders = env._pending_orders_total()
        inventory_position = current_inventory + pending_orders
        return max(0, min(env.max_order, target_inventory - inventory_position))

    return policy


def random_policy(observation, env):
    return env.action_space.sample()


def main():
    model, model_path = load_compatible_dqn()

    if model is not None:
        summarize_policy(
            f"Trained DQN ({model_path})",
            lambda observation, env: model.predict(observation, deterministic=True)[0],
        )
    else:
        print("No compatible DQN model found.")
        print(f"Run `python train_dqn.py` to train one for `{EVALUATION_CONFIG_NAME}`.")
        print()

    summarize_policy("Random policy", random_policy)
    for target_inventory in BASE_STOCK_TARGETS:
        summarize_policy(
            f"Base-stock S={target_inventory}",
            base_stock_policy(target_inventory),
        )


if __name__ == "__main__":
    main()
