from stable_baselines3 import DQN

from config import MAX_ORDER
from inventory_env import InventoryEnv


MODEL_PATH = "models/dqn_inventory"
EVALUATION_EPISODES = 100


def run_episode(policy, seed):
    env = InventoryEnv()
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    unmet_demand = 0
    total_inventory = 0
    total_order = 0
    done = False

    while not done:
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        unmet_demand += info["unmet_demand"]
        total_inventory += info["inventory"]
        total_order += info["order_quantity"]
        done = terminated or truncated

    return total_reward, unmet_demand, total_inventory / env.episode_length, total_order / env.episode_length


def summarize_policy(name, policy):
    results = [run_episode(policy, seed) for seed in range(EVALUATION_EPISODES)]

    rewards = [reward for reward, _, _, _ in results]
    unmet_demands = [unmet for _, unmet, _, _ in results]
    inventories = [inventory for _, _, inventory, _ in results]
    orders = [order for _, _, _, order in results]

    print(f"{name}")
    print(f"  Mean reward: {sum(rewards) / len(rewards):.2f}")
    print(f"  Best reward: {max(rewards):.2f}")
    print(f"  Worst reward: {min(rewards):.2f}")
    print(f"  Mean unmet demand: {sum(unmet_demands) / len(unmet_demands):.2f}")
    print(f"  Mean inventory: {sum(inventories) / len(inventories):.2f}")
    print(f"  Mean order: {sum(orders) / len(orders):.2f}")
    print()


def base_stock_policy(target_inventory):
    def policy(observation, env):
        current_inventory = int(observation[0])
        return max(0, min(MAX_ORDER, target_inventory - current_inventory))

    return policy


def random_policy(observation, env):
    return env.action_space.sample()


def main():
    model = DQN.load(MODEL_PATH)

    summarize_policy(
        "Trained DQN",
        lambda observation, env: model.predict(observation, deterministic=True)[0],
    )
    summarize_policy("Random policy", random_policy)
    summarize_policy("Base-stock S=25", base_stock_policy(25))
    summarize_policy("Base-stock S=30", base_stock_policy(30))
    summarize_policy("Base-stock S=35", base_stock_policy(35))
    summarize_policy("Base-stock S=40", base_stock_policy(40))


if __name__ == "__main__":
    main()
