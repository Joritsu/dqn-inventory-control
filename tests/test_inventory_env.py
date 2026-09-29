import unittest

from stable_baselines3 import DQN

from config import ENVIRONMENT_CONFIGS
from inventory_env import InventoryEnv


class InventoryEnvironmentTests(unittest.TestCase):
    def make_env(self, **overrides):
        settings = {
            "episode_length": 4,
            "max_inventory": 10,
            "initial_inventory": 0,
            "max_order": 6,
            "lead_time": 0,
            "mean_demand": 0,
            "demand_std": 0,
            "demand_trend_strength": 0,
            "demand_seasonal_amplitude": 0,
            "demand_spike_chance": 0,
        }
        settings.update(overrides)
        return InventoryEnv(**settings)

    def test_immediate_delivery_sales_and_reward(self):
        env = self.make_env(initial_inventory=3, mean_demand=5)
        env.reset(seed=1)

        _, reward, _, _, info = env.step(4)
        self.assertEqual(info["arriving_order"], 4)
        self.assertEqual(info["sold"], 5)
        self.assertEqual(info["unmet_demand"], 0)
        self.assertEqual(info["inventory"], 2)
        self.assertEqual(info["ordering_cost"], 4)
        self.assertEqual(info["holding_cost"], 2)
        self.assertEqual(reward, 34)

        _, reward, _, _, info = env.step(0)
        self.assertEqual(info["sold"], 2)
        self.assertEqual(info["unmet_demand"], 3)
        self.assertEqual(info["inventory"], 0)
        self.assertEqual(reward, -14)

    def test_delivery_lead_time_and_reset(self):
        env = self.make_env(lead_time=2)
        env.reset(seed=2)

        _, _, _, _, first = env.step(4)
        _, _, _, _, second = env.step(0)
        _, _, _, _, third = env.step(0)

        self.assertEqual(first["arriving_order"], 0)
        self.assertEqual(first["pending_orders"], [0, 4])
        self.assertEqual(second["arriving_order"], 0)
        self.assertEqual(second["pending_orders"], [4, 0])
        self.assertEqual(third["arriving_order"], 4)
        self.assertEqual(third["inventory"], 4)

        _, reset_info = env.reset(seed=2)
        self.assertEqual(reset_info["inventory"], 0)
        self.assertEqual(reset_info["pending_orders"], [0, 0])

    def test_warehouse_capacity_discards_excess_arrivals(self):
        env = self.make_env(initial_inventory=8)
        env.reset(seed=3)

        _, reward, _, _, info = env.step(5)
        self.assertEqual(info["arriving_order"], 5)
        self.assertEqual(info["inventory"], 10)
        self.assertEqual(info["holding_cost"], 10)
        self.assertEqual(reward, -14)

    def test_saved_best_models_match_both_scenarios(self):
        for name, config in ENVIRONMENT_CONFIGS.items():
            with self.subTest(scenario=name):
                env = InventoryEnv(**config["env_config"])
                model = DQN.load(
                    f"models/training_runs/{name}/best_model/best_model.zip",
                    device="cpu",
                )
                observation, _ = env.reset(seed=4)
                self.assertEqual(model.observation_space.shape, env.observation_space.shape)
                action, _ = model.predict(observation, deterministic=True)
                self.assertTrue(env.action_space.contains(action))
                env.step(action)


if __name__ == "__main__":
    unittest.main()
