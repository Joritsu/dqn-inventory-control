import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    DEMAND_SPIKE_CHANCE,
    DEMAND_SPIKE_MULTIPLIER,
    DEMAND_STD,
    EPISODE_LENGTH,
    HOLDING_COST,
    INITIAL_INVENTORY,
    MAX_INVENTORY,
    MAX_ORDER,
    MEAN_DEMAND,
    ORDER_COST,
    SALE_PRICE,
    SHORTAGE_COST,
)


class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        episode_length=EPISODE_LENGTH,
        max_inventory=MAX_INVENTORY,
        initial_inventory=INITIAL_INVENTORY,
        max_order=MAX_ORDER,
        mean_demand=MEAN_DEMAND,
        demand_std=DEMAND_STD,
        demand_spike_chance=DEMAND_SPIKE_CHANCE,
        demand_spike_multiplier=DEMAND_SPIKE_MULTIPLIER,
        sale_price=SALE_PRICE,
        order_cost=ORDER_COST,
        holding_cost=HOLDING_COST,
        shortage_cost=SHORTAGE_COST,
    ):
        super().__init__()

        self.episode_length = episode_length
        self.max_inventory = max_inventory
        self.initial_inventory = initial_inventory
        self.max_order = max_order
        self.mean_demand = mean_demand
        self.demand_std = demand_std
        self.demand_spike_chance = demand_spike_chance
        self.demand_spike_multiplier = demand_spike_multiplier
        self.sale_price = sale_price
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

        self.action_space = spaces.Discrete(self.max_order + 1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.max_inventory, self.episode_length], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.inventory = self.initial_inventory

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.initial_inventory
        return self._get_observation(), self._get_info()

    def step(self, action):
        order_quantity = int(action)
        order_quantity = int(np.clip(order_quantity, 0, self.max_order))

        self.inventory = min(self.max_inventory, self.inventory + order_quantity)
        demand, demand_spike = self._sample_demand()
        sold = min(self.inventory, demand)
        unmet_demand = max(0, demand - self.inventory)
        self.inventory -= sold

        revenue = sold * self.sale_price
        ordering_cost = self.order_cost if order_quantity > 0 else 0.0
        holding_cost = self.inventory * self.holding_cost
        shortage_cost = unmet_demand * self.shortage_cost
        reward = revenue - ordering_cost - holding_cost - shortage_cost

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length

        info = self._get_info()
        info.update(
            {
                "demand": demand,
                "demand_spike": demand_spike,
                "sold": sold,
                "unmet_demand": unmet_demand,
                "order_quantity": order_quantity,
                "revenue": revenue,
                "ordering_cost": ordering_cost,
                "holding_cost": holding_cost,
                "shortage_cost": shortage_cost,
            }
        )

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, inventory: {self.inventory}")

    def _sample_demand(self):
        demand = self.np_random.normal(self.mean_demand, self.demand_std)
        demand_spike = self.np_random.random() < self.demand_spike_chance

        if demand_spike:
            demand *= self.demand_spike_multiplier

        return max(0, int(round(demand))), bool(demand_spike)

    def _get_observation(self):
        return np.array([self.inventory, self.current_step], dtype=np.float32)

    def _get_info(self):
        return {
            "inventory": self.inventory,
            "step": self.current_step,
        }
