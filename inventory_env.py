import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    DEMAND_SPIKE_CHANCE,
    DEMAND_SPIKE_MULTIPLIER,
    DEMAND_HISTORY_WINDOW,
    DEMAND_STD,
    EPISODE_LENGTH,
    HOLDING_COST,
    INITIAL_INVENTORY,
    LEAD_TIME,
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
        lead_time=LEAD_TIME,
        demand_history_window=DEMAND_HISTORY_WINDOW,
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
        self.lead_time = max(0, int(lead_time))
        self.demand_history_window = max(1, int(demand_history_window))
        self.mean_demand = mean_demand
        self.demand_std = demand_std
        self.demand_spike_chance = demand_spike_chance
        self.demand_spike_multiplier = demand_spike_multiplier
        self.sale_price = sale_price
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

        self.action_space = spaces.Discrete(self.max_order + 1)
        max_observed_demand = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [
                    self.max_inventory,
                    self.episode_length,
                    max_observed_demand,
                    max_observed_demand,
                    self.max_order * max(1, self.lead_time),
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.current_step = 0
        self.inventory = self.initial_inventory
        self.pending_orders = []
        self.demand_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.pending_orders = [0] * self.lead_time
        self.demand_history = []
        return self._get_observation(), self._get_info()

    def step(self, action):
        order_quantity = int(action)
        order_quantity = int(np.clip(order_quantity, 0, self.max_order))

        arriving_order = self._receive_arriving_order()

        if self.lead_time == 0:
            arriving_order += order_quantity
        else:
            self.pending_orders.append(order_quantity)

        self.inventory = min(self.max_inventory, self.inventory + arriving_order)
        demand, demand_spike = self._sample_demand()
        sold = min(self.inventory, demand)
        unmet_demand = max(0, demand - self.inventory)
        self.inventory -= sold
        self._record_demand(demand)

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
                "arriving_order": arriving_order,
                "pending_orders": list(self.pending_orders),
                "pending_orders_total": self._pending_orders_total(),
                "revenue": revenue,
                "ordering_cost": ordering_cost,
                "holding_cost": holding_cost,
                "shortage_cost": shortage_cost,
            }
        )

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"Step: {self.current_step}, inventory: {self.inventory}, "
            f"pending orders: {self.pending_orders}"
        )

    def _sample_demand(self):
        demand = self.np_random.normal(self.mean_demand, self.demand_std)
        demand_spike = self.np_random.random() < self.demand_spike_chance

        if demand_spike:
            demand *= self.demand_spike_multiplier

        return max(0, int(round(demand))), bool(demand_spike)

    def _receive_arriving_order(self):
        if self.lead_time == 0:
            return 0

        return self.pending_orders.pop(0)

    def _record_demand(self, demand):
        self.demand_history.append(demand)
        self.demand_history = self.demand_history[-self.demand_history_window :]

    def _last_demand(self):
        if not self.demand_history:
            return 0

        return self.demand_history[-1]

    def _average_recent_demand(self):
        if not self.demand_history:
            return 0.0

        return float(np.mean(self.demand_history))

    def _pending_orders_total(self):
        return int(sum(self.pending_orders))

    def _get_observation(self):
        return np.array(
            [
                self.inventory,
                self.current_step,
                self._last_demand(),
                self._average_recent_demand(),
                self._pending_orders_total(),
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "inventory": self.inventory,
            "step": self.current_step,
            "last_demand": self._last_demand(),
            "average_recent_demand": self._average_recent_demand(),
            "pending_orders": list(self.pending_orders),
            "pending_orders_total": self._pending_orders_total(),
        }
