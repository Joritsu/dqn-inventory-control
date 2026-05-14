import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    DEMAND_SPIKE_CHANCE,
    DEMAND_SPIKE_MULTIPLIER,
    DEMAND_HISTORY_WINDOW,
    DEMAND_SEASONAL_AMPLITUDE,
    DEMAND_SEASONAL_PERIOD,
    DEMAND_STD,
    DEMAND_TREND_STRENGTH,
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
        demand_trend_strength=DEMAND_TREND_STRENGTH,
        demand_seasonal_amplitude=DEMAND_SEASONAL_AMPLITUDE,
        demand_seasonal_period=DEMAND_SEASONAL_PERIOD,
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
        self.demand_trend_strength = demand_trend_strength
        self.demand_seasonal_amplitude = demand_seasonal_amplitude
        self.demand_seasonal_period = max(1, int(demand_seasonal_period))
        self.demand_spike_chance = demand_spike_chance
        self.demand_spike_multiplier = demand_spike_multiplier
        self.sale_price = sale_price
        self.order_cost = order_cost
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

        self.action_space = spaces.Discrete(self.max_order + 1)
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
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
                "expected_demand": self._expected_demand(),
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
        demand = self.np_random.normal(self._expected_demand(), self.demand_std)
        demand_spike = self.np_random.random() < self.demand_spike_chance

        if demand_spike:
            demand *= self.demand_spike_multiplier

        return max(0, int(round(demand))), bool(demand_spike)

    def _expected_demand(self):
        progress = self.current_step / max(1, self.episode_length - 1)
        trend_multiplier = max(0.0, 1.0 + self.demand_trend_strength * progress)
        seasonal_phase = 2 * np.pi * self.current_step / self.demand_seasonal_period
        seasonal_multiplier = 1.0 + self.demand_seasonal_amplitude * np.sin(seasonal_phase)
        return max(0.0, self.mean_demand * trend_multiplier * seasonal_multiplier)

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

    def _demand_scale(self):
        trend_floor = min(1.0, 1.0 + self.demand_trend_strength)
        trend_ceiling = max(1.0, 1.0 + self.demand_trend_strength)
        max_trend_multiplier = max(abs(trend_floor), abs(trend_ceiling))
        max_seasonal_multiplier = 1.0 + abs(self.demand_seasonal_amplitude)
        expected_peak_demand = self.mean_demand * max_trend_multiplier * max_seasonal_multiplier
        expected_spike_demand = (expected_peak_demand + 3 * self.demand_std) * self.demand_spike_multiplier
        return max(1.0, self.max_inventory, self.max_order, expected_spike_demand)

    def _pending_order_scale(self):
        return max(1.0, self.max_order * max(1, self.lead_time))

    def _normalize(self, value, scale):
        return float(np.clip(value / scale, 0.0, 1.0))

    def _get_observation(self):
        return np.array(
            [
                self._normalize(self.inventory, self.max_inventory),
                self._normalize(self.current_step, self.episode_length),
                self._normalize(self._last_demand(), self._demand_scale()),
                self._normalize(self._average_recent_demand(), self._demand_scale()),
                self._normalize(self._pending_orders_total(), self._pending_order_scale()),
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "inventory": self.inventory,
            "step": self.current_step,
            "last_demand": self._last_demand(),
            "average_recent_demand": self._average_recent_demand(),
            "expected_demand": self._expected_demand(),
            "pending_orders": list(self.pending_orders),
            "pending_orders_total": self._pending_orders_total(),
        }
