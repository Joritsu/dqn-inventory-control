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
    PENDING_ORDER_OBSERVATION_SLOTS,
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
        pending_order_observation_slots=PENDING_ORDER_OBSERVATION_SLOTS,
        demand_regimes=None,
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
        self.pending_order_observation_slots = max(0, int(pending_order_observation_slots))
        self.demand_regimes = self._prepare_demand_regimes(demand_regimes)

        self.action_space = spaces.Discrete(self.max_order + 1)
        observation_size = 7 + self.pending_order_observation_slots
        self.observation_space = spaces.Box(
            low=np.zeros(observation_size, dtype=np.float32),
            high=np.ones(observation_size, dtype=np.float32),
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
        regime = self._active_demand_regime()
        demand_std = regime.get("demand_std", self.demand_std)
        demand_spike_chance = regime.get("demand_spike_chance", self.demand_spike_chance)
        demand_spike_multiplier = regime.get("demand_spike_multiplier", self.demand_spike_multiplier)

        demand = self.np_random.normal(self._expected_demand(), demand_std)
        demand_spike = self.np_random.random() < demand_spike_chance

        if demand_spike:
            demand *= demand_spike_multiplier

        return max(0, int(round(demand))), bool(demand_spike)

    def _expected_demand(self):
        regime = self._active_demand_regime()
        mean_demand = regime.get("mean_demand", self.mean_demand)
        demand_trend_strength = regime.get("demand_trend_strength", self.demand_trend_strength)
        demand_seasonal_amplitude = regime.get("demand_seasonal_amplitude", self.demand_seasonal_amplitude)
        demand_seasonal_period = max(1, int(regime.get("demand_seasonal_period", self.demand_seasonal_period)))

        progress = self.current_step / max(1, self.episode_length - 1)
        trend_multiplier = max(0.0, 1.0 + demand_trend_strength * progress)
        seasonal_phase = 2 * np.pi * self.current_step / demand_seasonal_period
        seasonal_multiplier = 1.0 + demand_seasonal_amplitude * np.sin(seasonal_phase)
        return max(0.0, mean_demand * trend_multiplier * seasonal_multiplier)

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

    def _prepare_demand_regimes(self, demand_regimes):
        if not demand_regimes:
            return []

        return sorted(
            [dict(regime) for regime in demand_regimes],
            key=lambda regime: int(regime.get("start_step", 0)),
        )

    def _active_demand_regime(self):
        active_regime = {}

        for regime in self.demand_regimes:
            if self.current_step >= int(regime.get("start_step", 0)):
                active_regime = regime
            else:
                break

        return active_regime

    def _normalized_pending_order_pipeline(self):
        pipeline = list(self.pending_orders[: self.pending_order_observation_slots])
        missing_slots = self.pending_order_observation_slots - len(pipeline)

        if missing_slots > 0:
            pipeline.extend([0] * missing_slots)

        return [
            self._normalize(order_quantity, self.max_order)
            for order_quantity in pipeline
        ]

    def _seasonal_features(self):
        regime = self._active_demand_regime()
        demand_seasonal_period = max(1, int(regime.get("demand_seasonal_period", self.demand_seasonal_period)))
        seasonal_phase = 2 * np.pi * self.current_step / demand_seasonal_period
        seasonal_sin = (np.sin(seasonal_phase) + 1.0) / 2.0
        seasonal_cos = (np.cos(seasonal_phase) + 1.0) / 2.0
        return [float(seasonal_sin), float(seasonal_cos)]

    def _demand_scale(self):
        demand_profiles = [
            {
                "mean_demand": self.mean_demand,
                "demand_std": self.demand_std,
                "demand_trend_strength": self.demand_trend_strength,
                "demand_seasonal_amplitude": self.demand_seasonal_amplitude,
                "demand_spike_multiplier": self.demand_spike_multiplier,
            }
        ]
        demand_profiles.extend(self.demand_regimes)
        expected_spike_demand = 0.0

        for profile in demand_profiles:
            trend_strength = profile.get("demand_trend_strength", self.demand_trend_strength)
            trend_floor = min(1.0, 1.0 + trend_strength)
            trend_ceiling = max(1.0, 1.0 + trend_strength)
            max_trend_multiplier = max(abs(trend_floor), abs(trend_ceiling))
            max_seasonal_multiplier = 1.0 + abs(
                profile.get("demand_seasonal_amplitude", self.demand_seasonal_amplitude)
            )
            expected_peak_demand = (
                profile.get("mean_demand", self.mean_demand)
                * max_trend_multiplier
                * max_seasonal_multiplier
            )
            profile_spike_demand = (
                expected_peak_demand + 3 * profile.get("demand_std", self.demand_std)
            ) * profile.get("demand_spike_multiplier", self.demand_spike_multiplier)
            expected_spike_demand = max(expected_spike_demand, profile_spike_demand)

        return max(1.0, self.max_inventory, self.max_order, expected_spike_demand)

    def _pending_order_scale(self):
        return max(1.0, self.max_order * max(1, self.lead_time))

    def _normalize(self, value, scale):
        return float(np.clip(value / scale, 0.0, 1.0))

    def _get_observation(self):
        observation = [
            self._normalize(self.inventory, self.max_inventory),
            self._normalize(self.current_step, self.episode_length),
            self._normalize(self._last_demand(), self._demand_scale()),
            self._normalize(self._average_recent_demand(), self._demand_scale()),
            self._normalize(self._pending_orders_total(), self._pending_order_scale()),
        ]
        observation.extend(self._normalized_pending_order_pipeline())
        observation.extend(self._seasonal_features())
        return np.array(observation, dtype=np.float32)

    def _get_info(self):
        seasonal_sin, seasonal_cos = self._seasonal_features()
        active_regime = self._active_demand_regime()
        return {
            "inventory": self.inventory,
            "step": self.current_step,
            "last_demand": self._last_demand(),
            "average_recent_demand": self._average_recent_demand(),
            "expected_demand": self._expected_demand(),
            "pending_orders": list(self.pending_orders),
            "pending_orders_total": self._pending_orders_total(),
            "pending_order_pipeline": self._normalized_pending_order_pipeline(),
            "seasonal_sin": seasonal_sin,
            "seasonal_cos": seasonal_cos,
            "demand_regime_start_step": active_regime.get("start_step"),
        }
