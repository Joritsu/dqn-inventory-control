APP_TITLE = "Adaptive Inventory Management System"
APP_DESCRIPTION = ""

EPISODE_LENGTH = 100
MAX_INVENTORY = 100
INITIAL_INVENTORY = 50

MAX_ORDER = 50
LEAD_TIME = 2
PENDING_ORDER_OBSERVATION_SLOTS = 10
DEMAND_HISTORY_WINDOW = 5
MEAN_DEMAND = 20
DEMAND_STD = 8
DEMAND_TREND_STRENGTH = 0.50
DEMAND_SEASONAL_AMPLITUDE = 0.25
DEMAND_SEASONAL_PERIOD = 25
DEMAND_SPIKE_CHANCE = 0.05
DEMAND_SPIKE_MULTIPLIER = 1.8
SALE_PRICE = 8.0
ORDER_COST = 4.0
HOLDING_COST = 1.0
SHORTAGE_COST = 10.0


ENVIRONMENT_CONFIGS = {
    "baseline": {
        "description": "Default demand trend, seasonality, spikes, and cost settings.",
        "env_config": {},
    },
    "adaptive_regime_shift": {
        "description": (
            "Demand starts low, jumps to a high-demand regime, then enters a temporary "
            "spike-heavy season. The agent must infer the regime from recent demand."
        ),
        "env_config": {
            "episode_length": 120,
            "max_inventory": 140,
            "initial_inventory": 60,
            "max_order": 70,
            "lead_time": 3,
            "mean_demand": 16,
            "demand_std": 5,
            "demand_trend_strength": 0.0,
            "demand_seasonal_amplitude": 0.15,
            "demand_seasonal_period": 30,
            "demand_spike_chance": 0.02,
            "demand_spike_multiplier": 1.8,
            "shortage_cost": 14.0,
            "demand_regimes": [
                {
                    "start_step": 0,
                    "mean_demand": 12,
                    "demand_std": 4,
                    "demand_spike_chance": 0.01,
                    "demand_spike_multiplier": 1.6,
                },
                {
                    "start_step": 35,
                    "mean_demand": 32,
                    "demand_std": 7,
                    "demand_spike_chance": 0.04,
                    "demand_spike_multiplier": 1.8,
                },
                {
                    "start_step": 75,
                    "mean_demand": 18,
                    "demand_std": 6,
                    "demand_spike_chance": 0.12,
                    "demand_spike_multiplier": 2.5,
                },
            ],
        },
    },
}

TRAINING_CONFIG_NAMES = ["baseline", "adaptive_regime_shift"]
