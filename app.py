import streamlit as st
from stable_baselines3 import DQN

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
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
from inventory_env import InventoryEnv


MODEL_PATH = "models/dqn_inventory_trend"
BASE_STOCK_TARGET = 110


@st.cache_resource
def load_model():
    return DQN.load(MODEL_PATH)


def run_policy(policy_name, env_config, model=None, policy=None, seed=123):
    env = InventoryEnv(**env_config)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    rows = []

    for _ in range(env.episode_length):
        if policy is not None:
            action = policy(observation, env)
        elif model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rows.append(
            {
                "step": info["step"],
                "inventory": info["inventory"],
                "order": info["order_quantity"],
                "arriving_order": info["arriving_order"],
                "pending_orders": info["pending_orders_total"],
                "demand": info["demand"],
                "expected_demand": info["expected_demand"],
                "demand_spike": info["demand_spike"],
                "sold": info["sold"],
                "unmet_demand": info["unmet_demand"],
                "reward": reward,
            }
        )

        if terminated or truncated:
            break

    st.subheader(policy_name)
    st.metric("Total reward", f"{total_reward:.2f}")
    st.line_chart(
        rows,
        x="step",
        y=["inventory", "demand", "expected_demand", "order", "pending_orders"],
    )
    st.dataframe(rows, width="stretch")


def base_stock_policy(target_inventory, max_order):
    def policy(observation, env):
        current_inventory = int(round(observation[0] * env.max_inventory))
        pending_orders = int(round(observation[4] * env._pending_order_scale()))
        inventory_position = current_inventory + pending_orders
        return max(0, min(max_order, target_inventory - inventory_position))

    return policy


st.set_page_config(page_title=APP_TITLE, layout="centered")

st.sidebar.header("Scenario")

episode_length = st.sidebar.slider("Episode length", 20, 300, EPISODE_LENGTH, 10)
max_inventory = st.sidebar.slider("Maximum inventory", 10, 300, MAX_INVENTORY, 5)
initial_inventory = st.sidebar.slider(
    "Initial inventory",
    0,
    max_inventory,
    min(INITIAL_INVENTORY, max_inventory),
    5,
)
max_order = st.sidebar.slider("Maximum order", 1, 150, MAX_ORDER, 1)
lead_time = st.sidebar.slider("Delivery lead time", 0, 10, LEAD_TIME, 1)
demand_history_window = st.sidebar.slider(
    "Demand history window",
    1,
    20,
    DEMAND_HISTORY_WINDOW,
    1,
)

st.sidebar.header("Demand")

mean_demand = st.sidebar.slider("Mean demand", 1, 100, MEAN_DEMAND, 1)
demand_std = st.sidebar.slider("Demand standard deviation", 0, 50, DEMAND_STD, 1)
demand_spike_chance = st.sidebar.slider(
    "Demand spike chance",
    0.0,
    1.0,
    float(DEMAND_SPIKE_CHANCE),
    0.01,
)
demand_spike_multiplier = st.sidebar.slider(
    "Demand spike multiplier",
    1.0,
    5.0,
    float(DEMAND_SPIKE_MULTIPLIER),
    0.1,
)
demand_trend_strength = st.sidebar.slider(
    "Demand trend strength",
    -0.8,
    1.5,
    float(DEMAND_TREND_STRENGTH),
    0.05,
)
demand_seasonal_amplitude = st.sidebar.slider(
    "Seasonal amplitude",
    0.0,
    0.8,
    float(DEMAND_SEASONAL_AMPLITUDE),
    0.05,
)
demand_seasonal_period = st.sidebar.slider(
    "Seasonal period",
    5,
    100,
    DEMAND_SEASONAL_PERIOD,
    1,
)

st.sidebar.header("Costs")

sale_price = st.sidebar.number_input("Sale price", min_value=0.0, value=float(SALE_PRICE), step=0.5)
order_cost = st.sidebar.number_input(
    "Fixed order cost",
    min_value=0.0,
    value=float(ORDER_COST),
    step=0.5,
)
holding_cost = st.sidebar.number_input(
    "Holding cost",
    min_value=0.0,
    value=float(HOLDING_COST),
    step=0.5,
)
shortage_cost = st.sidebar.number_input(
    "Shortage cost",
    min_value=0.0,
    value=float(SHORTAGE_COST),
    step=0.5,
)

base_stock_limit = max_inventory + max_order * max(1, lead_time)
base_stock_target = st.sidebar.slider(
    "Base-stock target",
    0,
    base_stock_limit,
    min(BASE_STOCK_TARGET, base_stock_limit),
    1,
)

env_config = {
    "episode_length": episode_length,
    "max_inventory": max_inventory,
    "initial_inventory": initial_inventory,
    "max_order": max_order,
    "lead_time": lead_time,
    "demand_history_window": demand_history_window,
    "mean_demand": mean_demand,
    "demand_std": demand_std,
    "demand_trend_strength": demand_trend_strength,
    "demand_seasonal_amplitude": demand_seasonal_amplitude,
    "demand_seasonal_period": demand_seasonal_period,
    "demand_spike_chance": demand_spike_chance,
    "demand_spike_multiplier": demand_spike_multiplier,
    "sale_price": sale_price,
    "order_cost": order_cost,
    "holding_cost": holding_cost,
    "shortage_cost": shortage_cost,
}

st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

st.subheader("Scenario configuration")

st.write(f"Episode length: **{episode_length}**")
st.write(f"Maximum inventory: **{max_inventory}**")
st.write(f"Initial inventory: **{initial_inventory}**")
st.write(f"Maximum order: **{max_order}**")
st.write(f"Delivery lead time: **{lead_time}**")
st.write(f"Demand history window: **{demand_history_window}**")
st.write(f"Mean demand: **{mean_demand}**")
st.write(f"Demand standard deviation: **{demand_std}**")
st.write(f"Demand trend strength: **{demand_trend_strength:.2f}**")
st.write(f"Seasonal amplitude: **{demand_seasonal_amplitude:.2f}**")
st.write(f"Seasonal period: **{demand_seasonal_period}**")
st.write(f"Demand spike chance: **{demand_spike_chance:.0%}**")
st.write(f"Demand spike multiplier: **{demand_spike_multiplier}x**")

st.write(f"Sale price: **{sale_price}**")
st.write(f"Fixed order cost: **{order_cost}**")
st.write(f"Holding cost: **{holding_cost}**")
st.write(f"Shortage cost: **{shortage_cost}**")

st.subheader("Policy simulation")

random_col, dqn_col, base_stock_col = st.columns(3)

with random_col:
    run_random = st.button("Run random policy", width="stretch")

with dqn_col:
    run_dqn = st.button("Run trained DQN policy", width="stretch")

with base_stock_col:
    run_base_stock = st.button(f"Run base-stock S={base_stock_target}", width="stretch")

if run_random:
    run_policy("Random policy", env_config)

if run_dqn:
    try:
        run_policy("Trained DQN policy", env_config, load_model(), seed=None)
    except FileNotFoundError:
        st.error("No trained model found. Run `python train_dqn.py` first.")
    except ValueError:
        st.error("The trained model uses the old observation shape. Run `python train_dqn.py` again.")

if run_base_stock:
    run_policy(
        f"Base-stock S={base_stock_target} policy",
        env_config,
        policy=base_stock_policy(base_stock_target, max_order),
        seed=None,
    )

st.success("Gymnasium inventory environment is ready.")
