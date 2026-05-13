import streamlit as st
from stable_baselines3 import DQN

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
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
from inventory_env import InventoryEnv


MODEL_PATH = "models/dqn_inventory"
BASE_STOCK_TARGET = 30


@st.cache_resource
def load_model():
    return DQN.load(MODEL_PATH)


def run_policy(policy_name, env_config, model=None, policy=None, seed=123):
    env = InventoryEnv(**env_config)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    rows = []

    for _ in range(EPISODE_LENGTH):
        if policy is not None:
            action = policy(observation)
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
                "demand": info["demand"],
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
    st.line_chart(rows, x="step", y=["inventory", "demand", "order"])
    st.dataframe(rows, width="stretch")


def base_stock_policy(target_inventory, max_order):
    def policy(observation):
        current_inventory = int(observation[0])
        return max(0, min(max_order, target_inventory - current_inventory))

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
base_stock_target = st.sidebar.slider(
    "Base-stock target",
    0,
    max_inventory,
    min(BASE_STOCK_TARGET, max_inventory),
    1,
)

env_config = {
    "episode_length": episode_length,
    "max_inventory": max_inventory,
    "initial_inventory": initial_inventory,
    "max_order": max_order,
    "mean_demand": mean_demand,
    "demand_std": demand_std,
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
st.write(f"Mean demand: **{mean_demand}**")
st.write(f"Demand standard deviation: **{demand_std}**")
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

if run_base_stock:
    run_policy(
        f"Base-stock S={base_stock_target} policy",
        env_config,
        policy=base_stock_policy(base_stock_target, max_order),
        seed=None,
    )

st.success("Gymnasium inventory environment is ready.")
