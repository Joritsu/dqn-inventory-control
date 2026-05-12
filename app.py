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


def run_policy(policy_name, model=None, policy=None, seed=123):
    env = InventoryEnv()
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


def base_stock_policy(observation):
    current_inventory = int(observation[0])
    return max(0, min(MAX_ORDER, BASE_STOCK_TARGET - current_inventory))


st.set_page_config(page_title=APP_TITLE, layout="centered")

st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

st.subheader("Default configuration")

st.write(f"Episode length: **{EPISODE_LENGTH}**")
st.write(f"Maximum inventory: **{MAX_INVENTORY}**")
st.write(f"Initial inventory: **{INITIAL_INVENTORY}**")
st.write(f"Maximum order: **{MAX_ORDER}**")
st.write(f"Mean demand: **{MEAN_DEMAND}**")
st.write(f"Demand standard deviation: **{DEMAND_STD}**")
st.write(f"Demand spike chance: **{DEMAND_SPIKE_CHANCE:.0%}**")
st.write(f"Demand spike multiplier: **{DEMAND_SPIKE_MULTIPLIER}x**")

st.write(f"Sale price: **{SALE_PRICE}**")
st.write(f"Order cost: **{ORDER_COST}**")
st.write(f"Holding cost: **{HOLDING_COST}**")
st.write(f"Shortage cost: **{SHORTAGE_COST}**")

st.subheader("Policy simulation")

random_col, dqn_col, base_stock_col = st.columns(3)

with random_col:
    run_random = st.button("Run random policy", width="stretch")

with dqn_col:
    run_dqn = st.button("Run trained DQN policy", width="stretch")

with base_stock_col:
    run_base_stock = st.button("Run base-stock S=30", width="stretch")

if run_random:
    run_policy("Random policy")

if run_dqn:
    try:
        run_policy("Trained DQN policy", load_model(), seed=None)
    except FileNotFoundError:
        st.error("No trained model found. Run `python train_dqn.py` first.")

if run_base_stock:
    run_policy("Base-stock S=30 policy", policy=base_stock_policy, seed=None)

st.success("Gymnasium inventory environment is ready.")
