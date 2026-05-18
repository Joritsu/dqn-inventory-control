from pathlib import Path

import altair as alt
import pandas as pd
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
    ENVIRONMENT_CONFIGS,
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
    TRAINING_CONFIG_NAMES,
)
from inventory_env import InventoryEnv


TRAINING_RUNS_DIR = Path("models/training_runs")
BASE_STOCK_TARGET = 110


@st.cache_resource
def load_model(model_path):
    return DQN.load(model_path)


def model_exists(model_path):
    path = Path(model_path)
    model_file = path if path.suffix == ".zip" else Path(f"{model_path}.zip")
    return model_file.exists()


def model_is_compatible(model_path, env_config):
    try:
        model = DQN.load(str(model_path))
    except (FileNotFoundError, ValueError):
        return False

    return model.observation_space.shape == InventoryEnv(**env_config).observation_space.shape


def checkpoint_timesteps(checkpoint_path):
    name_parts = checkpoint_path.stem.split("_")

    if len(name_parts) < 2 or name_parts[-1] != "steps":
        return -1

    try:
        return int(name_parts[-2])
    except ValueError:
        return -1


def available_model_stages(config_name, env_config):
    model_stages = []
    run_dir = TRAINING_RUNS_DIR / config_name
    initial_model_path = run_dir / "initial_dqn_inventory"
    final_model_path = run_dir / "dqn_inventory"
    best_model_path = run_dir / "best_model" / "best_model"
    checkpoint_dir = run_dir / "checkpoints"

    if model_exists(initial_model_path) and model_is_compatible(initial_model_path, env_config):
        model_stages.append((f"{config_name}: initial policy", str(initial_model_path)))

    checkpoints = sorted(
        checkpoint_dir.glob("dqn_inventory_*_steps.zip"),
        key=checkpoint_timesteps,
    )
    for checkpoint in checkpoints:
        timesteps = checkpoint_timesteps(checkpoint)
        if timesteps >= 0 and model_is_compatible(str(checkpoint), env_config):
            model_stages.append((f"{config_name}: checkpoint {timesteps:,}", str(checkpoint)))

    if model_exists(final_model_path) and model_is_compatible(final_model_path, env_config):
        model_stages.append((f"{config_name}: final policy", str(final_model_path)))

    if model_exists(best_model_path) and model_is_compatible(best_model_path, env_config):
        model_stages.append((f"{config_name}: best policy", str(best_model_path)))

    return model_stages


def scenario_default(env_defaults, name, fallback):
    return env_defaults.get(name, fallback)


def scenario_key(config_name, name):
    return f"{config_name}_{name}"


def render_episode_chart(rows):
    if not rows:
        return

    chart_columns = {
        "inventory": "Inventory",
        "demand": "Demand",
        "expected_demand": "Expected demand",
        "order": "Order",
        "pending_orders": "Pending orders",
    }
    chart_data = pd.DataFrame(rows)
    long_chart_data = chart_data[["step", *chart_columns.keys()]].rename(columns=chart_columns).melt(
        id_vars="step",
        var_name="series",
        value_name="units",
    )
    series_selection = alt.selection_point(
        fields=["series"],
        bind="legend",
        toggle="true",
    )
    chart = (
        alt.Chart(long_chart_data)
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="Step"),
            y=alt.Y("units:Q", title="Units"),
            color=alt.Color("series:N", title="Series"),
            opacity=alt.condition(series_selection, alt.value(1.0), alt.value(0.15)),
            tooltip=[
                alt.Tooltip("step:Q", title="Step"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("units:Q", title="Value", format=",.2f"),
            ],
        )
        .add_params(series_selection)
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")


def run_policy(policy_name, env_config, model=None, policy=None, seed=123):
    env = InventoryEnv(**env_config)
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    total_cost = 0.0
    total_demand = 0
    total_sold = 0
    total_unmet_demand = 0
    total_inventory = 0
    rows = []

    for _ in range(env.episode_length):
        if policy is not None:
            action = policy(observation, env)
        elif model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        step_cost = info["ordering_cost"] + info["holding_cost"] + info["shortage_cost"]
        total_reward += reward
        total_cost += step_cost
        total_demand += info["demand"]
        total_sold += info["sold"]
        total_unmet_demand += info["unmet_demand"]
        total_inventory += info["inventory"]
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
                "ordering_cost": info["ordering_cost"],
                "holding_cost": info["holding_cost"],
                "shortage_cost": info["shortage_cost"],
                "total_step_cost": step_cost,
            }
        )

        if terminated or truncated:
            break

    steps = len(rows)
    service_level = total_sold / total_demand if total_demand > 0 else 0.0
    average_inventory = total_inventory / steps if steps > 0 else 0.0

    st.subheader(policy_name)
    top_metric_cols = st.columns(3)
    top_metric_cols[0].metric("Total reward", f"{total_reward:,.2f}")
    top_metric_cols[1].metric("Total cost", f"{total_cost:,.2f}")
    top_metric_cols[2].metric("Service level", f"{service_level:.2%}")

    bottom_metric_cols = st.columns(2)
    bottom_metric_cols[0].metric("Unmet demand", f"{total_unmet_demand:,}")
    bottom_metric_cols[1].metric("Average inventory", f"{average_inventory:,.2f}")
    render_episode_chart(rows)
    st.dataframe(rows, width="stretch")


def base_stock_policy(target_inventory, max_order):
    def policy(observation, env):
        current_inventory = int(round(observation[0] * env.max_inventory))
        pending_orders = env._pending_orders_total()
        inventory_position = current_inventory + pending_orders
        return max(0, min(max_order, target_inventory - inventory_position))

    return policy


st.set_page_config(page_title=APP_TITLE, layout="centered")

st.sidebar.header("Environment")

environment_names = [name for name in TRAINING_CONFIG_NAMES if name in ENVIRONMENT_CONFIGS]
selected_environment = st.sidebar.selectbox(
    "Environment config",
    environment_names,
    format_func=lambda name: name.replace("_", " ").title(),
)
selected_environment_definition = ENVIRONMENT_CONFIGS[selected_environment]
selected_environment_defaults = selected_environment_definition["env_config"]
st.sidebar.caption(selected_environment_definition["description"])

st.sidebar.header("Scenario")

episode_length = st.sidebar.slider(
    "Episode length",
    20,
    300,
    scenario_default(selected_environment_defaults, "episode_length", EPISODE_LENGTH),
    10,
    key=scenario_key(selected_environment, "episode_length"),
)
max_inventory = st.sidebar.slider(
    "Maximum inventory",
    10,
    300,
    scenario_default(selected_environment_defaults, "max_inventory", MAX_INVENTORY),
    5,
    key=scenario_key(selected_environment, "max_inventory"),
)
initial_inventory = st.sidebar.slider(
    "Initial inventory",
    0,
    max_inventory,
    min(scenario_default(selected_environment_defaults, "initial_inventory", INITIAL_INVENTORY), max_inventory),
    5,
    key=scenario_key(selected_environment, "initial_inventory"),
)
max_order = st.sidebar.slider(
    "Maximum order",
    1,
    150,
    scenario_default(selected_environment_defaults, "max_order", MAX_ORDER),
    1,
    key=scenario_key(selected_environment, "max_order"),
)
lead_time = st.sidebar.slider(
    "Delivery lead time",
    0,
    10,
    scenario_default(selected_environment_defaults, "lead_time", LEAD_TIME),
    1,
    key=scenario_key(selected_environment, "lead_time"),
)
demand_history_window = st.sidebar.slider(
    "Demand history window",
    1,
    20,
    scenario_default(selected_environment_defaults, "demand_history_window", DEMAND_HISTORY_WINDOW),
    1,
    key=scenario_key(selected_environment, "demand_history_window"),
)

st.sidebar.header("Demand")

mean_demand = st.sidebar.slider(
    "Mean demand",
    1,
    100,
    scenario_default(selected_environment_defaults, "mean_demand", MEAN_DEMAND),
    1,
    key=scenario_key(selected_environment, "mean_demand"),
)
demand_std = st.sidebar.slider(
    "Demand standard deviation",
    0,
    50,
    scenario_default(selected_environment_defaults, "demand_std", DEMAND_STD),
    1,
    key=scenario_key(selected_environment, "demand_std"),
)
demand_spike_chance = st.sidebar.slider(
    "Demand spike chance",
    0.0,
    1.0,
    float(scenario_default(selected_environment_defaults, "demand_spike_chance", DEMAND_SPIKE_CHANCE)),
    0.01,
    key=scenario_key(selected_environment, "demand_spike_chance"),
)
demand_spike_multiplier = st.sidebar.slider(
    "Demand spike multiplier",
    1.0,
    5.0,
    float(scenario_default(selected_environment_defaults, "demand_spike_multiplier", DEMAND_SPIKE_MULTIPLIER)),
    0.1,
    key=scenario_key(selected_environment, "demand_spike_multiplier"),
)
demand_trend_strength = st.sidebar.slider(
    "Demand trend strength",
    -0.8,
    1.5,
    float(scenario_default(selected_environment_defaults, "demand_trend_strength", DEMAND_TREND_STRENGTH)),
    0.05,
    key=scenario_key(selected_environment, "demand_trend_strength"),
)
demand_seasonal_amplitude = st.sidebar.slider(
    "Seasonal amplitude",
    0.0,
    0.8,
    float(scenario_default(selected_environment_defaults, "demand_seasonal_amplitude", DEMAND_SEASONAL_AMPLITUDE)),
    0.05,
    key=scenario_key(selected_environment, "demand_seasonal_amplitude"),
)
demand_seasonal_period = st.sidebar.slider(
    "Seasonal period",
    5,
    100,
    scenario_default(selected_environment_defaults, "demand_seasonal_period", DEMAND_SEASONAL_PERIOD),
    1,
    key=scenario_key(selected_environment, "demand_seasonal_period"),
)

st.sidebar.header("Costs")

sale_price = st.sidebar.number_input(
    "Sale price",
    min_value=0.0,
    value=float(scenario_default(selected_environment_defaults, "sale_price", SALE_PRICE)),
    step=0.5,
    key=scenario_key(selected_environment, "sale_price"),
)
order_cost = st.sidebar.number_input(
    "Fixed order cost",
    min_value=0.0,
    value=float(scenario_default(selected_environment_defaults, "order_cost", ORDER_COST)),
    step=0.5,
    key=scenario_key(selected_environment, "order_cost"),
)
holding_cost = st.sidebar.number_input(
    "Holding cost",
    min_value=0.0,
    value=float(scenario_default(selected_environment_defaults, "holding_cost", HOLDING_COST)),
    step=0.5,
    key=scenario_key(selected_environment, "holding_cost"),
)
shortage_cost = st.sidebar.number_input(
    "Shortage cost",
    min_value=0.0,
    value=float(scenario_default(selected_environment_defaults, "shortage_cost", SHORTAGE_COST)),
    step=0.5,
    key=scenario_key(selected_environment, "shortage_cost"),
)

base_stock_limit = max_inventory + max_order * max(1, lead_time)
base_stock_target = st.sidebar.slider(
    "Base-stock target",
    0,
    base_stock_limit,
    min(BASE_STOCK_TARGET, base_stock_limit),
    1,
)

env_config = dict(selected_environment_defaults)
env_config.update(
    {
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
)

st.title(APP_TITLE)
if APP_DESCRIPTION:
    st.write(APP_DESCRIPTION)

with st.expander("Scenario configuration"):
    st.write(f"Environment: **{selected_environment}**")
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
    if env_config.get("demand_regimes"):
        regime_steps = [regime.get("start_step", 0) for regime in env_config["demand_regimes"]]
        st.write(f"Demand regime starts: **{regime_steps}**")

    st.write(f"Sale price: **{sale_price}**")
    st.write(f"Fixed order cost: **{order_cost}**")
    st.write(f"Holding cost: **{holding_cost}**")
    st.write(f"Shortage cost: **{shortage_cost}**")

st.subheader("Policy simulation")

model_stages = available_model_stages(selected_environment, env_config)
selected_model_path = None

if model_stages:
    model_stage_labels = [label for label, _ in model_stages]
    preferred_labels = [
        f"{selected_environment}: best policy",
        f"{selected_environment}: final policy",
    ]
    default_stage_index = len(model_stage_labels) - 1
    for preferred_label in preferred_labels:
        if preferred_label in model_stage_labels:
            default_stage_index = model_stage_labels.index(preferred_label)
            break
    selected_model_label = st.selectbox(
        "DQN training stage",
        model_stage_labels,
        index=default_stage_index,
    )
    selected_model_path = dict(model_stages)[selected_model_label]
else:
    st.warning(f"No DQN models found for `{selected_environment}`. Run `python train_dqn.py {selected_environment}` first.")

random_col, dqn_col, base_stock_col = st.columns(3)

with random_col:
    run_random = st.button("Run random policy", width="stretch")

with dqn_col:
    run_dqn = st.button("Run selected DQN policy", width="stretch")

with base_stock_col:
    run_base_stock = st.button(f"Run base-stock S={base_stock_target}", width="stretch")

if run_random:
    run_policy("Random policy", env_config)

if run_dqn:
    if selected_model_path is None:
        st.error("No DQN model is available. Run `python train_dqn.py` first.")
    else:
        try:
            run_policy(f"DQN policy: {selected_model_label}", env_config, load_model(selected_model_path), seed=None)
        except FileNotFoundError:
            st.error("Selected model file was not found. Run `python train_dqn.py` again.")
        except ValueError:
            st.error("The selected model uses an incompatible observation shape. Run `python train_dqn.py` again.")

if run_base_stock:
    run_policy(
        f"Base-stock S={base_stock_target} policy",
        env_config,
        policy=base_stock_policy(base_stock_target, max_order),
        seed=None,
    )

st.success("Gymnasium inventory environment is ready.")
