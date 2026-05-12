from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_checker import check_env

from inventory_env import InventoryEnv


MODEL_PATH = "models/dqn_inventory"
BEST_MODEL_DIR = "models/best_dqn_inventory"
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 10
PATIENCE_EVALUATIONS = 8


def main():
    env = InventoryEnv()
    eval_env = InventoryEnv()
    check_env(env, warn=True)

    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE_EVALUATIONS,
        min_evals=5,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path="models/evaluations",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        callback_after_eval=early_stop_callback,
        verbose=1,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.97,
        exploration_fraction=0.40,
        exploration_final_eps=0.02,
        verbose=1,
        seed=123,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}.zip")
    print(f"Best evaluation model saved in {BEST_MODEL_DIR}.")


if __name__ == "__main__":
    main()
