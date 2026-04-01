"""
Train / test / run a PPO agent in the CPP GridWorld environment.

Usage:
    python train_grid_world_cpp.py train   – train and save the model
    python train_grid_world_cpp.py test    – evaluate coverage over 100 episodes
    python train_grid_world_cpp.py run     – run one episode with rendering
"""

import sys
import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime

def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")

if len(sys.argv) < 2 or sys.argv[1] not in ["train", "test", "run"]:
    print("Usage: python train_grid_world_cpp.py <train|test|run>")
    sys.exit(1)

mode = sys.argv[1]

try:
    gym.register(
        id="gymnasium_env/GridWorldCPP-v1",
        entry_point=GridWorldCPPEnv,
    )
except Exception:
    pass

# --- Hyperparameters ---
DIM = 5
OBSTACLES = 3
MAX_STEPS = 200
TOTAL_TIMESTEPS = 500_000
ENTROPY_COEF = 0.02
# -----------------------

if mode == "train":
    print("--- Starting Training (CPP) ---")
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}"
    model_path = f"data/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip"

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == "run":
    model_name = input("Enter model filename (without .zip, e.g. ppo_cpp_5_3_200_0.02_20260101_120000): ")
    model_path = f"data/{model_name}.zip"
    print(f"--- Loading {model_path} ---")

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human",
    )

    obs, info = env.reset()
    done = truncated = False
    step = 0
    total_reward = 0.0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        step += 1
        print(
            f"Step {step:3d} | {print_action(int(action)):5s} | "
            f"reward={reward:+.2f} | coverage={info['coverage_ratio']:.0%}"
        )

    print(f"--- Run Finished | steps={step} | total_reward={total_reward:.2f} | "
          f"coverage={info['coverage_ratio']:.0%} ---")
    env.close()

elif mode == "test":
    model_name = input("Enter model filename (without .zip, e.g. ppo_cpp_5_3_200_0.02_20260101_120000): ")
    model_path = f"data/{model_name}.zip"
    print(f"--- Loading {model_path} for testing ---")

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    NUM_EPISODES = 100
    full_coverage_count = 0
    coverage_ratios = []

    for i in range(NUM_EPISODES):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            steps += 1

        coverage_ratios.append(info["coverage_ratio"])
        if done and not truncated:
            full_coverage_count += 1
            print(f"Episode {i+1:3d}: Full coverage in {steps} steps.")
        else:
            print(f"Episode {i+1:3d}: Truncated. Coverage={info['coverage_ratio']:.0%} in {steps} steps.")

    print("--- Test Finished ---")
    print(f"Full coverage rate : {full_coverage_count}/{NUM_EPISODES} "
          f"({100 * full_coverage_count / NUM_EPISODES:.1f}%)")
    print(f"Mean coverage ratio: {sum(coverage_ratios) / len(coverage_ratios):.2%}")
    env.close()
