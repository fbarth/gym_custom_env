#
# python train_grid_world_cpp.py <train|test|run>
#

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import sys

def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")

if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test', 'run']:
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
MAX_STEPS = 400       # mais folgado: dá margem ao agente durante treino
TOTAL_TIMESTEPS = 3_000_000  # CPP é mais difícil que goal-reaching
ENTROPY_COEF = 0.05   # mais exploração
# -----------------------

if mode == 'train':
    print("--- Starting Training ---")
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ENTROPY_COEF, device="cpu",
                n_steps=2048, batch_size=64, n_epochs=10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'log/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}'
    model_path = f'data/ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'run':
    model_name = input("Enter model filename (without .zip): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for a run ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human"
    )

    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0.0

    print(f"Free cells to cover: {info['free_cells']}")
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1
        print(
            f"Step {steps:>3} | Action: {print_action(action.item()):<5} | "
            f"Reward: {reward:+.2f} | Coverage: {info['covered_cells']}/{info['free_cells']} "
            f"({info['coverage_ratio']*100:.1f}%)"
        )

    env.close()
    print("\n--- Run Finished ---")
    if done:
        print(f"SUCCESS – full coverage in {steps} steps!")
    else:
        print(f"TIMEOUT – {steps} steps, coverage {info['coverage_ratio']*100:.1f}%")
    print(f"Total reward: {total_reward:.2f}")

elif mode == 'test':
    model_name = input("Enter model filename (without .zip): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for testing ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v1",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )

    num_episodes = 100
    success_count = 0
    total_coverages = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1

        total_coverages.append(info['coverage_ratio'])

        if done and not truncated:
            success_count += 1
            print(f"Episode {i+1:>3}: SUCCESS in {steps:>3} steps | Coverage: 100.0%")
        else:
            print(f"Episode {i+1:>3}: TIMEOUT in {steps:>3} steps | Coverage: {info['coverage_ratio']*100:.1f}%")

    success_rate = (success_count / num_episodes) * 100
    avg_coverage = sum(total_coverages) / len(total_coverages) * 100

    print(f"\n--- Test Finished ---")
    print(f"Success Rate:    {success_rate:.2f}% ({success_count}/{num_episodes})")
    print(f"Avg Coverage:    {avg_coverage:.2f}%")
