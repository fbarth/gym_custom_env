#
# python train_grid_world_cpp.py <train|test|run>
#

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime
import sys

def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")

if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test', 'run']:
    print("Usage: python train_grid_world_cpp.py <train|test|run>")
    sys.exit(1)

mode = sys.argv[1]

try:
    gym.register(
        id="gymnasium_env/GridWorld-CPP-v0",
        entry_point=GridWorldCPPEnv,
    )
except Exception:
    pass

# --- Hyperparameters ---
GRID_SIZE = 5
MAX_STEPS = 100
TOTAL_TIMESTEPS = 50_000
ENTROPY_COEF = 0.02
LEARNING_RATE = 0.0003
# -----------------------

if mode == 'train':
    print("--- Starting Training for CPP ---")
    env = gym.make(
        "gymnasium_env/GridWorld-CPP-v0",
        size=GRID_SIZE,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    env = FlattenObservation(env)
    check_env(env)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        ent_coef=ENTROPY_COEF,
        learning_rate=LEARNING_RATE,
        device="cpu"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'log/ppo_cpp_{GRID_SIZE}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}'
    model_path = f'data/ppo_cpp_{GRID_SIZE}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip'

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Max steps per episode: {MAX_STEPS}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'run':
    model_name = input("Enter model filename (e.g., ppo_cpp_10_500_0.02_20250324_120000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for a run ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorld-CPP-v0",
        size=GRID_SIZE,
        max_steps=MAX_STEPS,
        render_mode="human"
    )
    env = FlattenObservation(env)

    (obs, _) = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        print(f"Step: {steps+1:3d}, Action: {print_action(action.item())}, Reward: {reward:7.2f}, Coverage: {info['coverage']:5.1f}%, Done: {done}, Truncated: {truncated}")
        steps += 1
    
    print("--- Run Finished ---")
    print(f"Total Steps: {steps}")
    print(f"Final Coverage: {info['coverage']:.1f}%")
    print(f"Total Reward: {total_reward:.2f}")

elif mode == 'test':
    model_name = input("Enter model filename (e.g., ppo_cpp_10_500_0.02_20250324_120000): ")
    model_path = f'data/{model_name}.zip'
    print(f'--- Loading model from {model_path} for testing ---')

    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorld-CPP-v0",
        size=GRID_SIZE,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"  # No rendering for faster testing
    )
    env = FlattenObservation(env)

    num_episodes = 10
    coverage_results = []
    total_reward_results = []
    
    print(f"Running {num_episodes} test episodes...")
    
    for i in range(num_episodes):
        (obs, _) = env.reset()
        done = False
        truncated = False
        steps = 0
        total_reward = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1
        
        coverage = info['coverage']
        coverage_results.append(coverage)
        total_reward_results.append(total_reward)
        
        print(f"Episode {i+1:2d}: Coverage: {coverage:5.1f}% | Steps: {steps:3d} | Total Reward: {total_reward:7.2f}")

    avg_coverage = sum(coverage_results) / len(coverage_results)
    avg_reward = sum(total_reward_results) / len(total_reward_results)
    max_coverage = max(coverage_results)
    
    print(f"\n--- Test Finished ---")
    print(f"Average Coverage: {avg_coverage:.1f}%")
    print(f"Max Coverage: {max_coverage:.1f}%")
    print(f"Average Total Reward: {avg_reward:.2f}")
