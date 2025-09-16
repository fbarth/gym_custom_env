
#
# python train_grid_world_render_v0.py <train|test>
#

import gymnasium as gym
from gymnasium_env.grid_world_obstacles import GridWorldRenderEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import sys
train = True if sys.argv[1] == 'train' else False

print(f"Train mode: {train}")

try:
    gym.register(
        id="gymnasium_env/GridWorld-v1",
        entry_point=GridWorldRenderEnv,
    )
except Exception as e:
    print(f"Environment registration failed: {e}")

if train:
    print("Starting training...") 
    env = gym.make("gymnasium_env/GridWorld-v1", size=10, obs_quantity=20, render_mode="rgb_array")
    print(f"Environment created. Observation space: {env.observation_space}")
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    print("Model created, starting training...")
    new_logger = configure('log/ppo_obstacles', ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    print("Logger configured, learning...")
    model.learn(total_timesteps=500_000)
    model.save("data/ppo_obstacles")
    print('model trained')

print('loading model')
model = PPO.load("data/ppo_obstacles")
env = gym.make("gymnasium_env/GridWorld-v1", size=10, obs_quantity=20, render_mode="human")
(obs, _) = env.reset()
done = False
truncated = False

while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action.item())
    print(f"Action: {action}, Reward: {reward}, Next State: {obs}, Done: {done}, Truncated: {truncated}")