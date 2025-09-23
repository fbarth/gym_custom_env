
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

DIM=20
OBSTACLES=40
MAX_STEPS=500
TOTAL_TIMESTEPS=500_000

if train:
    print("Starting training...") 
    env = gym.make(
        "gymnasium_env/GridWorld-v1", 
        size=DIM, 
        obs_quantity=OBSTACLES, 
        max_steps=MAX_STEPS, 
        render_mode="rgb_array"
    )
    print(f"Environment created. Observation space: {env.observation_space}")
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", ent_coef=0.02)
    print("Model created, starting training...")
    new_logger = configure(f'log/ppo_obstacles_{DIM}_{MAX_STEPS}', ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    print("Logger configured, learning...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(f"data/ppo_obstacles_{DIM}")
    print('model trained')

print('loading model')
model = PPO.load(f"data/ppo_obstacles_{DIM}")
env = gym.make(
    "gymnasium_env/GridWorld-v1", 
    size=DIM, 
    obs_quantity=OBSTACLES, 
    max_steps=MAX_STEPS, 
    render_mode="human"
)
(obs, _) = env.reset()
done = False
truncated = False

while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action.item())
    print(f"Action: {action}, Reward: {reward}, Next State: {obs}, Done: {done}, Truncated: {truncated}")