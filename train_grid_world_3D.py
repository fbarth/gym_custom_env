
#
# python train_grid_world_render_v0.py <train|test>
#

import gymnasium as gym
from gymnasium_env.grid_world_3D import GridWorldEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from datetime import datetime

def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
        4: "forward",
        5: "backward"
    }.get(action, "unknown")

import sys
train = True if sys.argv[1] == 'train' else False

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)

DIM=10
MAX_STEPS=500
TOTAL_TIMESTEPS=500_000
ENTROPY_COEF=0.02

if train:
    env = gym.make("gymnasium_env/GridWorld-v0", size=DIM, max_steps=MAX_STEPS)
    env = FlattenObservation(env)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=ENTROPY_COEF)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_logger = configure(
        f'log/ppo_grid_3d_{DIM}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}',
        ["stdout", "csv", "tensorboard"]
    )
    model.set_logger(new_logger)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(f'data/ppo_grid_3d_{DIM}')
    print('model trained')

print('loading model')
model = PPO.load(f'data/ppo_grid_3d_{DIM}')
env = gym.make("gymnasium_env/GridWorld-v0", size=DIM, max_steps=MAX_STEPS)
env = FlattenObservation(env)
(obs, _) = env.reset()
done = False

steps = 0
while not done and steps < MAX_STEPS:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action.item())
    print(f"Action: {print_action(action.item())}, Reward: {reward}, Next State: {obs}")
    steps += 1