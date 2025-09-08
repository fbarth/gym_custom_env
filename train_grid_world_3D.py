
#
# python train_grid_world_render_v0.py <train|test>
#

import gymnasium as gym
from gymnasium_env.grid_world_3D import GridWorldEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

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

if train:
    env = gym.make("gymnasium_env/GridWorld-v0", size=5)
    env = FlattenObservation(env)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    new_logger = configure('log/ppo_custom_env', ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=100_000)
    model.save("data/ppo_custom_env")
    print('model trained')

print('loading model')
model = PPO.load("data/ppo_custom_env")
env = gym.make("gymnasium_env/GridWorld-v0", size=5)
env = FlattenObservation(env)
(obs, _) = env.reset()
done = False

steps = 0
while not done and steps < 500:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action.item())
    print(f"Action: {print_action(action.item())}, Reward: {reward}, Next State: {obs}")
    steps += 1