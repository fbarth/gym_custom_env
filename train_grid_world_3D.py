
#
# python train_grid_world_render_v0.py <train|test|run>
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
if sys.argv[1] not in ['train', 'test', 'run']:
    print("Usage: python train_grid_world_3D.py <train|test|run>")
    sys.exit(1)

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)

DIM=10
MAX_STEPS=500
TOTAL_TIMESTEPS=500_000
ENTROPY_COEF=0.02

if sys.argv[1] == 'train':
    env = gym.make(
        "gymnasium_env/GridWorld-v0", 
        size=DIM, 
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
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
    model.save(f'data/ppo_grid_3d_{DIM}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip')
    print('model trained')

elif sys.argv[1] == 'run':
    model_name = input("Enter model filename (without path and extension): ")
    print('loading model')
    model = PPO.load(f'data/{model_name}.zip')
    env = gym.make(
        "gymnasium_env/GridWorld-v0", 
        size=DIM, 
        max_steps=MAX_STEPS, 
        render_mode="human"
    )
    env = FlattenObservation(env)
    (obs, _) = env.reset()
    done = False

    steps = 0
    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action.item())
        print(f"Action: {print_action(action.item())}, Reward: {reward}, Next State: {obs}")
        steps += 1

else:
    model_name = input("Enter model filename (without path and extension): ")
    print('loading model')
    success = 0
    model = PPO.load(f'data/{model_name}.zip')
    for i in range(100):    
        env = gym.make(
            "gymnasium_env/GridWorld-v0", 
            size=DIM, 
            max_steps=MAX_STEPS,
            render_mode="rgb_array"
        )
        env = FlattenObservation(env)
        (obs, _) = env.reset()
        done = False

        steps = 0
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action.item())
            print(f"Action: {print_action(action.item())}, Reward: {reward}, Next State: {obs}")
            steps += 1    
        
        if done:
            success += 1
            print(f"Episode {i+1}: Success in {steps} steps")
        else:
            print(f"Episode {i+1}: Failed to reach goal in {MAX_STEPS} steps")
    
    print(f"Success rate: {success / 100 * 100}%")