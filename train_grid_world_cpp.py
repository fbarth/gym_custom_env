"""
train_grid_world_cpp.py
-----------------------
Trains, tests, or runs a PPO agent on the Coverage Path Planning (CPP) environment.

Usage:
    python train_grid_world_cpp.py train   # train and save the model
    python train_grid_world_cpp.py test    # evaluate over 100 episodes
    python train_grid_world_cpp.py run     # run 1 episode with rendering
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

register(
    id="GridWorldCPP-v0",
    entry_point="gymnasium_env.grid_world_cpp:GridWorldCPPEnv",
    max_episode_steps=200,
)

MODEL_PATH = "data/ppo_grid_world_cpp"
LOG_PATH   = "log/grid_world_cpp"

ENV_KWARGS = dict(size=5, obs_quantity=3, max_steps=200)


def train():
    os.makedirs("data", exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    print("Training PPO agent on GridWorldCPP-v0 ...")
    vec_env = make_vec_env("GridWorldCPP-v0", n_envs=4, env_kwargs=ENV_KWARGS)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_PATH,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(total_timesteps=300_000)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}.zip")
    print(f"TensorBoard logs at: {LOG_PATH}")
    print("Run: tensorboard --logdir log/grid_world_cpp")


def test():
    env = gym.make("GridWorldCPP-v0", **ENV_KWARGS)
    model = PPO.load(MODEL_PATH, env=env)

    n_episodes = 100
    coverages = []
    rewards_list = []
    steps_list = []
    full_coverage = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        coverages.append(info["coverage_ratio"])
        rewards_list.append(total_reward)
        steps_list.append(info["steps"])
        if info["coverage_ratio"] == 1.0:
            full_coverage += 1

    env.close()
    print("=" * 50)
    print(f"Evaluation over {n_episodes} episodes:")
    print(f"  Full coverage achieved : {full_coverage}/{n_episodes} ({full_coverage}%)")
    print(f"  Mean coverage ratio    : {np.mean(coverages):.2%} ± {np.std(coverages):.2%}")
    print(f"  Mean total reward      : {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"  Mean steps per episode : {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print("=" * 50)


def run():
    env = gym.make("GridWorldCPP-v0", **ENV_KWARGS, render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    print("Running trained PPO agent (press Ctrl+C to stop) ...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

    print(f"Done in {step} steps | Coverage: {info['coverage_ratio']:.1%} | Reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test", "run"):
        print("Usage: python train_grid_world_cpp.py [train|test|run]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "test":
        test()
    elif cmd == "run":
        run()
