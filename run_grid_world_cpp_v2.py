"""
Run one episode of the CPP v2 environment with a random agent.

Difference from v1: the observation includes the nearest unvisited cell,
shown as a yellow square in the render.

Usage:
    python run_grid_world_cpp_v2.py
"""

import gymnasium as gym
from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2
from stable_baselines3.common.env_checker import check_env

def get_direction(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorldCPPV2-v0",
    entry_point=GridWorldCPPEnvV2,
)

env = gym.make(
    "gymnasium_env/GridWorldCPPV2-v0",
    render_mode="human",
    size=5,
    obs_quantity=3,
    max_steps=400,
)

check_env(env.unwrapped)

obs, info = env.reset()
print("=" * 55)
print("Coverage Path Planning v2 – Random Agent (5x5)")
print("Yellow cell = nearest unvisited target")
print("=" * 55)
print(f"Free cells to cover: {info['free_cells']}")

done = False
truncated = False
step = 0
total_reward = 0.0

while not done and not truncated:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    print(
        f"Step {step:>3} | Action: {get_direction(action):<5} | "
        f"Reward: {reward:+.2f} | Coverage: {info['covered_cells']}/{info['free_cells']} "
        f"({info['coverage_ratio']*100:.1f}%)"
    )

env.close()

print("\n" + "=" * 55)
if done:
    print(f"SUCCESS – full coverage in {step} steps!")
else:
    print(f"TIMEOUT – reached {step} steps without full coverage.")
print(f"Total reward: {total_reward:.2f}")
print(f"Final coverage: {info['covered_cells']}/{info['free_cells']} ({info['coverage_ratio']*100:.1f}%)")
