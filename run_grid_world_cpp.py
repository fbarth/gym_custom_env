"""
Run a single episode of the CPP GridWorld environment with a random agent.

Usage:
    python run_grid_world_cpp.py

The environment uses a 5x5 grid with 3 obstacles.
Cells are coloured light-green once visited; the agent is the blue circle.
"""

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3.common.env_checker import check_env

def get_direction(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorldCPP-v0",
    entry_point=GridWorldCPPEnv,
)

GRID_SIZE = 5
OBSTACLES = 3
MAX_STEPS = 200

env = gym.make(
    "gymnasium_env/GridWorldCPP-v0",
    render_mode="human",
    size=GRID_SIZE,
    obs_quantity=OBSTACLES,
    max_steps=MAX_STEPS,
)

check_env(env)

state, info = env.reset()
print(f"Initial state: {state}")
print(f"Free cells to cover: {info['free_cells']}")
print("-" * 60)

done = False
truncated = False
total_reward = 0.0
step = 0

while not done and not truncated:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    print(
        f"Step {step:3d} | action={get_direction(action):5s} | "
        f"reward={reward:+.2f} | total={total_reward:+.2f} | "
        f"coverage={info['coverage_ratio']:.0%} "
        f"({info['visited_cells']}/{info['free_cells']})"
    )

print("-" * 60)
if done:
    print(f"Full coverage achieved in {step} steps! Total reward: {total_reward:.2f}")
else:
    print(f"Truncated after {step} steps. Coverage: {info['coverage_ratio']:.0%}. Total reward: {total_reward:.2f}")

env.close()
