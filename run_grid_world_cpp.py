"""
run_grid_world_cpp.py
---------------------
Executes ONE episode of the CPP environment using a RANDOM agent.
Use this script to visually validate the environment before training.

Usage:
    python run_grid_world_cpp.py            # with pygame rendering
    python run_grid_world_cpp.py headless   # no rendering (faster)
"""

import sys
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="GridWorldCPP-v0",
    entry_point="gymnasium_env.grid_world_cpp:GridWorldCPPEnv",
    max_episode_steps=200,
)

# Choose render mode
render_mode = "human" if (len(sys.argv) < 2 or sys.argv[1] != "headless") else None

env = gym.make(
    "GridWorldCPP-v0",
    size=5,
    obs_quantity=3,
    max_steps=200,
    render_mode=render_mode,
)

obs, info = env.reset(seed=42)
print("=" * 50)
print("GridWorld CPP — Random Agent")
print(f"Grid size     : {5}x{5}")
print(f"Obstacles     : {3}")
print(f"Accessible    : {info['accessible_cells']} cells")
print("=" * 50)

total_reward = 0.0
done = False
step = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    done = terminated or truncated

    status = "NEW" if reward > 0.4 else ("DONE!" if terminated else "revisit")
    print(
        f"Step {step:3d} | action={action} | reward={reward:+.2f} | "
        f"coverage={info['coverage_ratio']:6.1%} "
        f"({info['covered_cells']}/{info['accessible_cells']}) | {status}"
    )

print("=" * 50)
print(f"Episode finished after {step} steps")
print(f"Total reward    : {total_reward:.2f}")
print(f"Final coverage  : {info['coverage_ratio']:.1%} "
      f"({info['covered_cells']}/{info['accessible_cells']} cells)")
if info['coverage_ratio'] == 1.0:
    print("✓ Full coverage achieved!")
else:
    print("✗ Full coverage NOT achieved (random agent).")
print("=" * 50)

env.close()
