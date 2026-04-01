#!/usr/bin/env python
"""
Script para testar o ambiente GridWorldCPP com um agente aleatório.
Este script executa em modo headless (sem renderização gráfica).
"""

import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from gymnasium.wrappers import FlattenObservation

gym.register(
    id="gymnasium_env/GridWorld-CPP-v0",
    entry_point=GridWorldCPPEnv,
)

print("=" * 80)
print("Testing GridWorldCPP Environment with Random Agent (Headless Mode)")
print("=" * 80)

# Test 1: Environment validation without rendering
print("\n[Test 1] Creating environment without rendering...")
env = gym.make("gymnasium_env/GridWorld-CPP-v0", render_mode=None, size=5, max_steps=50)
env = FlattenObservation(env)
print(f"✓ Environment created successfully")
print(f"  - Observation space: {env.observation_space}")
print(f"  - Action space: {env.action_space}")

# Test 2: Reset environment
print("\n[Test 2] Resetting environment...")
state, info = env.reset()
print(f"✓ Environment reset successfully")
print(f"  - State shape: {state.shape}")
print(f"  - Initial coverage: {info['coverage']:.1f}%")

# Test 3: Run one episode
print("\n[Test 3] Running one episode with random actions...")
done = False
truncated = False
steps = 0
total_reward = 0.0
episode_data = []

while not done and not truncated:
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    episode_data.append({
        'step': steps,
        'action': action,
        'reward': reward,
        'coverage': info['coverage'],
        'cells_visited': info['cells_visited']
    })
    
    if steps % 10 == 0:
        print(f"  Step {steps:3d}: Action={action}, Reward={reward:7.2f}, Coverage={info['coverage']:5.1f}%")

print(f"\n✓ Episode finished!")
print(f"  - Total steps: {steps}")
print(f"  - Final coverage: {info['coverage']:.1f}%")
print(f"  - Cells visited: {info['cells_visited']}/{info['total_cells']}")
print(f"  - Total reward: {total_reward:.2f}")

# Test 4: Multiple episodes
print("\n[Test 4] Running 5 episodes to check consistency...")
coverage_results = []

for episode in range(1, 6):
    state, _ = env.reset()
    done = False
    truncated = False
    
    while not done and not truncated:
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
    
    coverage_results.append(info['coverage'])
    print(f"  Episode {episode}: Final coverage = {info['coverage']:5.1f}%")

avg_coverage = sum(coverage_results) / len(coverage_results)
print(f"\n✓ Average coverage over 5 episodes: {avg_coverage:.1f}%")

env.close()

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
