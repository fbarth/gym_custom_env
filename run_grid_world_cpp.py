import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from gymnasium.wrappers import FlattenObservation

gym.register(
    id="gymnasium_env/GridWorld-CPP-v0",
    entry_point=GridWorldCPPEnv,
)

# Create environment with rendering
env = gym.make("gymnasium_env/GridWorld-CPP-v0", render_mode="human", size=5, max_steps=100)

# Flatten observation for random agent (matches typical RL setup)
env = FlattenObservation(env)

(state, _) = env.reset()
print(f"Initial State shape: {state.shape}")

done = False
truncated = False
steps = 0
total_reward = 0

while not done and not truncated:
    action = env.action_space.sample()  # Random action
    (next_state, reward, terminated, truncated, info) = env.step(action)
    
    print(f"Step: {steps:3d} | Action: {action} | Reward: {reward:7.2f} | Coverage: {info['coverage']:5.1f}% | Total Reward: {total_reward + reward:7.2f}")
    
    total_reward += reward
    done = terminated
    steps += 1

print(f"\nEpisode finished!")
print(f"Total Steps: {steps}")
print(f"Final Coverage: {info['coverage']:.1f}%")
print(f"Total Reward: {total_reward:.2f}")
print(f"Cells Visited: {info['cells_visited']}/{info['total_cells']}")

env.close()
