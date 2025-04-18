import gymnasium as gym
from gymnasium_env.grid_world import GridWorldEnv
from gymnasium.wrappers import FlattenObservation

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)

env = gym.make("gymnasium_env/GridWorld-v0", size=5)

# usando o wrapper
env = FlattenObservation(env)

(state, _) = env.reset()
done = False
steps = 0
while not done and steps < 100:
    action = env.action_space.sample()
    (next_state, reward, terminated, truncated, info) = env.step(action)
    print(f"Step: {steps}, Action: {action}, Reward: {reward}, Next State: {next_state}")
    done = terminated or truncated
    steps += 1

env.close()
