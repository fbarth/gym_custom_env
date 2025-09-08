import gymnasium as gym
from gymnasium_env.grid_world_render_state_matrix import GridWorldRenderEnv

def get_direction(action):
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down"
    }.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldRenderEnv,
)

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human", size=10)

(state, _) = env.reset()
done = False
steps = 0
while not done and steps < 100:
    action = env.action_space.sample()
    (next_state, reward, terminated, truncated, info) = env.step(action)
    print(f"Step: {steps}, Action: {get_direction(action)}, Reward: {reward}, Next State: {next_state}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    done = terminated or truncated
    steps += 1

env.close()
