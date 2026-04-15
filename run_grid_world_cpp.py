import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv

def get_direction(action):
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down"
    }.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorldCPP-v0",
    entry_point=GridWorldCPPEnv,
)

# Small 5x5 grid with 3 obstacles for easy visualization
env = gym.make(
    "gymnasium_env/GridWorldCPP-v0",
    render_mode="human",
    size=5,
    obs_quantity=3,
    max_steps=100,
)

NUM_EPISODES = 3

for ep in range(NUM_EPISODES):
    print(f"\n{'='*50}")
    print(f"Episode {ep + 1}")
    print(f"{'='*50}")

    (state, info) = env.reset()
    print(f"Initial State: {state}")
    print(f"Free cells to cover: {info['total_free_cells']}")

    done = False
    total_reward = 0
    step = 0

    while not done:
        action = env.action_space.sample()
        (next_state, reward, terminated, truncated, info) = env.step(action)
        total_reward += reward
        step += 1

        print(f"Step {step:3d} | Action: {get_direction(action):5s} | "
              f"Reward: {reward:+6.2f} | Coverage: {info['coverage']:.1%} | "
              f"Visited: {info['visited_cells']}/{info['total_free_cells']}")

        done = terminated or truncated

    print(f"\nEpisode {ep + 1} finished:")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final coverage: {info['coverage']:.1%}")
    print(f"  Full coverage: {'Yes' if terminated else 'No'}")

env.close()
