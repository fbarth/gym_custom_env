import gymnasium as gym
from gymnasium_env.grid_world_cpp import GridWorldCPPEnv
from stable_baselines3.common.env_checker import check_env

def get_direction(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")

gym.register(
    id="gymnasium_env/GridWorldCPP-v0",
    entry_point=GridWorldCPPEnv,
)

env = gym.make(
    "gymnasium_env/GridWorldCPP-v0",
    render_mode="human",
    size=5,
    obs_quantity=3,
    max_steps=200,
)

check_env(env.unwrapped)

obs, info = env.reset()
print("=" * 50)
print("Coverage Path Planning – Random Agent (5x5)")
print("=" * 50)
print(f"Free cells to cover: {info['free_cells']}")
print(f"Initial obs: {obs}\n")

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

print("\n" + "=" * 50)
if done:
    print(f"SUCCESS – full coverage in {step} steps!")
else:
    print(f"TIMEOUT – reached {step} steps without full coverage.")
print(f"Total reward: {total_reward:.2f}")
print(f"Final coverage: {info['covered_cells']}/{info['free_cells']} ({info['coverage_ratio']*100:.1f}%)")
