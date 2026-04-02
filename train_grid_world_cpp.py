# python train_grid_world_cpp.py <train|test|run|random>
#
# Coverage Path Planning (CPP) — Training Script  [REVISED]
#
# === KEY HYPERPARAMETER CHANGES FROM ORIGINAL ===
#
# | Parameter       | Before      | After       | Why                                    |
# |-----------------|-------------|-------------|----------------------------------------|
# | TOTAL_TIMESTEPS | 1_000_000   | 3_000_000   | More samples for a richer obs space    |
# | gamma           | 0.99        | 0.995       | Completion bonus at step ~100 was      |
# |                 |             |             | discounted to 0.99^100 ≈ 37% of its   |
# |                 |             |             | face value; now 0.995^100 ≈ 61%       |
# | net_arch        | default     | [256,256,   | Larger network to process the full     |
# |                 | (64, 64)    |  128]       | 28-dim global-map observation          |
# | ent_coef        | 0.03        | 0.05        | More entropy → more exploration early  |
# | learning_rate   | 1e-4        | 3e-4        | Faster convergence with larger batch   |
# | n_steps         | 2048        | 4096        | Smoother gradient estimates            |
# | batch_size      | 128         | 256         | Matches the larger n_steps             |
# | n_epochs        | default (10)| 10          | Kept                                   |
# | gae_lambda      | default     | 0.95        | Explicit; good for CPP horizon         |
#

import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

from gymnasium_env.grid_world_obstacles_cpp import GridWorldRenderEnv


def print_action(action: int) -> str:
    return {0: "right", 1: "up", 2: "left", 3: "down"}.get(action, "unknown")


def resolve_model_paths(model_name: str):
    primary_model_path = f"data/{model_name}"
    best_model_path = Path("log") / model_name / "best_model" / "best_model"
    if best_model_path.with_suffix(".zip").exists():
        return primary_model_path, str(best_model_path)
    return primary_model_path, None


if len(sys.argv) < 2 or sys.argv[1] not in ["train", "test", "run", "random"]:
    print("Usage: python train_grid_world_cpp.py <train|test|run|random>")
    sys.exit(1)

mode = sys.argv[1]

try:
    gym.register(
        id="gymnasium_env/GridWorldCPP-v0",
        entry_point=GridWorldRenderEnv,
    )
except Exception:
    pass

# ── Hyperparameters ────────────────────────────────────────────────────────────
DIM         = 5
OBSTACLES   = 3
MAX_STEPS   = 200
TOTAL_TIMESTEPS = 3_000_000   # was 1_000_000

ENTROPY_COEF  = 0.05          # was 0.03 — more entropy = more exploration
LEARNING_RATE = 3e-4          # was 1e-4
GAMMA         = 0.995         # was 0.99 — better long-horizon credit assignment
N_STEPS       = 4096          # was 2048
BATCH_SIZE    = 256           # was 128
N_EPOCHS      = 10
GAE_LAMBDA    = 0.95

# Larger MLP to process the richer 28-dim observation (was default 64×64)
POLICY_KWARGS = dict(net_arch=[256, 256, 128])
# ──────────────────────────────────────────────────────────────────────────────


if mode == "train":
    print("─" * 60)
    print("  CPP Training — REVISED CONFIG")
    print("─" * 60)
    print(f"  Grid           : {DIM}×{DIM} | Obstacles: {OBSTACLES} | Max steps: {MAX_STEPS}")
    print(f"  Timesteps      : {TOTAL_TIMESTEPS:,}")
    print(f"  gamma          : {GAMMA}  |  ent_coef: {ENTROPY_COEF}  |  lr: {LEARNING_RATE}")
    print(f"  n_steps/batch  : {N_STEPS}/{BATCH_SIZE}  |  net_arch: {POLICY_KWARGS['net_arch']}")
    print("─" * 60)

    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )
    check_env(env)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=ENTROPY_COEF,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gae_lambda=GAE_LAMBDA,
        policy_kwargs=POLICY_KWARGS,
        device="cpu",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"ppo_cpp_{DIM}_{OBSTACLES}_{MAX_STEPS}"
        f"_ent{ENTROPY_COEF}_g{GAMMA}_{timestamp}"
    )
    log_dir    = f"log/{run_name}"
    model_path = f"data/{run_name}.zip"

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    eval_env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(Path(log_dir) / "best_model"),
        log_path=str(Path(log_dir) / "eval"),
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        use_masking=True,
    )

    print(f"\n  Logging to    : {log_dir}")
    print(f"  Best model    : {Path(log_dir) / 'best_model'}")
    print(f"  Final model   : {model_path}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(model_path)
    print("Model trained and saved.")


elif mode == "run":
    model_name = input("Enter model filename (without .zip): ")
    model_path, best_model_path = resolve_model_paths(model_name)
    load_path = best_model_path or model_path
    print(f"--- Loading model from {load_path}.zip for a visual run ---")

    model = MaskablePPO.load(load_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human",
    )

    obs, _ = env.reset()
    done = False
    truncated = False
    steps = 0
    while not done and not truncated:
        action, _ = model.predict(
            obs,
            deterministic=False,
            action_masks=env.unwrapped.action_masks(),
        )
        obs, reward, done, truncated, info = env.step(action.item())
        print(
            f"Step {steps+1:3d} | Action: {print_action(action.item()):5s} | "
            f"Reward: {reward:+.3f} | Coverage: {info['coverage_ratio']*100:.1f}% "
            f"({info['visited_cells']}/{info['total_free_cells']} cells)"
        )
        steps += 1

    final_coverage = info["coverage_ratio"] * 100
    print("\n--- Run Finished ---")
    print(f"Final coverage: {final_coverage:.1f}% in {steps} steps.")
    if done:
        print("Result: FULL COVERAGE ACHIEVED")
    else:
        print("Result: Truncated — coverage incomplete.")


elif mode == "test":
    model_name = input("Enter model filename (without .zip): ")
    model_path, best_model_path = resolve_model_paths(model_name)
    load_path = best_model_path or model_path
    print(f"--- Loading model from {load_path}.zip for batch testing ---")

    model = MaskablePPO.load(load_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    num_episodes = 1000
    full_coverage_count = 0
    coverage_rates = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        last_info = {}
        while not done and not truncated:
            action, _ = model.predict(
                obs,
                deterministic=False,
                action_masks=env.unwrapped.action_masks(),
            )
            obs, reward, done, truncated, last_info = env.step(action.item())

        coverage = last_info.get("coverage_ratio", 0.0) * 100
        coverage_rates.append(coverage)
        if done:
            full_coverage_count += 1

    print("\n─── Test Finished ───────────────────────────────────")
    print(
        f"  Full coverage rate : {full_coverage_count}/{num_episodes} "
        f"({full_coverage_count / num_episodes * 100:.1f}%)"
    )
    print(f"  Mean coverage      : {sum(coverage_rates)/len(coverage_rates):.1f}%")
    print(f"  Min / Max coverage : {min(coverage_rates):.1f}% / {max(coverage_rates):.1f}%")
    print("────────────────────────────────────────────────────")


elif mode == "random":
    print("─" * 60)
    print("  Random Agent Baseline")
    print("─" * 60)
    print(f"  Grid: {DIM}×{DIM} | Obstacles: {OBSTACLES} | Max steps: {MAX_STEPS}")
    print("─" * 60)

    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    num_episodes = 1000
    full_coverage_count = 0
    coverage_rates = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        last_info = {}
        while not done and not truncated:
            # Uniformly random — no masking, truly naive baseline
            action = env.action_space.sample()
            obs, reward, done, truncated, last_info = env.step(action)

        coverage = last_info.get("coverage_ratio", 0.0) * 100
        coverage_rates.append(coverage)
        if done:
            full_coverage_count += 1

    print("\n─── Random Agent Baseline ───────────────────────────")
    print(
        f"  Full coverage rate : {full_coverage_count}/{num_episodes} "
        f"({full_coverage_count / num_episodes * 100:.1f}%)"
    )
    print(f"  Mean coverage      : {sum(coverage_rates)/len(coverage_rates):.1f}%")
    print(f"  Min / Max coverage : {min(coverage_rates):.1f}% / {max(coverage_rates):.1f}%")
    print("─────────────────────────────────────────────────────")