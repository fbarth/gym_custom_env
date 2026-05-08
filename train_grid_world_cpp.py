#
# python train_grid_world_cpp.py train   -- curriculum: 5x5 → 7x7 → mix → 10x10 exclusive
# python train_grid_world_cpp.py test    -- evaluate a saved model on 5x5, 10x10, 20x20
# python train_grid_world_cpp.py run     -- visual run of a saved model
#
# Strategy overview
# -----------------
# GridWorldCPPEnvV2 uses a Dict observation space (MultiInputPolicy):
#   - 5×5 local window (partial obs, compliant with assignment rules)
#   - visited_pooled: 2×8×8 max-pool of visited map — fixed size for all grids,
#     enabling cross-grid transfer without LSTM
#   - frontier: BFS direction+distance to nearest unvisited cell
#   - progress: steps/max_steps
#
# MaskablePPO (sb3-contrib): action masks prevent sampling moves that hit walls
# or obstacles, eliminating wasted steps and noisy gradient updates.
#
# Four-phase curriculum:
#   Phase 1 — 5×5  (1 M steps): agent learns core CPP strategy
#   Phase 2 — 5×5 + 7×7 mix (2 M steps): smooth upscaling
#   Phase 3 — 7×7 + 10×10 mix (4 M steps): generalises to target size
#   Phase 4 — 10×10 exclusive (5 M steps): pushes success rate toward ~90%
#

import sys
import os
import numpy as np
from datetime import datetime

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium_env.grid_world_cpp_v2 import GridWorldCPPEnvV2

os.makedirs("data", exist_ok=True)
os.makedirs("log", exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────

PHASE1_TIMESTEPS = 1_000_000
PHASE2_TIMESTEPS = 2_000_000
PHASE3_TIMESTEPS = 4_000_000
PHASE4_TIMESTEPS = 5_000_000

N_ENVS = 4

ENV_CONFIGS = {
    5:  {"obs_quantity": 3,  "max_steps": 200},
    7:  {"obs_quantity": 5,  "max_steps": 300},
    10: {"obs_quantity": 10, "max_steps": 700},
    20: {"obs_quantity": 40, "max_steps": 1000},
}

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.05,
    device="cpu",
    verbose=1,
)

# ──────────────────────────────────────────────────────────────────────────────


def make_env_fn(size: int):
    cfg = ENV_CONFIGS[size]
    def _init():
        env = GridWorldCPPEnvV2(size=size, **cfg)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test", "run"):
    print(__doc__)
    sys.exit(1)

mode = sys.argv[1]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
if mode == "train":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    check_env(GridWorldCPPEnvV2(size=5, **ENV_CONFIGS[5]))

    # Phase 1 — 5×5 ──────────────────────────────────────────────────────────
    print("\n=== Phase 1 — 5×5 warm-up ===")
    log_p1  = f"log/cpp_v2_5x5_{timestamp}"
    path_p1 = f"data/cpp_v2_5x5_{timestamp}"

    env_p1 = DummyVecEnv([make_env_fn(5) for _ in range(N_ENVS)])
    model  = MaskablePPO("MultiInputPolicy", env_p1, **PPO_KWARGS)
    model.set_logger(configure(log_p1, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=PHASE1_TIMESTEPS)
    model.save(path_p1)
    print(f"Phase 1 saved → {path_p1}.zip")

    # Phase 2 — 5×5 + 7×7 ────────────────────────────────────────────────────
    print("\n=== Phase 2 — mixed 5×5 + 7×7 ===")
    log_p2  = f"log/cpp_v2_5x5_7x7_{timestamp}"
    path_p2 = f"data/cpp_v2_5x5_7x7_{timestamp}"

    half = N_ENVS // 2
    env_p2 = DummyVecEnv(
        [make_env_fn(5) for _ in range(half)] +
        [make_env_fn(7) for _ in range(N_ENVS - half)]
    )
    model.set_env(env_p2)
    model.set_logger(configure(log_p2, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=PHASE2_TIMESTEPS, reset_num_timesteps=False)
    model.save(path_p2)
    print(f"Phase 2 saved → {path_p2}.zip")

    # Phase 3 — 7×7 + 10×10 ──────────────────────────────────────────────────
    print("\n=== Phase 3 — mixed 7×7 + 10×10 ===")
    log_p3  = f"log/cpp_v2_7x7_10x10_{timestamp}"
    path_p3 = f"data/cpp_v2_7x7_10x10_{timestamp}"

    env_p3 = DummyVecEnv(
        [make_env_fn(7)  for _ in range(half)] +
        [make_env_fn(10) for _ in range(N_ENVS - half)]
    )
    model.set_env(env_p3)
    model.set_logger(configure(log_p3, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=PHASE3_TIMESTEPS, reset_num_timesteps=False)
    model.save(path_p3)
    print(f"Phase 3 saved → {path_p3}.zip")

    # Phase 4 — 10×10 exclusive ───────────────────────────────────────────────
    print("\n=== Phase 4 — 10×10 exclusive fine-tune ===")
    log_p4  = f"log/cpp_v2_10x10_{timestamp}"
    path_p4 = f"data/cpp_v2_10x10_{timestamp}"

    env_p4 = DummyVecEnv([make_env_fn(10) for _ in range(N_ENVS)])
    model.set_env(env_p4)
    model.set_logger(configure(log_p4, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=PHASE4_TIMESTEPS, reset_num_timesteps=False)
    model.save(path_p4)
    print(f"Phase 4 saved → {path_p4}.zip")

    print("\n=== Training complete ===")
    print(f"Evaluate with:  python train_grid_world_cpp.py test")

# ── TEST ──────────────────────────────────────────────────────────────────────
elif mode == "test":
    model_name = input("Model filename (without .zip): ").strip()
    model = MaskablePPO.load(f"data/{model_name}.zip")

    for size in (5, 10, 20):
        cfg = ENV_CONFIGS[size]
        env = GridWorldCPPEnvV2(size=size, **cfg)
        coverages = []
        full_count = 0

        for ep in range(100):
            obs, info = env.reset(seed=ep)
            done = False
            while not done:
                masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            cov = info["reachable_ratio"] * 100
            coverages.append(cov)
            if cov >= 99.0:
                full_count += 1

        env.close()
        print(f"\n{size}×{size}  (obstacles={cfg['obs_quantity']}, "
              f"max_steps={cfg['max_steps']}, n=100 episodes)")
        print(f"  mean coverage  : {np.mean(coverages):.1f}%  (of reachable cells)")
        print(f"  std            : {np.std(coverages):.1f}%")
        print(f"  min / max      : {np.min(coverages):.1f}% / {np.max(coverages):.1f}%")
        print(f"  full (>=99%)   : {full_count}/100")

# ── RUN ───────────────────────────────────────────────────────────────────────
elif mode == "run":
    size_str = input("Grid size [5 / 10 / 20] (default 10): ").strip() or "10"
    size = int(size_str)
    if size not in ENV_CONFIGS:
        print(f"Unknown size {size}. Choose 5, 10, or 20.")
        sys.exit(1)

    model_name = input("Model filename (without .zip): ").strip()
    model = MaskablePPO.load(f"data/{model_name}.zip")

    cfg = ENV_CONFIGS[size]
    env = GridWorldCPPEnvV2(size=size, **cfg, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    print(f"\nFinal coverage: {info['reachable_ratio'] * 100:.1f}%  "
          f"in {info['steps']} steps")
    env.close()
