from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# GridWorldRenderEnv — Coverage Path Planning (CPP)  [REVISED]
#
# === KEY CHANGES FROM ORIGINAL ===
#
# 1. OBSERVATION — full grid map instead of 3×3 local patch
#    Old: agent_x, agent_y, coverage_scaled(0-4), 4 neighbors, 3×3 patch = 16 values
#    New: agent_x, agent_y, coverage_pct(0-100), size×size grid map       = 28 values
#         Grid encoding: 0=unvisited free, 1=visited, 2=obstacle/wall
#    WHY: A 3×3 patch is functionally blind — the agent cannot see unvisited
#         cells more than one step away, making systematic planning impossible.
#
# 2. REWARD — revisit penalty reduced from -1.0 → -0.2
#    WHY: With revisit=-1.0, traversing 3 visited cells to reach an unvisited
#         corner yields net reward (3×-1) + 1 = -2. The agent correctly learned
#         NOT to explore far corners. Reducing the revisit penalty lets the agent
#         treat visited cells as corridors without being punished for it.
#
# 3. REWARD — completion bonus +20 → +30, stagnation threshold 10 → 7
#    WHY: Stronger terminal signal; tighter anti-loop pressure.
#
# 4. REWARD — truncation partial reward scaled up: 5×ratio → 10×ratio
#    WHY: Better gradient signal when the agent finishes at, say, 90% coverage.
#
# Observation:
#   [agent_x, agent_y, coverage_pct, grid[0,0], grid[1,0], ..., grid[size-1,size-1]]
#   coverage_pct : 0–100 integer (100 = full coverage)
#   grid cell    : 0 = free/unvisited, 1 = visited, 2 = obstacle
#
# Actions:  0=right  1=up  2=left  3=down
#
# Reward summary:
#   +1.0   first visit to a free cell
#   -0.2   revisiting an already-visited cell
#   -1.0   illegal move (wall or obstacle) — agent stays in place
#   -0.02  per-step time penalty
#   +30.0  completion bonus (all free cells visited)
#   -0.5   per-step stagnation penalty once steps_since_new >= 7
#   -5.0 + 10×coverage_ratio  on truncation
#

class GridWorldRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size: int = 5,
        obs_quantity: int = 5,
        max_steps: int = 200,
    ):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity

        self.count_steps = 0
        self.max_steps = max_steps
        self.steps_since_new = 0

        # Obstacle storage — list for rendering, set for O(1) lookup
        self.obstacles_locations: list = []
        self._obstacle_set: set = set()

        # Coverage state
        self.visited: Optional[np.ndarray] = None
        self.total_free_cells: int = 0

        self._agent_location = np.array([-1, -1], dtype=int)

        # ------------------------------------------------------------------
        # Observation space: [agent_x, agent_y, coverage_pct] + flat grid
        # ------------------------------------------------------------------
        obs_len = 2 + 1 + size * size
        low = np.zeros(obs_len, dtype=int)
        high = np.array(
            [size - 1, size - 1, 100] + [2] * (size * size),
            dtype=int,
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=int)

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1,  0]),   # right
            1: np.array([0, -1]),   # up
            2: np.array([-1, 0]),   # left
            3: np.array([0,  1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coverage_ratio(self) -> float:
        if self.total_free_cells == 0:
            return 0.0
        return float(self.visited.sum()) / self.total_free_cells

    def _build_grid_obs(self) -> np.ndarray:
        """
        Encode the full size×size grid as a flat integer array.
        0 = free/unvisited, 1 = visited, 2 = obstacle.
        Obstacle cells are never marked as visited, so there is no conflict.
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        grid[self.visited] = 1
        for ox, oy in self._obstacle_set:
            grid[ox, oy] = 2
        return grid.flatten()

    def _get_obs(self) -> np.ndarray:
        coverage_pct = int(self._coverage_ratio() * 100)
        core = np.array([*self._agent_location, coverage_pct], dtype=int)
        return np.concatenate((core, self._build_grid_obs()))

    def _get_info(self) -> dict:
        return {
            "coverage_ratio": self._coverage_ratio(),
            "visited_cells": int(self.visited.sum()),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
        }

    def action_masks(self) -> np.ndarray:
        """Return a binary mask of valid actions (used by MaskablePPO)."""
        mask = np.ones(self.action_space.n, dtype=np.int8)
        for action, direction in self._action_to_direction.items():
            nxt = self._agent_location + direction
            in_bounds = (0 <= nxt[0] < self.size) and (0 <= nxt[1] < self.size)
            hits_obstacle = tuple(nxt) in self._obstacle_set
            if (not in_bounds) or hits_obstacle:
                mask[action] = 0
        return mask

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.count_steps = 0
        self.steps_since_new = 0
        self.obstacles_locations = []
        self._obstacle_set = set()

        # Place agent at a random free cell
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles (not on the agent's start cell, not on each other)
        for _ in range(self.obs_quantity):
            loc = self._agent_location.copy()
            while (
                tuple(loc) == tuple(self._agent_location)
                or tuple(loc) in self._obstacle_set
            ):
                loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(loc)
            self._obstacle_set.add(tuple(loc))

        # Initialise visited grid; start cell is already visited
        self.visited = np.zeros((self.size, self.size), dtype=bool)
        self.visited[self._agent_location[0], self._agent_location[1]] = True

        # Total free (non-obstacle) cells the agent must cover
        self.total_free_cells = (
            self.size * self.size - len(self._obstacle_set)
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        proposed = self._agent_location + direction

        # Check bounds and obstacles
        in_bounds = (0 <= proposed[0] < self.size) and (0 <= proposed[1] < self.size)
        hits_obstacle = tuple(proposed) in self._obstacle_set

        if (not in_bounds) or hits_obstacle:
            # Illegal move — agent stays in place
            reward = -1.0
            self.steps_since_new += 1
        else:
            self._agent_location = proposed.copy()
            cell = (int(proposed[0]), int(proposed[1]))

            if not self.visited[cell]:
                # ✓ New cell — primary positive signal
                self.visited[cell] = True
                reward = 1.0
                self.steps_since_new = 0
            else:
                # Revisit — mild penalty; agent should be free to use visited
                # cells as corridors without being heavily punished for it.
                reward = -0.2
                self.steps_since_new += 1

        self.count_steps += 1

        # ── Terminal condition ──────────────────────────────────────────────
        terminated = bool(self.visited.sum() >= self.total_free_cells)
        if terminated:
            reward += 30.0      # stronger completion bonus (was +20)

        # ── Per-step time penalty ───────────────────────────────────────────
        reward -= 0.02          # slightly higher than original -0.01

        # ── Stagnation penalty ──────────────────────────────────────────────
        # Kicks in after 7 consecutive steps without discovering a new cell.
        # (Threshold was 10; reduced to 7 for tighter anti-loop pressure.)
        if self.steps_since_new >= 7 and not terminated:
            reward -= 0.5       # was -1.0

        # ── Truncation ──────────────────────────────────────────────────────
        truncated = bool(self.count_steps >= self.max_steps and not terminated)
        if truncated:
            # -5 flat penalty + partial coverage credit (10× instead of 5×
            # to give a stronger gradient when coverage is, say, 90%)
            reward -= 5.0
            reward += 10.0 * self._coverage_ratio()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering (unchanged from original)
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld — Coverage Path Planning")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        # Visited cells (light green)
        for x in range(self.size):
            for y in range(self.size):
                if self.visited is not None and self.visited[x, y]:
                    pygame.draw.rect(
                        canvas,
                        (144, 238, 144),
                        pygame.Rect(pix * x, pix * y, pix, pix),
                    )

        # Obstacles (black)
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(pix * obs[0], pix * obs[1], pix, pix),
            )

        # Agent (blue circle)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location + 0.5) * pix).astype(int),
            int(pix / 3),
        )

        # Grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix * x), (self.window_size, pix * x), width=2)
            pygame.draw.line(canvas, 0, (pix * x, 0), (pix * x, self.window_size), width=2)

        # HUD
        if self.visited is not None:
            try:
                font = pygame.font.SysFont("monospace", 18)
                pct = self._coverage_ratio() * 100
                label = font.render(
                    f"Coverage: {pct:.1f}%  Step: {self.count_steps}",
                    True,
                    (50, 50, 50),
                )
                canvas.blit(label, (6, 6))
            except Exception:
                pass

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
