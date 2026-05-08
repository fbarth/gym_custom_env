from typing import Optional
from collections import deque
import numpy as np
import gymnasium as gym
import pygame

#
# GridWorldCPPEnvV2 — Coverage Path Planning with partial (local-window) observation.
#
# Observation space (Dict — MultiInputPolicy):
#   "agent"         Box(3,)   — (x/N, y/N, coverage_ratio over reachable cells)
#   "neighbors"     Box(25,)  — local 5×5 window (0=free, 0.5=visited, 1=wall/obs)
#   "frontier"      Box(3,)   — BFS direction+distance to nearest unvisited cell
#   "visited_pooled" Box(128,) — 2×8×8 max-pool of visited map (cross-grid memory)
#   "progress"      Box(1,)   — count_steps / max_steps
#
# visited_pooled details:
#   Canal 0: max-pool of self.visited — which regions the agent has covered
#   Canal 1: one-hot of agent position in pooled space
#   Resolution fixed at 8×8 regardless of grid size → enables cross-grid transfer
#   without LSTM, using only cells the agent has actually visited.
#
# frontier details:
#   BFS from agent over non-obstacle cells.
#   Returns (dx/N, dy/N, dist/diameter) to nearest unvisited cell.
#   Provides global navigation hint without exposing the full map.
#
# action_masks():
#   Returns bool[4] — False for actions that hit a wall or known obstacle.
#   Used by MaskablePPO (sb3-contrib) to avoid sampling invalid actions.
#
# Reward structure:
#   R_NEW_CELL * (1 + coverage)  visiting a new cell (endgame-scaled)
#   R_REVISIT                    revisiting a cell
#   R_STEP                       per-step cost (waived when coverage >= 0.80)
#   OSCILLATION_PENALTY          extra hit when returning to a recent position
#   MILESTONE_BONUSES            one-time bonuses at 25/50/75% coverage
#   R_COMPLETE                   bonus on full reachable coverage
#
# Termination uses BFS-reachable cells, not accessible cells, so episodes with
# isolated free cells (unreachable due to obstacle layout) are still solvable.
#


class GridWorldCPPEnvV2(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    WINDOW_SIZE = 5   # local-view side (5×5 = partial obs for all grid sizes)
    POOL_SIZE   = 8   # visited_pooled resolution (fixed regardless of grid size)

    R_NEW_CELL = 1.0
    R_REVISIT  = -1.0
    R_STEP     = -0.1
    R_COMPLETE = 10.0

    MILESTONE_BONUSES   = {0.25: 2.0, 0.50: 3.0, 0.75: 5.0}
    HISTORY_SIZE        = 10
    OSCILLATION_PENALTY = -0.3

    def __init__(self, render_mode=None, size: int = 5,
                 obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size_px = 512
        self.obs_quantity = obs_quantity
        self.max_steps = max_steps
        self.count_steps = 0

        self.obstacles_map = np.zeros((size, size), dtype=bool)
        self.obstacles_locations: list = []
        self.visited = np.zeros((size, size), dtype=np.int8)
        self._agent_location = np.array([0, 0], dtype=int)
        self._milestones_hit: set = set()
        self._reachable_cells: int = size * size
        self._recent_positions: deque = deque(maxlen=self.HISTORY_SIZE)

        P = self.POOL_SIZE
        W = self.WINDOW_SIZE
        self.observation_space = gym.spaces.Dict({
            "agent":          gym.spaces.Box(0., 1., shape=(3,),    dtype=np.float32),
            "neighbors":      gym.spaces.Box(0., 1., shape=(W * W,), dtype=np.float32),
            "frontier":       gym.spaces.Box(-1., 1., shape=(3,),   dtype=np.float32),
            "visited_pooled": gym.spaces.Box(0., 1., shape=(2 * P * P,), dtype=np.float32),
            "progress":       gym.spaces.Box(0., 1., shape=(1,),    dtype=np.float32),
        })

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([ 1,  0]),  # right
            1: np.array([ 0, -1]),  # up
            2: np.array([-1,  0]),  # left
            3: np.array([ 0,  1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    # ── action masking ────────────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return bool[4]: True = action is valid (no wall, no obstacle)."""
        masks = np.ones(4, dtype=bool)
        for action, direction in self._action_to_direction.items():
            new_loc = self._agent_location + direction
            if not (0 <= new_loc[0] < self.size and 0 <= new_loc[1] < self.size):
                masks[action] = False
            elif self.obstacles_map[new_loc[0], new_loc[1]]:
                masks[action] = False
        return masks

    # ── observation helpers ───────────────────────────────────────────────────

    def _local_window(self) -> np.ndarray:
        half = self.WINDOW_SIZE // 2
        ax, ay = self._agent_location
        flat = np.ones(self.WINDOW_SIZE ** 2, dtype=np.float32)
        idx = 0
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                nx, ny = ax + di, ay + dj
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.obstacles_map[nx, ny]:
                        flat[idx] = 1.0
                    elif self.visited[nx, ny]:
                        flat[idx] = 0.5
                    else:
                        flat[idx] = 0.0
                idx += 1
        return flat

    def _visited_pooled(self) -> np.ndarray:
        """2×POOL×POOL max-pool of visited map + agent one-hot. Fixed size for all grids."""
        P = self.POOL_SIZE
        pool = np.zeros((2, P, P), dtype=np.float32)
        ax, ay = self._agent_location

        for pi in range(P):
            for pj in range(P):
                xi0 = int(pi * self.size / P)
                xi1 = min(max(int((pi + 1) * self.size / P), xi0 + 1), self.size)
                yi0 = int(pj * self.size / P)
                yi1 = min(max(int((pj + 1) * self.size / P), yi0 + 1), self.size)

                pool[0, pi, pj] = float(self.visited[xi0:xi1, yi0:yi1].any())
                if xi0 <= ax < xi1 and yi0 <= ay < yi1:
                    pool[1, pi, pj] = 1.0

        return pool.flatten()

    def _frontier_feature(self) -> np.ndarray:
        """BFS from agent to nearest unvisited reachable cell.
        Returns (dx/N, dy/N, dist/diameter) — zero vector when fully covered."""
        ax, ay = self._agent_location
        queue = deque([(ax, ay, 0)])
        seen = {(ax, ay)}

        while queue:
            x, y, d = queue.popleft()
            for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + ddx, y + ddy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and not self.obstacles_map[nx, ny]
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    if not self.visited[nx, ny]:
                        dx = (nx - ax) / max(self.size, 1)
                        dy = (ny - ay) / max(self.size, 1)
                        dist_norm = min((d + 1) / (np.sqrt(2) * self.size), 1.0)
                        return np.array([dx, dy, dist_norm], dtype=np.float32)
                    queue.append((nx, ny, d + 1))

        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _bfs_reachable(self, start) -> int:
        """Count cells reachable from start via BFS."""
        seen = {tuple(start)}
        queue = deque([tuple(start)])
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and not self.obstacles_map[nx, ny]
                        and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return len(seen)

    def _get_obs(self) -> dict:
        ax, ay = self._agent_location
        covered = int(self.visited.sum())
        coverage = covered / self._reachable_cells if self._reachable_cells > 0 else 1.0
        return {
            "agent":          np.array([ax / max(self.size - 1, 1),
                                        ay / max(self.size - 1, 1),
                                        coverage], dtype=np.float32),
            "neighbors":      self._local_window(),
            "frontier":       self._frontier_feature(),
            "visited_pooled": self._visited_pooled(),
            "progress":       np.array([self.count_steps / self.max_steps],
                                       dtype=np.float32),
        }

    def _get_info(self) -> dict:
        accessible = self.size ** 2 - int(self.obstacles_map.sum())
        covered = int(self.visited.sum())
        reachable = self._reachable_cells
        return {
            "coverage_ratio":  covered / accessible if accessible > 0 else 1.0,
            "reachable_ratio": covered / reachable  if reachable  > 0 else 1.0,
            "covered_cells":   covered,
            "accessible_cells": accessible,
            "reachable_cells": reachable,
            "steps":           self.count_steps,
        }

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_map[:] = False
        self.obstacles_locations = []
        self.visited[:] = 0
        self._milestones_hit = set()
        self._recent_positions.clear()

        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )

        for _ in range(self.obs_quantity):
            loc = self._agent_location.copy()
            while (np.array_equal(loc, self._agent_location)
                   or self.obstacles_map[loc[0], loc[1]]):
                loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(loc)
            self.obstacles_map[loc[0], loc[1]] = True

        x, y = self._agent_location
        self.visited[x, y] = 1
        self._reachable_cells = self._bfs_reachable(self._agent_location)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        new_loc = np.clip(self._agent_location + direction, 0, self.size - 1)
        if not self.obstacles_map[new_loc[0], new_loc[1]]:
            self._agent_location = new_loc

        self.count_steps += 1
        x, y = self._agent_location
        new_cell = (self.visited[x, y] == 0)
        self.visited[x, y] = 1

        covered = int(self.visited.sum())
        terminated = (covered == self._reachable_cells)
        truncated = (not terminated) and (self.count_steps >= self.max_steps)

        coverage = covered / self._reachable_cells if self._reachable_cells > 0 else 1.0

        # Step penalty waived at high coverage to ease endgame pressure
        step_cost = 0.0 if coverage >= 0.80 else self.R_STEP

        if new_cell:
            reward = self.R_NEW_CELL * (1.0 + coverage) + step_cost
        else:
            reward = self.R_REVISIT + step_cost

        if terminated:
            reward += self.R_COMPLETE

        # Anti-oscillation penalty
        pos = tuple(self._agent_location)
        if pos in self._recent_positions:
            reward += self.OSCILLATION_PENALTY
        self._recent_positions.append(pos)

        # Milestone bonuses
        for threshold, bonus in self.MILESTONE_BONUSES.items():
            if threshold not in self._milestones_hit and coverage >= threshold:
                reward += bonus
                self._milestones_hit.add(threshold)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_px, self.window_size_px)
            )
            pygame.display.set_caption("Coverage Path Planning (partial obs)")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_px, self.window_size_px))
        canvas.fill((255, 255, 255))
        sq = self.window_size_px / self.size

        for xi in range(self.size):
            for yi in range(self.size):
                if self.visited[xi, yi]:
                    pygame.draw.rect(canvas, (144, 238, 144),
                                     pygame.Rect(sq * xi, sq * yi, sq, sq))

        for obs in self.obstacles_locations:
            pygame.draw.rect(canvas, (0, 0, 0),
                             pygame.Rect(sq * obs[0], sq * obs[1], sq, sq))

        cx = (self._agent_location[0] + 0.5) * sq
        cy = (self._agent_location[1] + 0.5) * sq
        pygame.draw.circle(canvas, (0, 0, 255), (cx, cy), sq / 3)

        for i in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, sq * i),
                             (self.window_size_px, sq * i), width=3)
            pygame.draw.line(canvas, 0, (sq * i, 0),
                             (sq * i, self.window_size_px), width=3)

        if self.render_mode == "human":
            font = pygame.font.SysFont(None, 28)
            reachable = self._reachable_cells
            covered = int(self.visited.sum())
            label = font.render(
                f"Coverage: {covered}/{reachable} "
                f"({100 * covered // max(reachable, 1)}%)",
                True, (60, 60, 60),
            )
            canvas.blit(label, (4, 4))
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
