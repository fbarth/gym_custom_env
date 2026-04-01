from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorldRenderEnv.
#
# Unlike the navigation environment (grid_world_obstacles.py), there is no
# fixed target. The agent must visit every free cell in the grid at least once.
#
# Reward function (CPP):
#   +1.0   for stepping onto a cell not yet visited (new coverage)
#   -0.5   for stepping onto a cell already visited (revisit penalty)
#   +50.0  bonus when full coverage of all free cells is achieved
#   -0.05  per step (efficiency incentive)
#   -10.0  if max_steps is exhausted before full coverage
#
# The observation is a flat array:
#   [agent_x, agent_y, visited_map[0..size*size-1], neighbors[0..3]]
# where visited_map encodes:
#    1  = cell already visited by the agent
#    0  = free cell not yet visited
#   -1  = obstacle cell
#
# The episode terminates (terminated=True) when all free cells have been visited,
# or is truncated (truncated=True) when max_steps is reached.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.array([0, 0, 0, 0], dtype=int)  # right, up, left, down
        # visited_map is a 2D boolean array; True = visited
        self._visited_map = np.zeros((size, size), dtype=bool)
        self._free_cells = 0  # number of non-obstacle cells (set in reset)

        # Observation: agent (x,y) + flat visited_map (size*size) + neighbors (4)
        obs_size = 2 + size * size + 4
        low = np.full(obs_size, -1, dtype=int)
        high = np.full(obs_size, max(size - 1, 1), dtype=int)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=int)

        # 4 discrete actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cell_to_idx(self, location):
        return location[0] * self.size + location[1]

    def _visited_flat(self):
        """Return visited_map as a flat int array: 1=visited, 0=free, -1=obstacle."""
        flat = np.zeros(self.size * self.size, dtype=int)
        for x in range(self.size):
            for y in range(self.size):
                if any(np.array_equal(np.array([x, y]), loc) for loc in self.obstacles_locations):
                    flat[x * self.size + y] = -1
                elif self._visited_map[x, y]:
                    flat[x * self.size + y] = 1
        return flat

    def _get_obs(self):
        return np.concatenate([
            self._agent_location,
            self._visited_flat(),
            self._neighbors,
        ]).astype(int)

    def _get_info(self):
        visited_count = int(self._visited_map.sum())
        return {
            "visited_cells": visited_count,
            "free_cells": self._free_cells,
            "coverage_ratio": visited_count / self._free_cells if self._free_cells > 0 else 0.0,
            "size": self.size,
        }

    def _set_neighbors(self):
        directions = [
            np.array([1, 0]),
            np.array([0, -1]),
            np.array([-1, 0]),
            np.array([0, 1]),
        ]
        for i, d in enumerate(directions):
            neighbor = self._agent_location + d
            in_bounds = (0 <= neighbor[0] < self.size) and (0 <= neighbor[1] < self.size)
            is_obstacle = any(np.array_equal(neighbor, loc) for loc in self.obstacles_locations)
            self._neighbors[i] = 1 if (not in_bounds or is_obstacle) else 0

    def _count_free_cells(self):
        total = self.size * self.size
        return total - len(self.obstacles_locations)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._visited_map = np.zeros((self.size, self.size), dtype=bool)

        # Place agent at a random location
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles (not on agent)
        for _ in range(self.obs_quantity):
            obstacle = self._agent_location.copy()
            while (np.array_equal(obstacle, self._agent_location) or
                   any(np.array_equal(obstacle, loc) for loc in self.obstacles_locations)):
                obstacle = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle)

        self._free_cells = self._count_free_cells()

        # Mark the starting cell as visited
        self._visited_map[self._agent_location[0], self._agent_location[1]] = True

        self._set_neighbors()

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Obstacle collision: agent stays put
        if any(np.array_equal(new_location, loc) for loc in self.obstacles_locations):
            new_location = old_location

        self._agent_location = new_location
        self._set_neighbors()
        self.count_steps += 1

        # --- CPP reward ---
        is_new = not self._visited_map[self._agent_location[0], self._agent_location[1]]
        self._visited_map[self._agent_location[0], self._agent_location[1]] = True

        reward = -0.05  # per-step cost
        if is_new:
            reward += 1.0   # reward for new coverage
        else:
            reward -= 0.5   # revisit penalty

        # Check termination: full coverage of all free cells
        terminated = int(self._visited_map.sum()) == self._free_cells
        if terminated:
            reward += 50.0  # bonus for complete coverage

        truncated = False
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 10.0  # penalty for not finishing within budget

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld CPP")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        # Draw visited cells (light green)
        for x in range(self.size):
            for y in range(self.size):
                if self._visited_map[x, y]:
                    pygame.draw.rect(
                        canvas,
                        (144, 238, 144),
                        pygame.Rect(pix * x, pix * y, pix, pix),
                    )

        # Draw obstacles (black)
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(pix * obs[0], pix * obs[1], pix, pix),
            )

        # Draw agent (blue circle)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location + 0.5) * pix).astype(int),
            int(pix / 3),
        )

        # Grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, pix * i), (self.window_size, pix * i), width=2)
            pygame.draw.line(canvas, (0, 0, 0), (pix * i, 0), (pix * i, self.window_size), width=2)

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
