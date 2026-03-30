from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorldRenderEnv.
#
# The agent (blue circle) must visit every free cell in the grid (green cells)
# while avoiding obstacles (black squares).  There is no fixed target: the
# episode ends when all free cells have been visited, or when the maximum
# number of steps is reached.
#
# Reward function (inspired by actor-critic CPP literature):
#   +1.0  : agent steps onto a cell it has never visited before
#   -0.5  : agent revisits a cell it already covered
#   -0.1  : step penalty applied every timestep (encourages efficiency)
#   +10.0 : bonus when 100 % of free cells are covered (episode ends)
#   -5.0  : penalty when max_steps is reached without full coverage
#
# Observation space (flattened array):
#   [agent_x, agent_y,                  (2 values)
#    neighbor_right, neighbor_up,        (4 values – 1 = blocked, 0 = free)
#    neighbor_left,  neighbor_down,
#    visited_grid (size × size)]         (size² binary values)
#
# Reference:
#   Schmid et al., "A Deep Reinforcement Learning Approach for the Patrolling
#   Problem of Water Resources Through Autonomous Surface Vehicles:
#   The Ypacarai Lake Case", IEEE Access, 2025 (doi:10.1109/ACCESS.2025.10946186)

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
        self._visited_grid = np.zeros((size, size), dtype=int)
        self._free_cells_count = 0

        # Observation: agent pos (2) + neighbors (4) + visited grid (size*size)
        obs_size = 2 + 4 + size * size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(size - 1, 1),
            shape=(obs_size,),
            dtype=int
        )

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
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        flattened = []
        flattened.extend(self._agent_location)
        flattened.extend(self._neighbors)
        flattened.extend(self._visited_grid.flatten())
        return np.array(flattened, dtype=int)

    def _get_info(self):
        covered = int(np.sum(self._visited_grid))
        return {
            "covered_cells": covered,
            "free_cells": self._free_cells_count,
            "coverage_ratio": covered / self._free_cells_count if self._free_cells_count > 0 else 0.0,
            "size": self.size,
        }

    def _set_neighbors(self):
        """Update the 4-element neighbor array: 1 = blocked (wall/obstacle), 0 = free."""
        directions = [
            np.array([1, 0]),   # right
            np.array([0, -1]),  # up
            np.array([-1, 0]),  # left
            np.array([0, 1]),   # down
        ]
        for i, direction in enumerate(directions):
            neighbor = self._agent_location + direction
            in_bounds = (0 <= neighbor[0] < self.size) and (0 <= neighbor[1] < self.size)
            is_obstacle = any(np.array_equal(neighbor, loc) for loc in self.obstacles_locations)
            self._neighbors[i] = 0 if (in_bounds and not is_obstacle) else 1

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.count_steps = 0
        self.obstacles_locations = []
        self._visited_grid = np.zeros((self.size, self.size), dtype=int)

        # Place agent at a random position
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles (never on the agent's starting cell)
        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while (np.array_equal(obstacle_location, self._agent_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        # Total cells the agent must cover
        self._free_cells_count = self.size * self.size - len(self.obstacles_locations)

        # Starting cell counts as visited
        x, y = self._agent_location
        self._visited_grid[x, y] = 1

        self._set_neighbors()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # Move agent (clipped to grid bounds)
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Stay in place if the new cell is an obstacle
        if any(np.array_equal(self._agent_location, loc) for loc in self.obstacles_locations):
            self._agent_location = old_location

        self._set_neighbors()
        self.count_steps += 1

        x, y = self._agent_location
        is_new_cell = self._visited_grid[x, y] == 0

        # --- Reward function ---
        if is_new_cell:
            self._visited_grid[x, y] = 1
            reward = 1.0    # new cell discovered
        else:
            reward = -0.5   # revisit penalty

        reward -= 0.1       # step penalty (efficiency incentive)

        covered = int(np.sum(self._visited_grid))
        terminated = covered >= self._free_cells_count

        if terminated:
            reward += 10.0  # full coverage bonus

        truncated = False
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            reward -= 5.0   # timeout penalty

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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
            pygame.display.set_caption("GridWorld – Coverage Path Planning")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix = self.window_size / self.size

        # Visited cells – light green
        for xi in range(self.size):
            for yi in range(self.size):
                if self._visited_grid[xi, yi] == 1:
                    pygame.draw.rect(
                        canvas,
                        (144, 238, 144),
                        pygame.Rect(pix * xi, pix * yi, pix, pix),
                    )

        # Obstacles – black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(pix * obs[0], pix * obs[1], pix, pix),
            )

        # Agent – blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location[0] + 0.5) * pix,
             (self._agent_location[1] + 0.5) * pix),
            pix / 3,
        )

        # Grid lines
        for i in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix * i), (self.window_size, pix * i), width=3)
            pygame.draw.line(canvas, 0, (pix * i, 0), (pix * i, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
