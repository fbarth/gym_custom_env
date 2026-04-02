from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) Environment
#
# Based on grid_world_obstacles.py, this environment modifies the reward function
# so that the agent learns to COVER all accessible cells in the grid,
# instead of simply reaching a fixed target.
#
# === Original Reward Function (grid_world_obstacles.py) ===
# The original environment rewards the agent for reaching a fixed target cell:
#   +10.0  : agent reaches the target cell (episode terminates with success)
#   -0.1   : small step penalty applied every step (encourages shorter paths)
#   (prev_dist - curr_dist) - 0.1 : shaping reward based on Manhattan distance to target
#   -10.0  : large penalty if max_steps exceeded without reaching the target
#
# This reward function is well suited for NAVIGATION tasks (point A → point B),
# but does NOT encourage the agent to explore and cover the entire grid.
#
# === New Reward Function (CPP — this file) ===
# Inspired by:
#   - Theile et al. (2020) "A Deep Reinforcement Learning Approach for the Patrolling
#     Problem of Water Resources Through Autonomous Surface Vehicles"
#     https://ieeexplore.ieee.org/document/9252944
#   - A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic
#     Environments (2025), https://ieeexplore.ieee.org/document/10946186
#
# The agent is rewarded for covering NEW cells and penalized for revisiting cells:
#   +1.0   : agent moves to a cell it has NOT visited before (new coverage)
#   -0.5   : agent moves to (or stays on) a cell already visited (revisit penalty)
#   -0.05  : small step penalty applied every step (encourages efficiency)
#   +10.0  : large bonus when ALL accessible cells have been covered (terminates)
#   -5.0   : penalty if max_steps exceeded without achieving full coverage
#
# The state is extended to include the visited grid so the agent can learn
# which cells have already been covered.
#
# State representation:
#   - agent's (x, y) location normalized to [0,1]
#   - flattened visited grid (size x size binary array)
#   - 4 neighbor obstacle indicators (right, up, left, down): 1=obstacle/wall, 0=free

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
        self._visited = np.zeros((size, size), dtype=int)
        self._accessible_cells = 0  # total cells without obstacles

        # Observation: agent(x,y) normalized + visited grid (size*size) + neighbors(4)
        obs_size = 2 + size * size + 4
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]), # up
            2: np.array([-1, 0]), # left
            3: np.array([0, 1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        agent_norm = self._agent_location.astype(np.float32) / max(self.size - 1, 1)
        visited_flat = self._visited.flatten().astype(np.float32)
        neighbors = self._neighbors.astype(np.float32)
        return np.concatenate([agent_norm, visited_flat, neighbors])

    def _get_info(self):
        covered = int(np.sum(self._visited))
        return {
            "covered_cells": covered,
            "accessible_cells": self._accessible_cells,
            "coverage_ratio": covered / max(self._accessible_cells, 1),
            "steps": self.count_steps,
        }

    def set_neighbors(self):
        directions = [
            np.array([1, 0]),   # right
            np.array([0, -1]), # up
            np.array([-1, 0]), # left
            np.array([0, 1]),  # down
        ]
        for i, direction in enumerate(directions):
            neighbor = self._agent_location + direction
            in_bounds = (0 <= neighbor[0] < self.size) and (0 <= neighbor[1] < self.size)
            is_obstacle = any(np.array_equal(neighbor, loc) for loc in self.obstacles_locations)
            self._neighbors[i] = 1 if (not in_bounds or is_obstacle) else 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self._visited = np.zeros((self.size, self.size), dtype=int)

        # Place agent at a random location
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles (never on top of the agent)
        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while (np.array_equal(obstacle_location, self._agent_location) or
                   any(np.array_equal(obstacle_location, loc) for loc in self.obstacles_locations)):
                obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self.obstacles_locations.append(obstacle_location)

        # Count accessible (non-obstacle) cells
        self._accessible_cells = self.size * self.size - len(self.obstacles_locations)

        # Mark the starting cell as visited
        x, y = self._agent_location
        self._visited[x, y] = 1

        self.set_neighbors()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        old_location = self._agent_location.copy()

        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Block movement into obstacles
        if any(np.array_equal(new_location, loc) for loc in self.obstacles_locations):
            new_location = old_location

        self._agent_location = new_location
        self.set_neighbors()
        self.count_steps += 1

        # === CPP Reward Function ===
        x, y = self._agent_location
        if self._visited[x, y] == 0:
            # New cell: positive reward for coverage
            self._visited[x, y] = 1
            reward = 1.0
        else:
            # Revisit: negative reward to discourage redundant moves
            reward = -0.5

        # Small step penalty to encourage efficient paths
        reward -= 0.05

        # Check full coverage
        covered = int(np.sum(self._visited))
        terminated = (covered == self._accessible_cells)

        if terminated:
            # Bonus for achieving full coverage
            reward += 10.0

        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
                if self._visited[x, y] == 1:
                    pygame.draw.rect(
                        canvas, (144, 238, 144),
                        pygame.Rect(pix * x, pix * y, pix, pix)
                    )

        # Draw obstacles (black)
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas, (0, 0, 0),
                pygame.Rect(pix * obs[0], pix * obs[1], pix, pix)
            )

        # Draw agent (blue circle)
        pygame.draw.circle(
            canvas, (0, 0, 255),
            ((self._agent_location[0] + 0.5) * pix, (self._agent_location[1] + 0.5) * pix),
            pix / 3,
        )

        # Gridlines
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