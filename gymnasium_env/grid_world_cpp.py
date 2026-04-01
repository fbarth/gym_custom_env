from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

#
# Coverage Path Planning (CPP) Environment
#
# This environment implements a Coverage Path Planning task where the agent must visit
# all (or as many as possible) cells in a grid. The agent is rewarded for visiting new cells
# and penalized for revisiting cells or taking too many steps.
#
# The state is represented as a dictionary with:
# - agent: the agent's (x, y) location
# - target: the target's (x, y) location (goal is to visit all cells, not reach a specific target)
# - visited: a flattened array indicating which cells have been visited
#
# The action space is discrete with 4 actions: move right, up, left, down.
#
# The reward function encourages:
# - Positive reward for visiting new cells (+1)
# - Small penalty for each step taken (-0.1)
# - Penalty for revisiting cells (-1)
# - Bonus reward for complete coverage (+100)
#
# The episode ends when the agent reaches maximum steps or achieves 100% coverage.

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, max_steps: int = 500):
        """
        Initialize the Coverage Path Planning environment.
        
        Args:
            render_mode: None, "human", or "rgb_array"
            size: Size of the square grid (size x size)
            max_steps: Maximum number of steps per episode
        """
        self.size = size
        self.window_size = 512
        self.max_steps = max_steps
        self.current_step = 0
        
        # Grid to track visited cells
        self._visited_grid = np.zeros((size, size), dtype=bool)
        
        # Agent location
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        
        # Target (not used in pure CPP, but kept for compatibility)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's location and visited cells
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "visited": gym.spaces.Box(0, 1, shape=(size * size,), dtype=np.uint8),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """Get the current observation."""
        return {
            "agent": self._agent_location.astype(np.int32),
            "target": self._target_location.astype(np.int32),
            "visited": self._visited_grid.flatten().astype(np.uint8),
        }
    
    def _get_info(self):
        """Get additional info about the environment state."""
        coverage = np.sum(self._visited_grid) / (self.size * self.size) * 100
        return {
            "coverage": coverage,
            "cells_visited": np.sum(self._visited_grid),
            "total_cells": self.size * self.size,
            "step": self.current_step,
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset visited grid
        self._visited_grid = np.zeros((self.size, self.size), dtype=bool)
        
        # Place agent at random location
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        
        # Mark initial location as visited (convert x,y to y,x for grid indexing)
        self._visited_grid[self._agent_location[1], self._agent_location[0]] = True
        
        # Set target to a random location (not used in CPP, but kept for compatibility)
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        The reward function for CPP:
        - +1 for visiting a new cell
        - -1 for revisiting a cell
        - -0.1 for each step (movement cost)
        - +100 for achieving 100% coverage
        
        Args:
            action: Action to take (0: right, 1: up, 2: left, 3: down)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Move agent
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        ).astype(np.int32)
        self.current_step += 1

        # Check if this is a new cell (convert x,y to y,x for grid indexing)
        agent_y, agent_x = self._agent_location[1], self._agent_location[0]
        is_new_cell = not self._visited_grid[agent_y, agent_x]
        
        # Calculate reward
        reward = -0.1  # Cost for each step
        
        if is_new_cell:
            reward += 1.0  # Reward for visiting a new cell
            self._visited_grid[agent_y, agent_x] = True
        else:
            reward -= 1.0  # Penalty for revisiting a cell
        
        # Check coverage
        total_cells = self.size * self.size
        visited_cells = np.sum(self._visited_grid)
        coverage = visited_cells / total_cells
        
        # Bonus for full coverage
        if coverage >= 1.0:
            reward += 100.0
        
        # Determine if episode should end
        terminated = bool(coverage >= 1.0)  # Episode ends if full coverage is achieved
        truncated = bool(self.current_step >= self.max_steps)  # Episode ends if max steps exceeded
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render the environment using pygame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        
        pix_square_size = self.window_size / self.size
        
        # Draw visited cells in light gray
        for i in range(self.size):
            for j in range(self.size):
                if self._visited_grid[i, j]:
                    pygame.draw.rect(
                        canvas,
                        (200, 200, 200),  # Light gray for visited cells
                        pygame.Rect(
                            (j * pix_square_size, i * pix_square_size),
                            (pix_square_size, pix_square_size),
                        ),
                    )
        
        # Draw agent as a blue circle
        agent_center_x = (self._agent_location[0] + 0.5) * pix_square_size
        agent_center_y = (self._agent_location[1] + 0.5) * pix_square_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue
            (agent_center_x, agent_center_y),
            pix_square_size / 3,
        )

        # Draw grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # Draw coverage info
        coverage = np.sum(self._visited_grid) / (self.size * self.size) * 100
        font = pygame.font.Font(None, 36)
        text = font.render(f"Coverage: {coverage:.1f}%", True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
