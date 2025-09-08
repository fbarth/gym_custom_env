from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# This code is based on the example from Gymnasium: 
# https://gymnasium.farama.org/introduction/create_custom_env/
#
# This environment implements a simple grid world without obstacles. 
# The agent (blue circle) must reach the target (red square) in as few steps as possible.
#
# The state is represented as a full matrix with the agent's and target's coordinates.
# The action space is discrete with 4 actions: move right, up, left, down.
# The agent receives a reward of 1 when it reaches the target, and 0 otherwise.
# The episode ends when the agent reaches the target.

class GridWorldRenderEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5):
        # The size of the square grid
        self.size = size
        self.window_size = 512

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # The state is represented with the agent's and target's location and the grid matrix
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "matrix": gym.spaces.Box(0, 2, shape=(size, size, 1), dtype=np.uint8)  # Changed to uint8
            }
        )

        self._matrix = np.zeros((size, size), dtype=np.uint8)  # Also change matrix dtype    

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "matrix": np.expand_dims(self._matrix, axis=-1)}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "size": self.size
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        
        self._matrix[self._agent_location[0], self._agent_location[1]] = 1
        self._matrix[self._target_location[0], self._target_location[1]] = 2

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def distance(self, location, target):
        x = (location[0] - target[0])*(location[0] - target[0])
        y = (location[1] - target[1])*(location[1] - target[1])
        return np.sqrt(x+y)

    def step(self, action):
        truncated = False

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Store previous distance for reward calculation
        prev_distance = self.distance(self._agent_location, self._target_location)
        old_location = self._agent_location.copy()

        # Update the agent's location to zero before moving
        self._matrix[self._agent_location[0], self._agent_location[1]] = 0

        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        self._matrix[self._agent_location[0], self._agent_location[1]] = 1

        # Calculate current distance
        current_distance = self.distance(self._agent_location, self._target_location)

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Improved reward function
        if terminated:
            reward = 100.0  # Large positive reward for reaching target
        else:
            # Distance-based reward: positive if getting closer, negative if getting farther
            distance_reward = (prev_distance - current_distance) * 10.0
        
            # Step penalty to encourage efficiency
            step_penalty = -1.0
        
            # Optional: Add penalty for staying in same position (if agent hits wall)
            same_position_penalty = -5.0 if np.array_equal(old_location,self._agent_location) else 0.0
        
            reward = distance_reward + step_penalty + same_position_penalty
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )    
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()