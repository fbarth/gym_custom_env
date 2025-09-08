from typing import Optional
import numpy as np
import gymnasium as gym

#
# This code is based on the example available at:
# https://gymnasium.farama.org/introduction/create_custom_env/
#
# The example above was adapted to create a 3D grid environment.
#

class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size
        self.count_steps = 0
        self.max_steps = 100

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(3,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(3,), dtype=int),
            }
        )

        # We have 6 actions, corresponding to "right", "up", "left", "down", "forward", "backward"
        self.action_space = gym.spaces.Discrete(6)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0, 0]),  # right
            1: np.array([0, 1, 0]),  # up
            2: np.array([-1, 0, 0]),  # left
            3: np.array([0, -1, 0]),  # down
            4: np.array([0, 0, 1]),  # forward
            5: np.array([0, 0, -1]),  # backward
        }

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
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
        self._agent_location = self.np_random.integers(0, self.size, size=3, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=3, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        self.count_steps = 0

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        reward = 0

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        if terminated:
            reward = 1

        if self.count_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        if truncated:
            reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info