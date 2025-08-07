import gymnasium as gym
import numpy as np
from gymnasium import spaces


#환경 정의
class Vector2DEnv(gym.Env):
    def __init__(self, map_range=30.0, step_size=0.1, max_steps=300, threshold=0.1):
        super(Vector2DEnv, self).__init__()
        self.map_range = map_range
        self.step_size = step_size
        self.max_steps = max_steps
        self.threshold = threshold

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.random.uniform(-self.map_range, self.map_range, size=2)
        self.goal_pos = np.random.uniform(-self.map_range, self.map_range, size=2)
        self.steps = 0
        return self._get_state(), {}  # <-- info dict 추가

    def _get_state(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def step(self, action):
        norm = np.linalg.norm(action)
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)

        self.agent_pos += action
        self.steps += 1

        dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        terminated = dist < self.threshold
        truncated = self.steps >= self.max_steps
        reward = 1000.0 if terminated else -dist

        return self._get_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")