import gymnasium as gym
import numpy as np
from gymnasium import spaces


#환경 정의
class Vector2DEnv(gym.Env):
    def __init__(self):
        super(Vector2DEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.max_steps = 100
        self.threshold = 0.1  # 거리 임계값
        self.reset()

    def reset(self):
        self.agent_pos = np.random.uniform(-1, 1, size=2)
        self.goal_pos = np.random.uniform(-1, 1, size=2)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def step(self, action):
        #self.agent_pos += np.clip(action, -0.1, 0.1)  # 이동 크기 제한
        norm = np.linalg.norm(action)
        if norm > 0.1:
            action = (action / norm) * 0.1

        self.agent_pos += action
        self.steps += 1

        dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        done = dist < self.threshold or self.steps >= self.max_steps
        reward = -dist if not done else 100.0

        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")