import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 환경 정의
class Vector2DEnv(gym.Env):
    def __init__(self, map_range=12.8, step_size=0.1, max_steps=300, threshold=0.1,
                 num_obstacles=3, obstacle_radius=0.5):
        super(Vector2DEnv, self).__init__()
        self.map_range = map_range
        self.step_size = step_size
        self.max_steps = max_steps
        self.success_radius = max(threshold, 0.25)
        self.threshold = self.success_radius

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius

        # 상태: agent(x,y) + goal(x,y) + 각 장애물(x,y)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 + 2 * self.num_obstacles,),  # 4(에이전트+목표) + 장애물 좌표
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.steps = 0
        self.prev_action = None
        self.prev_dist = None

        self.K = 1.0
        self.C = 0.01
        self.R_SUCCESS = 150

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.random.uniform(-self.map_range, self.map_range, size=2)
        self.goal_pos = np.random.uniform(-self.map_range, self.map_range, size=2)

        # 장애물 위치 랜덤 생성 (agent/goal과 너무 가까운 경우 재생성)
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                pos = np.random.uniform(-self.map_range, self.map_range, size=2)
                if np.linalg.norm(pos - self.agent_pos) > self.obstacle_radius * 2 \
                   and np.linalg.norm(pos - self.goal_pos) > self.obstacle_radius * 2:
                    self.obstacles.append(pos)
                    break
        self.obstacles = np.array(self.obstacles, dtype=np.float32)

        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([self.agent_pos, self.goal_pos, self.obstacles.flatten()]).astype(np.float32)

    def step(self, action):
        # 이동
        norm = np.linalg.norm(action)
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)
        new_pos = self.agent_pos + action

        # 장애물 충돌 체크
        collision = False
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) < self.obstacle_radius:
                collision = True
                break

        if not collision:
            self.agent_pos = new_pos  # 충돌 없으면 이동 반영
        else:
            # 충돌 시 페널티
            return self._get_state(), -5.0, False, False, {}

        self.steps += 1

        # 거리 계산
        dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        terminated = dist < self.success_radius
        truncated = self.steps >= self.max_steps

        # 보상 shaping
        dist_norm_prev = self.prev_dist / self.map_range
        dist_norm = dist / self.map_range
        shaped = self.K * (dist_norm_prev - dist_norm) - self.C

        to_goal = (self.goal_pos - self.agent_pos)
        if np.linalg.norm(to_goal) > 1e-6 and np.linalg.norm(action) > 1e-6:
            cos = np.dot(to_goal, action) / (np.linalg.norm(to_goal) * np.linalg.norm(action))
            shaped += 0.02 * float(cos)

        jerk = np.linalg.norm(action - self.prev_action)
        shaped -= 0.005 * float(jerk)

        reward = shaped + (self.R_SUCCESS if terminated else 0.0)
        self.prev_dist = dist
        self.prev_action = action.copy()

        return self._get_state(), float(reward), terminated, truncated, {}

