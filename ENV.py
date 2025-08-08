import gymnasium as gym
import numpy as np
from gymnasium import spaces

# 환경 정의
class Vector2DEnv(gym.Env):
    def __init__(self, map_range=12.8, step_size=0.1, max_steps=300, threshold=0.1):
        super(Vector2DEnv, self).__init__()
        self.map_range = map_range
        self.step_size = step_size
        self.max_steps = max_steps

        # ✅ 2번 정책: 목표 근처 강제 종료
        #    threshold가 너무 작으면 알짱거림이 생기니, 최소 0.25로 올려 강제 종료 반경을 확보
        self.success_radius = max(threshold, 0.25)
        self.threshold = self.success_radius  # ModelTest의 성공 판정과 일치시키기 위해 동기화

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # 보상 shaping용 상태
        self.prev_dist = None
        self.prev_action = None

        # 보상 계수 (초기 가이드값)
        self.K = 1.0          # 거리 개선 보상 계수
        self.C = 0.01         # 시간 페널티(스텝당)
        self.R_SUCCESS = 150  # 도달 보너스(너무 크지 않게)

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.random.uniform(-self.map_range, self.map_range, size=2)
        self.goal_pos = np.random.uniform(-self.map_range, self.map_range, size=2)
        self.steps = 0
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.prev_dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)

    def step(self, action):
        # step_size에 맞춰 이동량 제한
        norm = np.linalg.norm(action)
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)

        self.agent_pos += action
        self.steps += 1

        # 거리 계산
        dist = np.linalg.norm(self.goal_pos - self.agent_pos)

        # ✅ 강제 종료: 목표 근처 반경(success_radius) 안이면 즉시 종료
        terminated = dist < self.success_radius
        truncated  = self.steps >= self.max_steps

        # ---- 보상 설계 (Potential-based shaping + time penalty) ----
        dist_norm_prev = self.prev_dist / self.map_range
        dist_norm      = dist / self.map_range
        shaped = self.K * (dist_norm_prev - dist_norm)   # 가까워지면 +, 멀어지면 -

        shaped -= self.C  # 시간 페널티

        # (선택) 방향 정렬 보상
        to_goal = (self.goal_pos - self.agent_pos)
        if np.linalg.norm(to_goal) > 1e-6 and np.linalg.norm(action) > 1e-6:
            cos = np.dot(to_goal, action) / (np.linalg.norm(to_goal) * np.linalg.norm(action))
            shaped += 0.02 * float(cos)

        # (선택) 급격한 조향 페널티
        jerk = np.linalg.norm(action - self.prev_action)
        shaped -= 0.005 * float(jerk)

        # 최종 보상 (성공 보너스 추가)
        reward = shaped + (self.R_SUCCESS if terminated else 0.0)

        # 상태 업데이트
        self.prev_dist = dist
        self.prev_action = action.copy()

        return self._get_state(), float(reward), terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
