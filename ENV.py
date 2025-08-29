# ENV.py — Minimal Navigation Env (No A*, Terminal Reward Only)
# - 모든 중간 보상 제거, 도착 시에만 보상
# - A* 경로계획/관련 항목 완전 제거
# - 고정 맵(fixed_maze) / 고정 시작·목표(fixed_agent_goal) 옵션
# - 인스턴스별 RNG(self.rng / self.nprng)로 시드 고정
# - reset(seed=...) 반복 호출 시에도 위치가 매번 같아지지 않도록 보호장치(_ever_reset)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random


class Vector2DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 # === World / motion ===
                 map_range=12.8,
                 step_size=0.1,
                 max_steps=300,
                 success_radius=0.10,
                 player_size=(0.1, 0.1),

                 # === Maze ===
                 maze_cells=(9, 9),
                 maze_margin=0.2,
                 maze_variable_bars=False,
                 maze_bar_max_len=6,

                 # === Collision ===
                 on_collision="slide",   # "slide" | "none"

                 # === Observation ===
                 obs_max_walls=256,
                 obs_with_extras=False,

                 # === Reward ===
                 R_SUCCESS=500.0,

                 # === Fixed map / start-goal ===
                 fixed_maze=False,
                 fixed_agent_goal=False,

                 # === Seed (determinism) ===
                 seed=None,
                 ):
        super().__init__()

        # --- Per-instance RNG (중요!) ---
        self._seed_value = seed
        self.rng = random.Random(seed)                 # Python RNG
        self.nprng = np.random.default_rng(seed)       # NumPy RNG

        # --- Core params ---
        self.map_range = float(map_range)
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)
        self.player_half = np.array(player_size, dtype=np.float32) * 0.5

        # Maze
        self.maze_cols = int(maze_cells[0])
        self.maze_rows = int(maze_cells[1])
        self.maze_margin = float(maze_margin)
        self.maze_variable_bars = bool(maze_variable_bars)
        self.maze_bar_max_len = int(max(1, maze_bar_max_len))

        # Collision mode
        self.on_collision = "slide" if str(on_collision).lower() == "slide" else "none"

        # Observation
        self.obs_max_walls = int(obs_max_walls)
        self.obs_with_extras = bool(obs_with_extras)

        # Reward
        self._R_SUCCESS = float(R_SUCCESS)
        self.goal_snap_on_success = True

        # Fixed map / start-goal
        self.fixed_maze = bool(fixed_maze)
        self.fixed_agent_goal = bool(fixed_agent_goal)
        self._fixed_grid = None
        self._fixed_agent_pos = None
        self._fixed_goal_pos = None

        # --- Runtime state ---
        self.agent_pos = None
        self.goal_pos = None
        self._wall_centers = None
        self._wall_halves = None

        self._maze_grid = None
        self._maze_cell_size = None
        self._maze_origin = None

        self._last_action = np.zeros(2, dtype=np.float32)
        self._recent_vel = np.zeros(2, dtype=np.float32)
        self.steps = 0

        # reset 재시드 억제용 플래그
        self._ever_reset = False

        # --- Observation space ---
        base_dim = 6 + 6 * self.obs_max_walls
        extra_dim = 8 if self.obs_with_extras else 0
        self._obs_dim = base_dim + extra_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # ---------------- Maze helpers ----------------
    def _maze_world_params(self):
        world_w = world_h = 2.0 * self.map_range
        cell_size = min(
            (world_w - 2 * self.maze_margin) / self.maze_cols,
            (world_h - 2 * self.maze_margin) / self.maze_rows
        )
        tile_half = np.array([cell_size * 0.5, cell_size * 0.5], dtype=np.float32)
        origin = np.array([
            -self.map_range + self.maze_margin + tile_half[0],
            -self.map_range + self.maze_margin + tile_half[1],
        ], dtype=np.float32)
        return cell_size, tile_half, origin

    def _generate_maze_grid(self, cols, rows):
        """DFS 미로 생성: 1=벽, 0=길 (self.rng 사용으로 시드 고정)"""
        grid = np.ones((rows, cols), dtype=np.int8)
        start = (1, 1)
        stack = [start]
        grid[start[0], start[1]] = 0

        rng = self.rng  # 인스턴스 RNG

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr, nc] == 1:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
            # 셔플 (rng 사용)
            for i in range(len(neighbors) - 1, 0, -1):
                j = rng.randrange(i + 1)
                neighbors[i], neighbors[j] = neighbors[j], neighbors[i]

            carved = False
            for nr, nc, wr, wc in neighbors:
                if grid[nr, nc] == 1:
                    grid[wr, wc] = 0
                    grid[nr, nc] = 0
                    stack.append((nr, nc))
                    carved = True
                    break
            if not carved:
                stack.pop()
        return grid

    def _maze_to_walls(self, grid, origin, cell_size):
        """격자(1=벽)를 월드 좌표 사각형 벽들로 변환 (self.nprng 사용)"""
        centers, halves = [], []
        if not self.maze_variable_bars:
            tile_half = np.array([cell_size * 0.5, cell_size * 0.5], dtype=np.float32)
            wall_rc = np.argwhere(grid == 1)
            for (r, c) in wall_rc:
                ctr = origin + np.array([c * cell_size, r * cell_size], dtype=np.float32)
                centers.append(ctr); halves.append(tile_half.copy())
            return (np.stack(centers, axis=0) if centers else np.zeros((0, 2), np.float32),
                    np.stack(halves, axis=0) if halves else np.zeros((0, 2), np.float32))

        # variable bars
        rows, cols = grid.shape
        for r in range(rows):
            c = 0
            while c < cols:
                if grid[r, c] != 1:
                    c += 1; continue
                c0 = c
                while c < cols and grid[r, c] == 1:
                    c += 1
                run_len = c - c0
                i = 0
                while i < run_len:
                    seg_len = int(self.nprng.integers(1, min(self.maze_bar_max_len, run_len - i) + 1))
                    mid_col = c0 + i + seg_len * 0.5
                    center = origin + np.array([mid_col * cell_size, r * cell_size], dtype=np.float32)
                    half = np.array([0.5 * seg_len * cell_size, 0.5 * cell_size], dtype=np.float32)
                    centers.append(center); halves.append(half)
                    i += seg_len
        return (np.stack(centers, axis=0) if centers else np.zeros((0, 2), np.float32),
                np.stack(halves, axis=0) if halves else np.zeros((0, 2), np.float32))

    # ---------------- Collision / movement ----------------
    def _collides(self, new_center):
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            return False
        dx = np.abs(new_center[0] - self._wall_centers[:, 0]) <= (self.player_half[0] + self._wall_halves[:, 0])
        dy = np.abs(new_center[1] - self._wall_centers[:, 1]) <= (self.player_half[1] + self._wall_halves[:, 1])
        return bool(np.any(dx & dy))

    def _resolve_movement(self, pos, action):
        dx, dy = float(action[0]), float(action[1])
        tried = pos + np.array([dx, dy], dtype=np.float32)
        if not self._collides(tried):
            return tried, False
        if self.on_collision == "slide":
            tried_x = pos + np.array([dx, 0.0], dtype=np.float32)
            if not self._collides(tried_x):
                return tried_x, True
            tried_y = pos + np.array([0.0, dy], dtype=np.float32)
            if not self._collides(tried_y):
                return tried_y, True
        return pos.copy(), True

    # ---------------- Observation ----------------
    def _pack_observation(self):
        s = self.map_range
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            rel = np.zeros((0, 2), np.float32)
            hal = np.zeros((0, 2), np.float32)
            dist = np.zeros((0,), np.float32)
        else:
            rel_all = self._wall_centers - self.agent_pos[None, :]
            dist_all = np.linalg.norm(rel_all, axis=1)
            idx = np.argsort(dist_all)[:self.obs_max_walls]
            rel = rel_all[idx]
            hal = self._wall_halves[idx]
            dist = dist_all[idx]

        K = rel.shape[0]
        obs_rel = np.zeros((self.obs_max_walls, 2), np.float32)
        obs_hal = np.zeros((self.obs_max_walls, 2), np.float32)
        obs_d = np.zeros((self.obs_max_walls,), np.float32)
        mask = np.zeros((self.obs_max_walls,), np.float32)
        if K > 0:
            obs_rel[:K] = rel
            obs_hal[:K] = hal
            obs_d[:K] = dist
            mask[:K] = 1.0

        goal_rel = (self.goal_pos - self.agent_pos) / s

        parts = [
            (self.agent_pos / s),
            (self.goal_pos / s),
            goal_rel,
            (obs_rel.flatten() / s),
            (obs_hal.flatten() / s),
            (obs_d / s),
            mask
        ]

        if self.obs_with_extras:
            # A* 제거: next_dir/cte는 0으로 채움
            next_dir = np.zeros(2, dtype=np.float32)
            vel = self._recent_vel
            nv = float(np.linalg.norm(vel))
            cos = 0.0
            cte = 0.0
            parts += [self._last_action, self._recent_vel, next_dir, np.array([cos, cte], dtype=np.float32)]

        return np.concatenate(parts).astype(np.float32)

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        """
        - seed가 들어와도 '첫 reset' 또는 options={"force_reseed": True}일 때만 reseed.
          (테스트 코드가 매 에피소드마다 reset(seed=42)를 호출해도 위치가 고정되지 않도록 함)
        """
        options = options or {}

        # reseed 조건: (첫 reset) OR (강제 reseed 명시)
        if seed is not None and ((not self._ever_reset) or options.get("force_reseed", False)):
            self._seed_value = seed
            self.rng = random.Random(seed)
            self.nprng = np.random.default_rng(seed)

        # 미로 생성(또는 고정 미로 재사용)
        if self.fixed_maze and self._fixed_grid is not None:
            grid = self._fixed_grid.copy()
        else:
            grid = self._generate_maze_grid(self.maze_cols, self.maze_rows)
            if self.fixed_maze:
                self._fixed_grid = grid.copy()

        maze_cell, _, maze_origin = self._maze_world_params()
        self._maze_grid = grid
        self._maze_cell_size = maze_cell
        self._maze_origin = maze_origin

        # 벽 캐시
        wall_centers, wall_halves = self._maze_to_walls(grid, maze_origin, maze_cell)
        self._wall_centers = wall_centers
        self._wall_halves = wall_halves

        # 에이전트/목표 배치(고정 여부)
        if self.fixed_agent_goal and (self._fixed_agent_pos is not None) and (self._fixed_goal_pos is not None):
            self.agent_pos = self._fixed_agent_pos.copy()
            self.goal_pos  = self._fixed_goal_pos.copy()
        else:
            free = np.argwhere(grid == 0)
            if len(free) < 2:
                self.agent_pos = maze_origin.copy()
                self.goal_pos = maze_origin.copy()
            else:
                a_idx = self.rng.randrange(len(free))
                g_idx = self.rng.randrange(len(free))
                tries = 0
                while g_idx == a_idx and tries < 20:
                    g_idx = self.rng.randrange(len(free)); tries += 1
                ar, ac = free[a_idx]
                gr, gc = free[g_idx]
                self.agent_pos = maze_origin + np.array([ac * maze_cell, ar * maze_cell], dtype=np.float32)
                self.goal_pos  = maze_origin + np.array([gc * maze_cell, gr * maze_cell], dtype=np.float32)
            if self.fixed_agent_goal:
                self._fixed_agent_pos = self.agent_pos.copy()
                self._fixed_goal_pos = self.goal_pos.copy()

        # 운동 상태 초기화
        self._last_action[:] = 0.0
        self._recent_vel[:] = 0.0
        self.steps = 0
        self._ever_reset = True

        return self._pack_observation(), {}

    def step(self, action):
        # 최대 이동량 step_size로 정규화
        norm = float(np.linalg.norm(action))
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)
        else:
            action = np.zeros(2, dtype=np.float32)

        old_pos = self.agent_pos.copy()

        # 이동 + 충돌 처리
        new_pos, collided = self._resolve_movement(old_pos, action)
        self.agent_pos = new_pos
        self.steps += 1

        # 운동량 기록(옵션 관측용)
        self._recent_vel = (self.agent_pos - old_pos)
        self._last_action = np.array(action, dtype=np.float32)

        # 종료 판정
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.agent_pos))
        terminated = dist_to_goal < self.success_radius
        truncated = self.steps >= self.max_steps

        # === 보상: 도착 시에만 ===
        reward = self._R_SUCCESS if terminated else 0.0

        if terminated and self.goal_snap_on_success:
            self.agent_pos = self.goal_pos.copy()

        info = {
            "dist_to_goal": dist_to_goal,
            "collided": bool(collided),
        }
        return self._pack_observation(), float(reward), terminated, truncated, info

    # ---------------- Render ----------------
    def render(self, mode='human'):
        n_all = 0 if self._wall_centers is None else self._wall_centers.shape[0]
        print(f"[ENV] Agent:{self.agent_pos} Goal:{self.goal_pos} Walls:{n_all}")

    # Property for success bonus
    @property
    def R_SUCCESS(self):
        return self._R_SUCCESS

    @R_SUCCESS.setter
    def R_SUCCESS(self, v):
        self._R_SUCCESS = float(v)
