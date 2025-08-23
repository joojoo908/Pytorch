# ENV.py — A* cell-length shaping reward + Decoupled global A* grid
# (fixed: keep path & snap start/goal to nearest free cell)
# (added: idle penalty for near-zero movement)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from heapq import heappush, heappop  # A*


class Vector2DEnv(gym.Env):
    def __init__(self,
                 # === 월드/이동 ===
                 map_range=12.8,          # 한 변의 절반 → 전체 월드 길이 = 2*map_range
                 step_size=0.1,
                 max_steps=300,
                 threshold=0.1,           # 성공 반경 하한
                 player_size=(0.1, 0.1),

                 # === 미로(충돌/시각화용) ===
                 maze_cells=(31, 31),
                 maze_margin=0.2,
                 maze_variable_bars=False,
                 maze_bar_max_len=6,

                 # === 관측 패킹 ===
                 obs_max_walls=256,

                 # === 보상 ===
                 R_SUCCESS=500.0,          # 성공 보상(에피소드 종료 시 추가)

                 # === A* (경로 탐색 격자) ===
                 # astar_grid를 지정하면 미로 크기와 무관하게 전역 2D 격자를 구성해 A* 수행
                 # 예: astar_grid=(128,128)
                 astar_grid=(128, 128),
                 astar_replan_steps=1,    # A* 재계획 주기(스텝)

                 # === 충돌 시 이동 보조 ===
                 on_collision="deflect",  # "slide" | "deflect"
                 deflect_angles_deg=(15, 30, 45, 60, 75, 90),
                 deflect_scales=(1.0, 0.75, 0.5, 0.25),
                 deflect_randomize=True,

                 corner_assist=False,
                 corner_eps_frac_cell=0.10,
                 corner_eps_frac_step=0.50,

                 # === A* 셀 길이 변화 기반 shaping 보상 ===
                 astar_shaping_scale=2.0,   # 경로 셀 수가 1 줄어들 때 주는 보상
                 astar_shaping_clip=0,    # 스텝당 shaping 보상 절댓값 클립(0이면 끔)

                 # === NEW: 정지 페널티 ===
                 idle_penalty=-1.0,          # 가만히 있을 때 주는 패널티(음수)
                 idle_move_eps_frac=0.05     # 스텝 크기의 몇 % 미만이면 '정지'로 간주
                 ):
        super().__init__()

        # --- 기본 ---
        self.map_range = float(map_range)
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        # 성공 반경은 환경 안정성 위해 하한 적용
        self.success_radius = max(float(threshold), 0.25)
        self.player_half = np.array(player_size, dtype=np.float32) * 0.5

        # --- 미로 ---
        self.maze_cols = int(maze_cells[0])
        self.maze_rows = int(maze_cells[1])
        self.maze_margin = float(maze_margin)
        self.maze_variable_bars = bool(maze_variable_bars)
        self.maze_bar_max_len = int(max(1, maze_bar_max_len))

        # --- 관측 공간 ---
        self.obs_max_walls = int(obs_max_walls)
        # [agent(2), goal(2), goal_rel(2)] + [rel(2)*K] + [half(2)*K] + [dist*K] + [mask*K]
        obs_dim = 6 + 6 * self.obs_max_walls
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- 보상 ---
        self.R_SUCCESS = float(R_SUCCESS)
        self.astar_shaping_scale = float(astar_shaping_scale)
        self.astar_shaping_clip = float(astar_shaping_clip) if (astar_shaping_clip is not None) else 0.0

        # --- A* 전역 격자 ---
        self.astar_rows = int(astar_grid[0])
        self.astar_cols = int(astar_grid[1])
        self.astar_replan_steps = int(astar_replan_steps)

        # --- 충돌 대응 ---
        self.on_collision = str(on_collision)
        self.deflect_angles_deg = tuple(deflect_angles_deg)
        self.deflect_scales = tuple(deflect_scales)
        self.deflect_randomize = bool(deflect_randomize)

        self.corner_assist = bool(corner_assist)
        self.corner_eps_frac_cell = float(corner_eps_frac_cell)
        self.corner_eps_frac_step = float(corner_eps_frac_step)

        # --- NEW: 정지 페널티 설정 ---
        self.idle_penalty = float(idle_penalty)
        self.idle_move_eps_frac = float(idle_move_eps_frac)

        # --- 런타임 상태 ---
        self.agent_pos = None
        self.goal_pos = None

        # 벽(사각형 모음)
        self._wall_centers = None
        self._wall_halves = None

        # 관측 캐시
        self.obstacles = None
        self.obstacles_half = None
        self.obs_mask = None
        self.n_obs = 0

        # 미로 좌표계
        self._maze_grid = None
        self._maze_cell_size = None
        self._maze_origin = None

        # A* 좌표계/점유 격자
        self._astar_cell_size = None
        self._astar_origin = None
        self._astar_occ = None     # 0=free, 1=blocked
        self._astar_path = None    # [(r,c), ...] in A* grid
        self._last_astar_plan_step = -1

        # shaping을 위한 직전 경로 길이
        self._astar_len_prev = None

        self.steps = 0

    # ---------------- 월드-미로 좌표 ----------------
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

    def _astar_world_params(self):
        """A* 전역 격자(예: 128×128)의 cell_size/origin (미로와 별개)."""
        world_w = world_h = 2.0 * self.map_range
        cell_size = min(
            (world_w - 2 * self.maze_margin) / self.astar_cols,
            (world_h - 2 * self.maze_margin) / self.astar_rows
        )
        half = np.array([cell_size * 0.5, cell_size * 0.5], dtype=np.float32)
        origin = np.array([
            -self.map_range + self.maze_margin + half[0],
            -self.map_range + self.maze_margin + half[1],
        ], dtype=np.float32)
        return cell_size, half, origin

    # 전역 A* 셀 중심을 월드 좌표로 (시각화 헬퍼)
    def _astar_cell_center_world(self, r, c):
        return self._astar_origin + np.array([c * self._astar_cell_size, r * self._astar_cell_size], dtype=np.float32)

    # ---------------- 미로 생성 ----------------
    def _generate_maze_grid(self, cols, rows):
        grid = np.ones((rows, cols), dtype=np.int8)
        start = (1, 1)
        stack = [start]
        grid[start[0], start[1]] = 0
        rng = random.Random()
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr, nc] == 1:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
            rng.shuffle(neighbors)
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

    def _maze_place_agent_and_goal(self, grid, origin, cell_size):
        free = np.argwhere(grid == 0)
        if len(free) < 2:
            self.agent_pos = origin.copy()
            self.goal_pos = origin.copy()
            return
        rng = random.Random()
        a_idx = rng.randrange(len(free))
        g_idx = rng.randrange(len(free))
        tries = 0
        while g_idx == a_idx and tries < 20:
            g_idx = rng.randrange(len(free)); tries += 1
        ar, ac = free[a_idx]
        gr, gc = free[g_idx]
        self.agent_pos = origin + np.array([ac * cell_size, ar * cell_size], dtype=np.float32)
        self.goal_pos = origin + np.array([gc * cell_size, gr * cell_size], dtype=np.float32)

    def _maze_to_walls(self, grid, origin, cell_size):
        centers, halves = [], []
        if not self.maze_variable_bars:
            tile_half = np.array([cell_size * 0.5, cell_size * 0.5], dtype=np.float32)
            wall_rc = np.argwhere(grid == 1)
            for (r, c) in wall_rc:
                ctr = origin + np.array([c * cell_size, r * cell_size], dtype=np.float32)
                centers.append(ctr); halves.append(tile_half.copy())
            return (np.stack(centers, axis=0) if centers else np.zeros((0, 2), np.float32),
                    np.stack(halves, axis=0) if halves else np.zeros((0, 2), np.float32))

        # 가변 막대(옵션)
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
                    seg_len = int(np.random.randint(1, min(self.maze_bar_max_len, run_len - i) + 1))
                    mid_col = c0 + i + seg_len * 0.5
                    center = origin + np.array([mid_col * cell_size, r * cell_size], dtype=np.float32)
                    half = np.array([0.5 * seg_len * cell_size, 0.5 * cell_size], dtype=np.float32)
                    centers.append(center); halves.append(half)
                    i += seg_len
        return (np.stack(centers, axis=0) if centers else np.zeros((0, 2), np.float32),
                np.stack(halves, axis=0) if halves else np.zeros((0, 2), np.float32))

    # ---------------- 전역 A* 점유 격자 ----------------
    def _build_astar_occupancy(self):
        """월드 전체를 astar_rows×astar_cols로 샘플링해 0/1 점유 격자 생성."""
        rows, cols = self.astar_rows, self.astar_cols
        occ = np.ones((rows, cols), dtype=np.int8)  # 1=blocked, 0=free
        for r in range(rows):
            y = self._astar_origin[1] + r * self._astar_cell_size
            for c in range(cols):
                x = self._astar_origin[0] + c * self._astar_cell_size
                p = np.array([x, y], dtype=np.float32)
                if not self._collides(p):
                    occ[r, c] = 0
        return occ

    # ---------------- 좌표→셀 인덱스 ----------------
    def _pos_to_astar_cell(self, p):
        r = int(np.floor((p[1] - self._astar_origin[1]) / self._astar_cell_size))
        c = int(np.floor((p[0] - self._astar_origin[0]) / self._astar_cell_size))
        r = max(0, min(self.astar_rows - 1, r))
        c = max(0, min(self.astar_cols - 1, c))
        return r, c

    # ---------------- NEW: 막힌 셀에서 가장 가까운 빈 셀 찾기 ----------------
    def _nearest_free_cell(self, occ, rc, max_radius=6):
        """막힌 rc=(r,c)에서 가장 가까운 free(0) 셀을 찾는다. 없으면 None."""
        r0, c0 = int(rc[0]), int(rc[1])
        R, C = occ.shape
        if 0 <= r0 < R and 0 <= c0 < C and occ[r0, c0] == 0:
            return (r0, c0)
        for rad in range(1, max_radius + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < R and 0 <= c < C and occ[r, c] == 0:
                        return (r, c)
        return None

    # ---------------- A* (4-이웃, 맨해튼 h) ----------------
    def _astar_pathfind(self, occ, start_rc, goal_rc):
        rows, cols = occ.shape
        sr, sc = start_rc; gr, gc = goal_rc
        if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
            return None
        # 시작/목표가 막혀 있으면 실패
        if occ[sr, sc] == 1 or occ[gr, gc] == 1:
            return None

        def h(r, c): return abs(r - gr) + abs(c - gc)
        gscore = {(sr, sc): 0}
        came = {}
        heap = []
        heappush(heap, (h(sr, sc), 0, (sr, sc)))
        visited = set()

        while heap:
            f, g, (r, c) = heappop(heap)
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if (r, c) == (gr, gc):
                path = [(r, c)]
                guard = rows * cols + 5
                while (r, c) in came and guard > 0:
                    r, c = came[(r, c)]
                    path.append((r, c)); guard -= 1
                if guard == 0:
                    return None
                path.reverse()
                return path

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and occ[nr, nc] == 0:
                    tg = g + 1
                    if tg < gscore.get((nr, nc), 10**9):
                        gscore[(nr, nc)] = tg
                        came[(nr, nc)] = (r, c)
                        heappush(heap, (tg + h(nr, nc), tg, (nr, nc)))
        return None

    # ---------------- 충돌/이동 ----------------
    def _collides(self, new_center):
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            return False
        dx = np.abs(new_center[0] - self._wall_centers[:, 0]) <= (self.player_half[0] + self._wall_halves[:, 0])
        dy = np.abs(new_center[1] - self._wall_centers[:, 1]) <= (self.player_half[1] + self._wall_halves[:, 1])
        return bool(np.any(dx & dy))

    @staticmethod
    def _rot(vec, deg):
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        x, y = float(vec[0]), float(vec[1])
        return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

    def _resolve_movement(self, pos, action):
        dx, dy = float(action[0]), float(action[1])
        tried = pos + np.array([dx, dy], dtype=np.float32)
        if not self._collides(tried):
            return tried, False, False, False

        tried_x = pos + np.array([dx, 0.0], dtype=np.float32)
        if not self._collides(tried_x):
            return tried_x, True, True, False
        tried_y = pos + np.array([0.0, dy], dtype=np.float32)
        if not self._collides(tried_y):
            return tried_y, True, True, False

        if self.on_collision.lower() == "deflect":
            signs = [+1, -1]
            if self.deflect_randomize and random.random() < 0.5:
                signs.reverse()
            base = np.array([dx, dy], dtype=np.float32)
            for scale in self.deflect_scales:
                cand_base = base * float(scale)
                for ang in self.deflect_angles_deg:
                    for sgn in signs:
                        v = self._rot(cand_base, sgn * ang)
                        candidate = pos + v
                        if not self._collides(candidate):
                            return candidate, True, False, False
        return pos.copy(), True, False, True

    # ---------------- 관측 패킹 ----------------
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
        return np.concatenate(parts).astype(np.float32)

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1) 미로 생성(충돌/시각화용)
        grid = self._generate_maze_grid(self.maze_cols, self.maze_rows)
        maze_cell, _, maze_origin = self._maze_world_params()
        self._maze_grid = grid
        self._maze_cell_size = maze_cell
        self._maze_origin = maze_origin

        self._maze_place_agent_and_goal(grid, maze_origin, maze_cell)

        wall_centers, wall_halves = self._maze_to_walls(grid, maze_origin, maze_cell)
        self._wall_centers = wall_centers
        self._wall_halves = wall_halves

        # 관측 캐시
        if wall_centers.shape[0] > 0:
            rel_all = wall_centers - self.agent_pos[None, :]
            idx = np.argsort(np.linalg.norm(rel_all, axis=1))[:self.obs_max_walls]
            K = idx.shape[0]
            self.n_obs = K
            self.obstacles = wall_centers[idx].copy()
            self.obstacles_half = wall_halves[idx].copy()
            self.obs_mask = np.zeros((self.obs_max_walls,), dtype=np.float32)
            self.obs_mask[:K] = 1.0
        else:
            self.n_obs = 0
            self.obstacles = np.zeros((0, 2), np.float32)
            self.obstacles_half = np.zeros((0, 2), np.float32)
            self.obs_mask = np.zeros((self.obs_max_walls,), np.float32)

        # 2) 전역 A* 좌표/점유격자 구축
        astar_cell, _, astar_origin = self._astar_world_params()
        self._astar_cell_size = astar_cell
        self._astar_origin = astar_origin
        self._astar_occ = self._build_astar_occupancy()

        # 3) 초기 A* 경로 (전역 격자에서) — 막힌 셀이면 스냅해서 시도
        sr, sc = self._pos_to_astar_cell(self.agent_pos)
        gr, gc = self._pos_to_astar_cell(self.goal_pos)
        s_rc = (sr, sc)
        g_rc = (gr, gc)
        if self._astar_occ[sr, sc] == 1:
            alt = self._nearest_free_cell(self._astar_occ, (sr, sc), max_radius=6)
            if alt is not None:
                s_rc = alt
        if self._astar_occ[gr, gc] == 1:
            alt = self._nearest_free_cell(self._astar_occ, (gr, gc), max_radius=6)
            if alt is not None:
                g_rc = alt

        self._astar_path = self._astar_pathfind(self._astar_occ, s_rc, g_rc)
        self._last_astar_plan_step = 0

        # 초기 경로 길이 저장
        self._astar_len_prev = (len(self._astar_path) if self._astar_path is not None else None)

        self.steps = 0
        return self._pack_observation(), {}

    def step(self, action):
        # 액션 정규화(스텝 크기 제한)
        norm = np.linalg.norm(action)
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)

        old_pos = self.agent_pos.copy()

        # 이동 & 충돌 대응
        new_pos, collided, slid, corner_block = self._resolve_movement(old_pos, action)
        if self.corner_assist and corner_block:
            eps = float(min(self.corner_eps_frac_cell * (self._maze_cell_size or 1.0),
                            self.corner_eps_frac_step * self.step_size))
            for dx, dy in [(eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps)]:
                probe = old_pos + np.array([dx, dy], dtype=np.float32)
                if not self._collides(probe):
                    new_pos = probe
                    break

        self.agent_pos = new_pos
        self.steps += 1

        # NEW: 이번 스텝에서 실제 이동 거리 (정지 판정용)
        displacement = float(np.linalg.norm(self.agent_pos - old_pos))
        idle_eps = max(1e-6, self.idle_move_eps_frac * self.step_size)

        # 종료/트렁케이트 판정
        dist = np.linalg.norm(self.goal_pos - self.agent_pos)
        terminated = dist < self.success_radius
        truncated = self.steps >= self.max_steps

        # ---- A* 재계획 (전역 격자) ----
        sr, sc = self._pos_to_astar_cell(self.agent_pos)
        gr, gc = self._pos_to_astar_cell(self.goal_pos)
        need = (self._astar_path is None) or ((self.steps - self._last_astar_plan_step) >= self.astar_replan_steps)
        if need:
            s_rc = (sr, sc)
            g_rc = (gr, gc)
            # 시작/목표 셀이 막히면 가장 가까운 빈 셀로 스냅
            if self._astar_occ[sr, sc] == 1:
                alt = self._nearest_free_cell(self._astar_occ, (sr, sc), max_radius=6)
                if alt is not None:
                    s_rc = alt
            if self._astar_occ[gr, gc] == 1:
                alt = self._nearest_free_cell(self._astar_occ, (gr, gc), max_radius=6)
                if alt is not None:
                    g_rc = alt

            new_path = self._astar_pathfind(self._astar_occ, s_rc, g_rc)
            if new_path is not None:
                self._astar_path = new_path
                self._last_astar_plan_step = self.steps
            else:
                # 실패해도 기존 경로는 유지 (지우지 않음)
                pass

        # ---- A* 경로 길이 변화 기반 shaping 보상 ----
        reward_shaping = 0.0
        cur_len = (len(self._astar_path) if self._astar_path is not None else None)
        if (self._astar_len_prev is not None) and (cur_len is not None):
            dL = float(self._astar_len_prev - cur_len)  # 줄어들면 +, 늘어나면 -
            reward_shaping = self.astar_shaping_scale * dL
            if self.astar_shaping_clip > 0.0:
                hi = self.astar_shaping_clip
                if reward_shaping > hi:
                    reward_shaping = hi
                if reward_shaping < -hi:
                    reward_shaping = -hi

        # 다음 스텝을 위해 현재 길이를 저장
        self._astar_len_prev = cur_len

        # ---- 총 보상: shaping + (성공 시 추가 보상) ----
        reward = reward_shaping + (self.R_SUCCESS if terminated else 0.0)

        # NEW: '정지'로 판단되면 패널티 적용(성공 스텝은 제외)
        if (not terminated) and (displacement < idle_eps):
            reward += self.idle_penalty

        return self._pack_observation(), float(reward), terminated, truncated, {}

    def render(self, mode='human'):
        n_all = 0 if self._wall_centers is None else self._wall_centers.shape[0]
        print(f"[MAZE] Agent:{self.agent_pos} Goal:{self.goal_pos} Walls:{n_all} "
              f"A* grid:{self.astar_rows}x{self.astar_cols}")
