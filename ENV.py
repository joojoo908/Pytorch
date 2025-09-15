# ENV.py — Sliding/Revert collision modes + Random-in-cell spawn + Robust geodesic sampling
# Actions:
#   action[0] in [-1,1] -> angle θ in [-pi, pi]
#   action[1] in [-1,1] -> speed s in [0, step_size]
#
# Rewards:
#   (1) Terminal success bonus
#   (2) Geodesic progress shaping  (from_start | delta), with robust sampling near walls
#   (3) Near-wall penalty          reward -= proximity_coef * max(0, threshold - clearance)
#   (4) Anti-stall penalty         if no best-distance update for >= patience steps
#   (5) (optional) Collision penalty (non-terminal)
#
# Movement/Collision:
#   - "slide": axis-separated sliding; choose the order (x→y vs y→x) that moves farther
#   - "revert": if a move collides, revert to previous position (no movement this step)
#
# Geodesic distance:
#   - Separate grid (default 512x512)
#   - Walls rasterized (optionally dilated by player size)
#   - Start/Goal are snapped to nearest free geodesic cell (no forced open)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import heapq
import random


class Vector2DEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 # === World / motion ===
                 map_range=12.8,
                 step_size=0.1,
                 max_steps=700,
                 success_radius=0.10,
                 player_size=(0.1, 0.1),

                 # === Maze (movement/collision) ===
                 maze_cells=(11, 11),
                 maze_margin=0.2,
                 maze_variable_bars=False,
                 maze_bar_max_len=6,

                 # === Collision ===
                 collision_terminate=True,       # True면 충돌시 에피소드 종료
                 collision_mode="slide",          # "slide" | "revert"
                 collision_penalty=0.3,           # 충돌(비종료) 시 즉시 감점

                 # === Observation ===
                 obs_max_walls=100,
                 obs_with_extras=False,           # True면 추가 항목 포함(아래 참고)

                 # === Terminal reward ===
                 R_SUCCESS=500.0,

                 # === Geodesic (distance/shaping) ===
                 geodesic_grid=(512, 512),        # geodesic-only grid resolution
                 geodesic_shaping=True,           # enable shaping
                 geodesic_coef=1.0,               # shaping scale
                 geodesic_positive_only=False,    # True: never penalize for getting farther
                 geodesic_clip=0.0,               # per-step clip for shaping increment (0=off)
                 geodesic_dilate_player=True,     # dilate walls by player half-size when rasterizing
                 geodesic_progress_mode="delta",  # "from_start" | "delta"

                 # === Near-wall penalty ===
                 proximity_penalty=False,         # 켬/끔
                 proximity_threshold=0.0,         # 월드 단위 임계거리
                 proximity_coef=0.0,              # 패널티 세기
                 proximity_clip=0.0,              # 0이면 클립 없음

                 # === Anti-stall (no-progress penalty) ===
                 stall_penalty_use=True,       # 정체 패널티 켬/끔
                 stall_patience=5,            # 갱신 없을 때 허용 스텝 수
                 stall_penalty_per_step=1.0,   # patience 이후 매 스텝 감점
                 stall_improve_eps=0.3,        # "개선"으로 인정할 최소 개선 폭
                 stall_use_geodesic=True,      # 가능한 경우 지오데식 거리로 측정

                 # === Fixed map / start-goal ===
                 fixed_maze=True,
                 fixed_agent_goal=False,

                 # === Seed ===
                 seed=None,
                 ):
        super().__init__()

        # --- RNG ---
        self._seed_value = seed
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

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

        # Collision
        self.collision_terminate = bool(collision_terminate)
        self.collision_mode = str(collision_mode)
        self.collision_penalty = float(collision_penalty)

        # Observation
        self.obs_max_walls = int(obs_max_walls)
        self.obs_with_extras = bool(obs_with_extras)

        # Rewards
        self._R_SUCCESS = float(R_SUCCESS)
        self.goal_snap_on_success = True

        # Geodesic options
        self.geo_rows = int(geodesic_grid[0])
        self.geo_cols = int(geodesic_grid[1])
        self.geo_use = bool(geodesic_shaping)
        self.geo_coef = float(geodesic_coef)
        self.geo_pos_only = bool(geodesic_positive_only)
        self.geo_clip = float(geodesic_clip)
        self.geo_dilate_player = bool(geodesic_dilate_player)
        self.geo_mode = str(geodesic_progress_mode).lower()  # "from_start" | "delta"

        # Near-wall penalty
        self.prox_use = bool(proximity_penalty)
        self.prox_thr = float(proximity_threshold)
        self.prox_coef = float(proximity_coef)
        self.prox_clip = float(proximity_clip)

        # Anti-stall config
        self.stall_use = bool(stall_penalty_use)
        self.stall_patience = int(stall_patience)
        self.stall_penalty_per_step = float(stall_penalty_per_step)
        self.stall_eps = float(stall_improve_eps)
        self.stall_use_geodesic = bool(stall_use_geodesic)

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

        # Geodesic cache/meta
        self._geo_map = None
        self._geo_prev = None
        self._geo_init = None
        self._geo_progress_given = 0.0
        self._geo_origin = None
        self._geo_cell_size = None
        self._geo_goal_rc = None

        # (angle, speed) 기록 + dx,dy 호환
        self._last_action = np.zeros(2, dtype=np.float32)          # (dx, dy) of last step
        self._last_angle_speed = np.zeros(2, dtype=np.float32)     # (theta, speed)
        self._recent_vel = np.zeros(2, dtype=np.float32)

        self.steps = 0
        self._ever_reset = False

        # Anti-stall runtime
        self._stall_best = None   # float: 관측된 최소 진행거리
        self._stall_wait = 0      # int: 갱신 없는 연속 스텝 수

        # ---------- Observation space ----------
        # 구성:
        #  agent_pos(2) + goal_pos(2)  +  per-wall[rel(2), half(2), mask(1)] * N
        base_dim = 4 + 5 * self.obs_max_walls
        # extras: last (angle, speed_norm), recent_vel(2), placeholder next_dir(2), zeros(2) = 8
        extra_dim = 8 if self.obs_with_extras else 0
        self._obs_dim = base_dim + extra_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        # 액션은 각도/속도 해석용 2D [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # ---------- Maze helpers ----------
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
        grid = np.ones((rows, cols), dtype=np.int8)
        start = (1, 1)
        stack = [start]
        grid[start[0], start[1]] = 0
        rng = self.rng
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid[nr, nc] == 1:
                    neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
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
        centers, halves = [], []
        if not self.maze_variable_bars:
            tile_half = np.array([cell_size * 0.5, cell_size * 0.5], dtype=np.float32)
            wall_rc = np.argwhere(grid == 1)
            for (r, c) in wall_rc:
                ctr = origin + np.array([c * cell_size, r * cell_size], dtype=np.float32)
                centers.append(ctr); halves.append(tile_half.copy())
            return (np.stack(centers, axis=0) if centers else np.zeros((0, 2), np.float32),
                    np.stack(halves, axis=0) if halves else np.zeros((0, 2), np.float32))
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

    # ---------- Geodesic grid helpers ----------
    def _setup_geodesic_grid_meta(self):
        world_w = world_h = 2.0 * self.map_range
        cw = world_w / float(self.geo_cols)
        ch = world_h / float(self.geo_rows)
        origin = np.array([-self.map_range + cw * 0.5,
                           -self.map_range + ch * 0.5], dtype=np.float32)
        self._geo_origin = origin
        self._geo_cell_size = np.array([cw, ch], dtype=np.float32)

    def _pos_to_geo_rc(self, p):
        cw, ch = self._geo_cell_size[0], self._geo_cell_size[1]
        r = int(np.floor((p[1] - self._geo_origin[1]) / ch))
        c = int(np.floor((p[0] - self._geo_origin[0]) / cw))
        r = max(0, min(self.geo_rows - 1, r))
        c = max(0, min(self.geo_cols - 1, c))
        return r, c

    def _geo_rc_to_world_center(self, r, c):
        cw, ch = self._geo_cell_size[0], self._geo_cell_size[1]
        return self._geo_origin + np.array([c * cw, r * ch], dtype=np.float32)

    def _rasterize_walls_to_geo_grid(self):
        rows, cols = self.geo_rows, self.geo_cols
        occ = np.zeros((rows, cols), dtype=np.uint8)
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            return occ
        cw, ch = self._geo_cell_size[0], self._geo_cell_size[1]
        half_cell = np.array([cw * 0.5, ch * 0.5], dtype=np.float32)
        cx = self._geo_origin[0] + np.arange(cols, dtype=np.float32) * cw
        cy = self._geo_origin[1] + np.arange(rows, dtype=np.float32) * ch
        grid_cx = np.broadcast_to(cx[None, :], (rows, cols))
        grid_cy = np.broadcast_to(cy[:, None], (rows, cols))
        for i in range(self._wall_centers.shape[0]):
            wc = self._wall_centers[i]
            wh = self._wall_halves[i]
            if self.geo_dilate_player:
                wh = wh + self.player_half
            maskx = np.abs(grid_cx - wc[0]) <= (half_cell[0] + wh[0] + 1e-9)
            masky = np.abs(grid_cy - wc[1]) <= (half_cell[1] + wh[1] + 1e-9)
            occ |= (maskx & masky).astype(np.uint8)
        return occ

    def _compute_geodesic_map_on_geo_grid(self, occ, goal_rc):
        rows, cols = occ.shape
        INF = np.inf
        dist = np.full((rows, cols), INF, dtype=float)
        gr, gc = goal_rc
        if not (0 <= gr < rows and 0 <= gc < cols) or occ[gr, gc] == 1:
            return dist
        dist[gr, gc] = 0.0
        pq = [(0.0, (gr, gc))]
        N8 = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
              (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]
        while pq:
            d,(r,c) = heapq.heappop(pq)
            if d != dist[r,c]:
                continue
            for dr,dc,w in N8:
                nr,nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and occ[nr, nc] == 0:
                    nd = d + w
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(pq, (nd, (nr, nc)))
        return dist

    # ---------- Collision / distances ----------
    def _collides(self, new_center):
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            return False
        dx = np.abs(new_center[0] - self._wall_centers[:, 0]) <= (self.player_half[0] + self._wall_halves[:, 0])
        dy = np.abs(new_center[1] - self._wall_centers[:, 1]) <= (self.player_half[1] + self._wall_halves[:, 1])
        return bool(np.any(dx & dy))

    def _nearest_wall_clearance(self, center):
        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            return float("inf")
        ax, ay = float(center[0]), float(center[1])
        ahx, ahy = float(self.player_half[0]), float(self.player_half[1])

        wc = self._wall_centers
        wh = self._wall_halves
        dx = np.maximum(0.0, np.abs(ax - wc[:, 0]) - (ahx + wh[:, 0]))
        dy = np.maximum(0.0, np.abs(ay - wc[:, 1]) - (ahy + wh[:, 1]))
        d = np.hypot(dx, dy)
        return float(np.min(d)) if d.size > 0 else float("inf")

    def _resolve_movement(self, pos, action_vec):
        """
        Sliding collision (기본) 또는 Revert 모드
        Returns: (new_pos, blocked_flag)
          - blocked_flag=True면 (revert) 부딪혀서 한 발짝도 못 움직였다는 뜻
        """
        dx, dy = float(action_vec[0]), float(action_vec[1])
        full_try = pos + np.array([dx, dy], dtype=np.float32)

        # Revert 모드: 부딪히면 이동 안 함
        if getattr(self, "collision_mode", "slide") != "slide":
            if self._collides(full_try):
                return pos.copy(), True
            return full_try, False

        # Slide 모드
        if not self._collides(full_try):
            return full_try, False

        # 축 분해: (x 후 y)
        cand_xy = pos.copy()
        moved_xy = False
        try_x = pos + np.array([dx, 0.0], dtype=np.float32)
        if not self._collides(try_x):
            cand_xy = try_x
            moved_xy = True
        try_xy = cand_xy + np.array([0.0, dy], dtype=np.float32)
        if not self._collides(try_xy):
            cand_xy = try_xy
            moved_xy = True

        # 축 분해: (y 후 x)
        cand_yx = pos.copy()
        moved_yx = False
        try_y = pos + np.array([0.0, dy], dtype=np.float32)
        if not self._collides(try_y):
            cand_yx = try_y
            moved_yx = True
        try_yx = cand_yx + np.array([dx, 0.0], dtype=np.float32)
        if not self._collides(try_yx):
            cand_yx = try_yx
            moved_yx = True

        # 더 멀리 이동한 후보 채택
        dist_xy = float(np.linalg.norm(cand_xy - pos))
        dist_yx = float(np.linalg.norm(cand_yx - pos))
        if dist_xy >= dist_yx:
            best = cand_xy
            moved = moved_xy
        else:
            best = cand_yx
            moved = moved_yx

        return (best if moved else pos.copy()), (not moved)

    # ---------- Robust geodesic sampling ----------
    def _geo_valid(self, d):
        return (d is not None) and np.isfinite(d) and (d < 1e17)

    def _geo_distance_robust(self, p, max_search=2):
        """
        현재 위치 p에서 지오데식 값을 읽되, 현재 셀이 벽/INF이면
        반경 max_search 내의 유효 셀에서 최솟값을 반환. 없으면 None.
        """
        if self._geo_map is None or self._geo_cell_size is None or self._geo_origin is None:
            return None
        r, c = self._pos_to_geo_rc(p)
        rows, cols = self.geo_rows, self.geo_cols

        best = None
        for rad in range(0, max_search + 1):
            r0 = max(0, r - rad); r1 = min(rows - 1, r + rad)
            c0 = max(0, c - rad); c1 = min(cols - 1, c + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    d = float(self._geo_map[rr, cc])
                    if self._geo_valid(d):
                        if (best is None) or (d < best):
                            best = d
            if best is not None:
                break
        return best

    # ---------- Free-cell snapping for start/goal ----------
    def _nearest_free_rc(self, r, c, occ, max_search=3):
        rows, cols = occ.shape
        if 0 <= r < rows and 0 <= c < cols and occ[r, c] == 0:
            return (r, c)
        for rad in range(1, max_search + 1):
            r0 = max(0, r - rad); r1 = min(rows - 1, r + rad)
            c0 = max(0, c - rad); c1 = min(cols - 1, c + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if occ[rr, cc] == 0:
                        return (rr, cc)
        return None

    def _snap_world_to_free_geo_cell_center(self, p, occ, max_search=3):
        r, c = self._pos_to_geo_rc(p)
        rc = self._nearest_free_rc(r, c, occ, max_search=max_search)
        if rc is None:
            return p
        rr, cc = rc
        return self._geo_rc_to_world_center(rr, cc)

    # ---------- Progress metric helper ----------
    def _progress_metric(self):
        """
        가능한 경우 지오데식 거리(robust)를 사용(더 '길 인식'에 맞음).
        지오데식이 불가/무효이면 유클리드 거리로 대체.
        값이 작을수록 목표에 가깝다.
        """
        use_geo = self.stall_use_geodesic and self.geo_use and (self._geo_map is not None)
        if use_geo:
            d = self._geo_distance_robust(self.agent_pos, max_search=3)
            if d is not None:
                return float(d)
        return float(np.linalg.norm(self.goal_pos - self.agent_pos))

    # ---------- Observation ----------
    def _pack_observation(self):
        s = self.map_range

        if self._wall_centers is None or self._wall_centers.shape[0] == 0:
            rel = np.zeros((0, 2), np.float32)
            hal = np.zeros((0, 2), np.float32)
        else:
            rel_all = self._wall_centers - self.agent_pos[None, :]
            dist_all = np.linalg.norm(rel_all, axis=1)
            idx = np.argsort(dist_all)[:self.obs_max_walls]
            rel = rel_all[idx]
            hal = self._wall_halves[idx]

        K = rel.shape[0]
        obs_rel = np.zeros((self.obs_max_walls, 2), np.float32)
        obs_hal = np.zeros((self.obs_max_walls, 2), np.float32)
        mask    = np.zeros((self.obs_max_walls,),   np.float32)
        if K > 0:
            obs_rel[:K] = rel
            obs_hal[:K] = hal
            mask[:K]    = 1.0

        parts = [
            (self.agent_pos / s),           # 2
            (self.goal_pos  / s),           # 2
            (obs_rel.flatten() / s),        # 2*N
            (obs_hal.flatten() / s),        # 2*N
            mask                            # 1*N
        ]

        if self.obs_with_extras:
            ang = float(self._last_angle_speed[0])
            spd = float(self._last_angle_speed[1] / max(1e-9, self.step_size))
            last_ang_spd = np.array([ang, spd], dtype=np.float32)
            next_dir = np.zeros(2, dtype=np.float32)  # placeholder
            parts += [last_ang_spd, self._recent_vel, next_dir, np.array([0.0, 0.0], dtype=np.float32)]

        return np.concatenate(parts).astype(np.float32)

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is not None and ((not self._ever_reset) or options.get("force_reseed", False)):
            self._seed_value = seed
            self.rng = random.Random(seed)
            self.nprng = np.random.default_rng(seed)

        # Maze build/reuse
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

        # Walls cache
        wall_centers, wall_halves = self._maze_to_walls(grid, maze_origin, maze_cell)
        self._wall_centers = wall_centers
        self._wall_halves = wall_halves

        # Place agent/goal
        free = np.argwhere(grid == 0)
        if self.fixed_agent_goal and (self._fixed_agent_pos is not None) and (self._fixed_goal_pos is not None):
            self.agent_pos = self._fixed_agent_pos.copy()
            self.goal_pos  = self._fixed_goal_pos.copy()
        else:
            if len(free) < 2:
                self.agent_pos = maze_origin.copy()
                self.goal_pos = maze_origin.copy()
            else:
                # Goal cell pick
                gr, gc = free[self.rng.randrange(len(free))]
                goal_center = maze_origin + np.array([gc * maze_cell, gr * maze_cell], dtype=np.float32)
                # Random position inside the cell (avoid edges by 90%)
                self.goal_pos = goal_center + (self.nprng.random(2) - 0.5) * maze_cell * 0.9

                # Agent cell pick (different from goal cell)
                ar, ac = free[self.rng.randrange(len(free))]
                tries = 0
                while (ar == gr and ac == gc) and tries < 20:
                    ar, ac = free[self.rng.randrange(len(free))]
                    tries += 1
                agent_center = maze_origin + np.array([ac * maze_cell, ar * maze_cell], dtype=np.float32)
                self.agent_pos = agent_center + (self.nprng.random(2) - 0.5) * maze_cell * 0.9

            if self.fixed_agent_goal:
                self._fixed_agent_pos = self.agent_pos.copy()
                self._fixed_goal_pos = self.goal_pos.copy()

        # Geodesic grid + occ
        self._setup_geodesic_grid_meta()
        occ = self._rasterize_walls_to_geo_grid()

        # Snap goal/agent to nearest free geodesic cell center if needed
        gr, gc = self._pos_to_geo_rc(self.goal_pos)
        if occ[gr, gc] == 1:
            self.goal_pos = self._snap_world_to_free_geo_cell_center(self.goal_pos, occ, max_search=3)

        ar, ac = self._pos_to_geo_rc(self.agent_pos)
        if occ[ar, ac] == 1:
            self.agent_pos = self._snap_world_to_free_geo_cell_center(self.agent_pos, occ, max_search=3)

        # Compute geodesic map (no forced open; we snapped instead)
        gr, gc = self._pos_to_geo_rc(self.goal_pos)
        self._geo_goal_rc = (gr, gc)
        self._geo_map = self._compute_geodesic_map_on_geo_grid(occ, self._geo_goal_rc) if self.geo_use else None

        # If agent is still on an unreachable cell, snap again
        if self.geo_use and self._geo_map is not None:
            ar, ac = self._pos_to_geo_rc(self.agent_pos)
            if not np.isfinite(float(self._geo_map[ar, ac])):
                self.agent_pos = self._snap_world_to_free_geo_cell_center(self.agent_pos, occ, max_search=3)

        # Initialize geodesic trackers (robust)
        if self.geo_use and self._geo_map is not None:
            d0 = self._geo_distance_robust(self.agent_pos, max_search=3)
            self._geo_prev = d0 if (d0 is not None) else None
            self._geo_init = self._geo_prev
            self._geo_progress_given = 0.0
        else:
            self._geo_prev = None
            self._geo_init = None
            self._geo_progress_given = 0.0

        # Motion state
        self._last_action[:] = 0.0
        self._last_angle_speed[:] = 0.0
        self._recent_vel[:] = 0.0
        self.steps = 0
        self._ever_reset = True

        # Anti-stall init
        cur = self._progress_metric()
        self._stall_best = float(cur)
        self._stall_wait = 0

        return self._pack_observation(), {}

    def step(self, action):
        """
        Interpret action as (angle, speed):
          angle in [-pi, pi], speed in [0, step_size]
        """
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            a = np.zeros(2, dtype=np.float32)

        # map to angle/speed
        theta = float(np.pi * np.clip(a[0], -1.0, 1.0))
        speed = float(((np.clip(a[1], -1.0, 1.0) + 1.0) * 0.5) * self.step_size)

        dx = speed * np.cos(theta)
        dy = speed * np.sin(theta)

        old_pos = self.agent_pos.copy()

        # Move with collision (slide or revert)
        new_pos, collided = self._resolve_movement(old_pos, np.array([dx, dy], dtype=np.float32))
        self.agent_pos = new_pos
        self.steps += 1

        # Kinematics
        self._recent_vel = (self.agent_pos - old_pos)
        self._last_action = np.array([dx, dy], dtype=np.float32)
        self._last_angle_speed = np.array([theta, speed], dtype=np.float32)

        # Termination
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.agent_pos))
        terminated = dist_to_goal < self.success_radius
        truncated = self.steps >= self.max_steps

        # --- Reward ---
        reward = 0.0
        terms = {}

        # Geodesic shaping (robust sampling)
        if self.geo_use and (self._geo_map is not None):
            d_now = self._geo_distance_robust(self.agent_pos, max_search=3)
            if d_now is None:
                # keep previous to avoid jumps
                d_now = self._geo_prev

            if d_now is not None:
                if self.geo_mode == "from_start":
                    if self._geo_init is not None:
                        raw_progress = self._geo_init - d_now
                        if self.geo_pos_only:
                            incr_raw = max(0.0, raw_progress - self._geo_progress_given)
                            self._geo_progress_given = max(self._geo_progress_given, raw_progress)
                        else:
                            incr_raw = raw_progress - self._geo_progress_given
                            self._geo_progress_given = raw_progress
                        incr = incr_raw
                        if self.geo_clip > 0.0:
                            incr = float(np.clip(incr, -self.geo_clip, self.geo_clip))
                        reward += self.geo_coef * incr
                        terms["geo_from_start"] = raw_progress
                        terms["geo_increment"] = incr
                else:  # "delta"
                    if self._geo_prev is not None:
                        delta = self._geo_prev - d_now
                        if self.geo_pos_only:
                            delta = max(0.0, delta)
                        if self.geo_clip > 0.0:
                            delta = float(np.clip(delta, -self.geo_clip, self.geo_clip))
                        reward += self.geo_coef * delta
                        terms["geo_delta"] = delta

                self._geo_prev = d_now

        # Near-wall penalty (linear wrt missing clearance)
        min_clear = self._nearest_wall_clearance(self.agent_pos)
        if self.prox_use and np.isfinite(min_clear) and (min_clear < self.prox_thr):
            pen = self.prox_thr - min_clear
            if self.prox_clip > 0.0:
                pen = float(np.clip(pen, 0.0, self.prox_clip))
            penalty = self.prox_coef * pen
            reward -= penalty
            terms["near_wall_penalty"] = -penalty
            terms["min_clearance"] = float(min_clear)

        # Anti-stall: no-progress penalty
        if self.stall_use:
            cur = self._progress_metric()  # 작을수록 좋음
            if (self._stall_best is None) or (cur < self._stall_best - self.stall_eps):
                self._stall_best = float(cur)
                self._stall_wait = 0
            else:
                self._stall_wait += 1
                if self._stall_wait >= self.stall_patience:
                    reward -= self.stall_penalty_per_step
                    terms["stall_penalty"] = terms.get("stall_penalty", 0.0) - self.stall_penalty_per_step
                    terms["stall_wait"] = int(self._stall_wait)
                    terms["stall_best"] = float(self._stall_best)
                    terms["stall_cur"] = float(cur)

        # (선택) 충돌 감점 (비종료)
        if collided and not self.collision_terminate and self.collision_penalty != 0.0:
            reward -= self.collision_penalty
            terms["collision_penalty"] = -self.collision_penalty

        # Optional: end episode on collision
        if collided and self.collision_terminate:
            terminated = True
            terms["collision_reset"] = True

        # Terminal success
        if terminated and (dist_to_goal < self.success_radius):
            reward += self._R_SUCCESS
            terms["success"] = self._R_SUCCESS
            if self.goal_snap_on_success:
                self.agent_pos = self.goal_pos.copy()
            # 성공 시 정체 상태 리셋(의미상)
            self._stall_best = 0.0
            self._stall_wait = 0

        info = {
            "dist_to_goal": dist_to_goal,
            "collided": bool(collided),
            "reward_terms": terms,
            "geo_dist": (float(self._geo_prev) if self._geo_prev is not None else None),
            "geo_grid_meta": {
                "origin": self._geo_origin.copy() if self._geo_origin is not None else None,
                "cell_size": self._geo_cell_size.copy() if self._geo_cell_size is not None else None,
            },
            "min_clearance": float(min_clear) if np.isfinite(min_clear) else None,
            "collision_mode": self.collision_mode
        }
        return self._pack_observation(), float(reward), terminated, truncated, info

    # ---------- Render ----------
    def render(self, mode='human'):
        n_all = 0 if self._wall_centers is None else self._wall_centers.shape[0]
        print(f"[ENV] Agent:{self.agent_pos} Goal:{self.goal_pos} Walls:{n_all}")

    # ---------- Property ----------
    @property
    def R_SUCCESS(self):
        return self._R_SUCCESS

    @R_SUCCESS.setter
    def R_SUCCESS(self, v):
        self._R_SUCCESS = float(v)
