# ENV_Patched.py — Environment with all discussed fixes
# - Robust global A* (keep path on failure, snap start/goal to nearest free)
# - Gated replanning: every N steps or when cross-track error (CTE) too large
# - Rewards: A* length shaping, recent-velocity alignment, CTE penalty,
#            idle penalty, collision penalty, success bonus
# - Uses recent *movement* for alignment (not just intended action)
# - Collision modes: "slide" (default) | "deflect" | "none"
# - No floor on success_radius (precise goals OK)
# - Optional extra features in observation (off by default for checkpoint compatibility)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from heapq import heappush, heappop


class Vector2DEnv(gym.Env):
    """2D navigation with rectangular obstacles and A*-driven shaping.

    Per-step reward (terms are weighted via *_scale / *_penalty params):
        r = scale * clip(L_{t-1}-L_t, ±astar_shaping_clip)
          + align_scale * cos(angle(recent_velocity, next_waypoint_dir))
          - cte_scale * cross_track_error
          + 1[idle]*idle_penalty
          + 1[collided]*(-collision_penalty)
          + 1[success]*R_SUCCESS

    Replanning: every astar_replan_steps OR if CTE > replan_cte_threshold_world.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 # === World / motion ===
                 map_range=12.8,
                 step_size=0.1,
                 max_steps=300,
                 success_radius=0.10,      # precise goals allowed; adjust as needed
                 player_size=(0.1, 0.1),

                 # === Maze (collision/visualization) ===
                 maze_cells=(25, 25),      # a bit denser than 7x7 to reduce long straights
                 maze_margin=0.2,
                 maze_variable_bars=False,
                 maze_bar_max_len=6,

                 # === A* grid (decoupled from maze resolution) ===
                 astar_grid=(256, 256),
                 astar_replan_steps=8,     # gate replanning (was 1)
                 replan_cte_threshold_frac=0.5,  # trigger if CTE > 0.5 * astar_cell_size

                 # === Collision handling ===
                 on_collision="slide",     # "slide" | "deflect" | "none"
                 deflect_angles_deg=(20, 35, 50),
                 deflect_scales=(1.0, 0.7, 0.4),
                 deflect_randomize=True,

                 # === Observation ===
                 obs_max_walls=256,
                 obs_with_extras=False,     # keep False for old checkpoints

                 # === Reward weights (rebalanced defaults) ===
                 astar_shaping_scale=1.0,
                 astar_shaping_clip=2.0,    # 0 → no clip
                 align_scale=3.0,           # encourage pointing toward next waypoint
                 cte_scale=1.0,             # penalize lateral deviation
                 waypoint_lookahead=2,
                 idle_penalty=-0.3,
                 idle_move_eps_frac=0.02,
                 collision_penalty=0.5,
                 R_SUCCESS=500.0,

                 # === Misc ===
                 goal_snap_on_success=True,
                 seed=None
                 ):
        super().__init__()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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

        # A*
        self.astar_rows = int(astar_grid[0])
        self.astar_cols = int(astar_grid[1])
        self.astar_replan_steps = int(astar_replan_steps)
        self.replan_cte_threshold_frac = float(replan_cte_threshold_frac)

        # Collision
        self.on_collision = str(on_collision)
        self.deflect_angles_deg = tuple(deflect_angles_deg)
        self.deflect_scales = tuple(deflect_scales)
        self.deflect_randomize = bool(deflect_randomize)

        # Observation
        self.obs_max_walls = int(obs_max_walls)
        self.obs_with_extras = bool(obs_with_extras)

        # Rewards
        self.astar_shaping_scale = float(astar_shaping_scale)
        self.astar_shaping_clip = float(astar_shaping_clip) if astar_shaping_clip is not None else 0.0
        self.align_scale = float(align_scale)
        self.cte_scale = float(cte_scale)
        self.waypoint_lookahead = int(max(0, waypoint_lookahead))
        self.idle_penalty = float(idle_penalty)
        self.idle_move_eps_frac = float(idle_move_eps_frac)
        self.collision_penalty = float(collision_penalty)
        self._R_SUCCESS = float(R_SUCCESS)

        self.goal_snap_on_success = bool(goal_snap_on_success)

        # --- Runtime state ---
        self.agent_pos = None
        self.goal_pos = None
        self._wall_centers = None
        self._wall_halves = None
        self.n_obs = 0
        self.obstacles = None
        self.obstacles_half = None
        self.obs_mask = None

        self._maze_grid = None
        self._maze_cell_size = None
        self._maze_origin = None

        self._astar_cell_size = None
        self._astar_origin = None
        self._astar_occ = None
        self._astar_path = None            # list[(r,c)]
        self._astar_path_world = None      # np.array[[x,y], ...]
        self._last_astar_plan_step = -1
        self._astar_len_prev = None

        self._last_action = np.zeros(2, dtype=np.float32)
        self._recent_vel = np.zeros(2, dtype=np.float32)

        self.steps = 0

        # --- Observation space ---
        base_dim = 6 + 6 * self.obs_max_walls
        extra_dim = 0
        if self.obs_with_extras:
            # prev_action(2) + recent_vel(2) + next_wp_dir(2) + cos(1) + cte(1)
            extra_dim = 8
        self._obs_dim = base_dim + extra_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # ---------------- World/grid helpers ----------------
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

    def _astar_cell_center_world(self, r, c):
        return self._astar_origin + np.array([c * self._astar_cell_size, r * self._astar_cell_size], dtype=np.float32)

    # ---------------- Maze generation ----------------
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
        # variable bars (optional)
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

    # ---------------- A* occupancy ----------------
    def _build_astar_occupancy(self):
        rows, cols = self.astar_rows, self.astar_cols
        occ = np.ones((rows, cols), dtype=np.int8)
        for r in range(rows):
            y = self._astar_origin[1] + r * self._astar_cell_size
            for c in range(cols):
                x = self._astar_origin[0] + c * self._astar_cell_size
                p = np.array([x, y], dtype=np.float32)
                if not self._collides(p):
                    occ[r, c] = 0
        return occ

    def _pos_to_astar_cell(self, p):
        r = int(np.floor((p[1] - self._astar_origin[1]) / self._astar_cell_size))
        c = int(np.floor((p[0] - self._astar_origin[0]) / self._astar_cell_size))
        r = max(0, min(self.astar_rows - 1, r))
        c = max(0, min(self.astar_cols - 1, c))
        return r, c

    def _nearest_free_cell(self, occ, rc, max_radius=6):
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

    def _astar_pathfind(self, occ, start_rc, goal_rc):
        rows, cols = occ.shape
        sr, sc = start_rc; gr, gc = goal_rc
        if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
            return None
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

    # ---------------- Collision / movement ----------------
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
            return tried, False, False
        # Axis slide
        tried_x = pos + np.array([dx, 0.0], dtype=np.float32)
        if not self._collides(tried_x):
            return tried_x, True, False
        tried_y = pos + np.array([0.0, dy], dtype=np.float32)
        if not self._collides(tried_y):
            return tried_y, True, False
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
                            return candidate, True, True
        # no solution — stay
        return pos.copy(), True, False

    # ---------------- Path utilities ----------------
    def _update_path_world(self):
        if self._astar_path is None:
            self._astar_path_world = None
            return
        pts = [self._astar_cell_center_world(r, c) for (r, c) in self._astar_path]
        self._astar_path_world = np.array(pts, dtype=np.float32)

    def _nearest_segment(self, p):
        if self._astar_path_world is None or len(self._astar_path_world) < 2:
            return None, None, None, None
        pts = self._astar_path_world
        best = (None, None, None, 1e9)
        for i in range(len(pts) - 1):
            a = pts[i]; b = pts[i + 1]
            ab = b - a
            ap = p - a
            denom = float(np.dot(ab, ab))
            t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            d = float(np.linalg.norm(p - proj))
            if d < best[3]:
                best = (i, i + 1, proj, d)
        return best

    def _next_waypoint_dir(self, p):
        if self._astar_path_world is None or len(self._astar_path_world) == 0:
            return np.zeros(2, dtype=np.float32)
        if len(self._astar_path_world) == 1:
            v = self._astar_path_world[0] - p
            n = np.linalg.norm(v); return (v / n if n > 1e-9 else np.zeros(2, np.float32))
        i0, i1, proj, _ = self._nearest_segment(p)
        if i0 is None:
            return np.zeros(2, dtype=np.float32)
        idx = min(i1 + self.waypoint_lookahead, len(self._astar_path_world) - 1)
        target = self._astar_path_world[idx]
        v = target - p
        n = np.linalg.norm(v)
        return (v / n if n > 1e-9 else np.zeros(2, dtype=np.float32))

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
            wp_dir = self._next_waypoint_dir(self.agent_pos)
            # alignment based on *recent velocity*
            vel = self._recent_vel
            nv = float(np.linalg.norm(vel))
            cos = 0.0
            if nv > 1e-9:
                vdir = vel / nv
                cos = float(np.clip(np.dot(vdir, wp_dir), -1.0, 1.0))
            _, _, _, cte = self._nearest_segment(self.agent_pos)
            if cte is None:
                cte = 0.0
            parts += [self._last_action, self._recent_vel, wp_dir, np.array([cos, cte], dtype=np.float32)]
        return np.concatenate(parts).astype(np.float32)

    # ---------------- Gym API ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Maze build
        grid = self._generate_maze_grid(self.maze_cols, self.maze_rows)
        maze_cell, _, maze_origin = self._maze_world_params()
        self._maze_grid = grid
        self._maze_cell_size = maze_cell
        self._maze_origin = maze_origin
        # Place agent & goal on free cells
        free = np.argwhere(grid == 0)
        if len(free) < 2:
            self.agent_pos = maze_origin.copy()
            self.goal_pos = maze_origin.copy()
        else:
            rng = random.Random()
            a_idx = rng.randrange(len(free))
            g_idx = rng.randrange(len(free))
            tries = 0
            while g_idx == a_idx and tries < 20:
                g_idx = rng.randrange(len(free)); tries += 1
            ar, ac = free[a_idx]
            gr, gc = free[g_idx]
            self.agent_pos = maze_origin + np.array([ac * maze_cell, ar * maze_cell], dtype=np.float32)
            self.goal_pos  = maze_origin + np.array([gc * maze_cell, gr * maze_cell], dtype=np.float32)
        # Walls cache
        wall_centers, wall_halves = self._maze_to_walls(grid, maze_origin, maze_cell)
        self._wall_centers = wall_centers
        self._wall_halves = wall_halves
        # A* world grid + occupancy
        astar_cell, _, astar_origin = self._astar_world_params()
        self._astar_cell_size = astar_cell
        self._astar_origin = astar_origin
        self._astar_occ = self._build_astar_occupancy()
        # Initial path (snap start/goal if blocked)
        sr, sc = self._pos_to_astar_cell(self.agent_pos)
        gr, gc = self._pos_to_astar_cell(self.goal_pos)
        s_rc = (sr, sc); g_rc = (gr, gc)
        if self._astar_occ[sr, sc] == 1:
            alt = self._nearest_free_cell(self._astar_occ, (sr, sc), max_radius=6)
            if alt is not None: s_rc = alt
        if self._astar_occ[gr, gc] == 1:
            alt = self._nearest_free_cell(self._astar_occ, (gr, gc), max_radius=6)
            if alt is not None: g_rc = alt
        self._astar_path = self._astar_pathfind(self._astar_occ, s_rc, g_rc)
        self._last_astar_plan_step = 0
        self._astar_len_prev = (len(self._astar_path) if self._astar_path is not None else None)
        self._update_path_world()
        # Kinematics
        self._last_action[:] = 0.0
        self._recent_vel[:] = 0.0
        self.steps = 0
        return self._pack_observation(), {}

    def step(self, action):
        # Cap magnitude to step_size while preserving direction
        norm = float(np.linalg.norm(action))
        if norm > 0:
            action = (action / norm) * min(norm, self.step_size)
        else:
            action = np.zeros(2, dtype=np.float32)
        old_pos = self.agent_pos.copy()
        # Move + collision resolution
        new_pos, collided, deflected = self._resolve_movement(old_pos, action)
        self.agent_pos = new_pos
        self.steps += 1
        # Kinematics
        self._recent_vel = (self.agent_pos - old_pos)
        self._last_action = np.array(action, dtype=np.float32)
        displacement = float(np.linalg.norm(self._recent_vel))
        # Termination
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.agent_pos))
        terminated = dist_to_goal < self.success_radius
        truncated = self.steps >= self.max_steps
        # Replan gating
        replan = False
        if (self._astar_path_world is None) or (self.steps - self._last_astar_plan_step >= self.astar_replan_steps):
            replan = True
        else:
            # cte trigger
            _, _, _, cte = self._nearest_segment(self.agent_pos)
            if cte is not None:
                if cte > self.replan_cte_threshold_frac * self._astar_cell_size:
                    replan = True
        if replan:
            sr, sc = self._pos_to_astar_cell(self.agent_pos)
            gr, gc = self._pos_to_astar_cell(self.goal_pos)
            s_rc = (sr, sc); g_rc = (gr, gc)
            if self._astar_occ[sr, sc] == 1:
                alt = self._nearest_free_cell(self._astar_occ, (sr, sc), max_radius=6)
                if alt is not None: s_rc = alt
            if self._astar_occ[gr, gc] == 1:
                alt = self._nearest_free_cell(self._astar_occ, (gr, gc), max_radius=6)
                if alt is not None: g_rc = alt
            new_path = self._astar_pathfind(self._astar_occ, s_rc, g_rc)
            if new_path is not None:
                self._astar_path = new_path
                self._last_astar_plan_step = self.steps
                self._update_path_world()
            # else: keep old path
        # === Reward terms ===
        reward = 0.0
        terms = {}
        # A* ΔL shaping
        cur_len = (len(self._astar_path) if self._astar_path is not None else None)
        dL_term = 0.0
        if (self._astar_len_prev is not None) and (cur_len is not None):
            dL = float(self._astar_len_prev - cur_len)
            dL_term = self.astar_shaping_scale * dL
            if self.astar_shaping_clip > 0.0:
                hi = self.astar_shaping_clip
                dL_term = float(np.clip(dL_term, -hi, hi))
        self._astar_len_prev = cur_len
        reward += dL_term; terms["astar_dL"] = dL_term
        # Alignment using *recent velocity*
        align = 0.0
        nv = float(np.linalg.norm(self._recent_vel))
        if nv > 1e-9 and self._astar_path_world is not None and len(self._astar_path_world) > 0:
            wp_dir = self._next_waypoint_dir(self.agent_pos)
            vdir = self._recent_vel / nv
            cos = float(np.clip(np.dot(vdir, wp_dir), -1.0, 1.0))
            align = self.align_scale * cos
        reward += align; terms["align"] = align
        # CTE penalty
        cte_pen = 0.0
        if self._astar_path_world is not None and len(self._astar_path_world) >= 2:
            _, _, _, cte = self._nearest_segment(self.agent_pos)
            if cte is not None:
                cte_pen = - self.cte_scale * float(cte)
        reward += cte_pen; terms["cte"] = cte_pen
        # Idle penalty
        idle = 0.0
        idle_eps = max(1e-6, self.idle_move_eps_frac * self.step_size)
        if (not terminated) and displacement < idle_eps:
            idle = self.idle_penalty
        reward += idle; terms["idle"] = idle
        # Collision penalty
        coll = 0.0
        if collided:
            coll = - self.collision_penalty
        reward += coll; terms["collision"] = coll
        # Success bonus
        succ = self._R_SUCCESS if terminated else 0.0
        reward += succ; terms["success"] = succ
        # Snap on success
        if terminated and self.goal_snap_on_success:
            self.agent_pos = self.goal_pos.copy()
        info = {
            "reward_terms": terms,
            "dist_to_goal": dist_to_goal,
            "astar_len": cur_len
        }
        return self._pack_observation(), float(reward), terminated, truncated, info

    # ---------------- Render ----------------
    def render(self, mode='human'):
        n_all = 0 if self._wall_centers is None else self._wall_centers.shape[0]
        print(f"[MAZE] Agent:{self.agent_pos} Goal:{self.goal_pos} Walls:{n_all} A* grid:{self.astar_rows}x{self.astar_cols}")

    # Property for success bonus (kept for interface parity)
    @property
    def R_SUCCESS(self):
        return self._R_SUCCESS

    @R_SUCCESS.setter
    def R_SUCCESS(self, v):
        self._R_SUCCESS = float(v)
