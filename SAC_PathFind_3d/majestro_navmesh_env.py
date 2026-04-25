from __future__ import annotations

import heapq
import importlib
import math
import os
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    try:
        import gym
        from gym import spaces
    except Exception:
        class _FallbackEnv:
            pass

        class _FallbackBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = tuple(shape)
                self.dtype = dtype

            def sample(self):
                low = np.full(self.shape, self.low, dtype=self.dtype)
                high = np.full(self.shape, self.high, dtype=self.dtype)
                return np.random.uniform(low, high).astype(self.dtype)

        class _FallbackSpaces:
            Box = _FallbackBox

        class _FallbackGym:
            Env = _FallbackEnv

        gym = _FallbackGym()
        spaces = _FallbackSpaces()


DT_POLYTYPE_GROUND = 0

ROLE_FRONT = 0
ROLE_COVER = 1
ROLE_BASE_MOVE = 2
ROLE_SURROUND = 3
ROLE_KITING = 4
ROLE_COUNT = 5
ROLE_NONE = -1
HEURISTIC_NAME_TO_ID = {
    "fixed": 0,
    "melee_dps": 1,
    "meleedps": 1,
    "ranged_dps": 2,
    "rangeddps": 2,
}
HEURISTIC_COUNT = len(HEURISTIC_NAME_TO_ID)
DT_VERTS_PER_POLYGON = 6
DT_NAVMESH_MAGIC = ord("D") << 24 | ord("N") << 16 | ord("A") << 8 | ord("V")
NAVMESHSET_MAGIC = ord("M") << 24 | ord("S") << 16 | ord("E") << 8 | ord("T")
NAVMESHSET_VERSION = 1


@dataclass
class Triangle:
    verts3d: np.ndarray
    verts2d: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    avg_height: float


def _nav_to_engine(v: Sequence[float]) -> np.ndarray:
    return np.array([float(v[2]) * 100.0, float(v[1]) * 100.0, float(v[0]) * 100.0], dtype=np.float32)


def _point_in_triangle_2d(point: np.ndarray, tri: np.ndarray, eps: float = 1e-5) -> bool:
    a = tri[0]
    b = tri[1]
    c = tri[2]
    v0 = c - a
    v1 = b - a
    v2 = point - a

    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False

    inv_den = 1.0 / den
    u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den
    w = 1.0 - u - v
    return (u >= -eps) and (v >= -eps) and (w >= -eps)


def _barycentric_height(point: np.ndarray, tri2d: np.ndarray, tri3d: np.ndarray) -> Optional[float]:
    a = tri2d[0]
    b = tri2d[1]
    c = tri2d[2]
    v0 = b - a
    v1 = c - a
    v2 = point - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-6:
        return None

    inv_den = 1.0 / den
    v = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den
    w = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den
    u = 1.0 - v - w
    if (u < -1e-4) or (v < -1e-4) or (w < -1e-4):
        return None
    heights = tri3d[:, 1]
    return float(u * heights[0] + v * heights[1] + w * heights[2])


class MajestroNavMeshData:
    def __init__(self, navmesh_path: str | Path):
        self.path = Path(navmesh_path)
        self.triangles: List[Triangle] = []
        self.bounds_min = np.zeros(2, dtype=np.float32)
        self.bounds_max = np.zeros(2, dtype=np.float32)
        self._load()

    def _load(self) -> None:
        raw = self.path.read_bytes()
        if len(raw) < 40:
            raise ValueError(f"NavMesh file too small: {self.path}")

        magic, version, num_tiles = struct.unpack_from("<3i", raw, 0)
        if magic != NAVMESHSET_MAGIC:
            raise ValueError(f"Unexpected navmesh magic: {hex(magic)}")
        if version != NAVMESHSET_VERSION:
            raise ValueError(f"Unexpected navmesh version: {version}")

        offset = 40
        all_min = np.array([np.inf, np.inf], dtype=np.float32)
        all_max = np.array([-np.inf, -np.inf], dtype=np.float32)

        for _ in range(num_tiles):
            tile_ref, data_size = struct.unpack_from("<II", raw, offset)
            offset += 8
            if tile_ref == 0 or data_size <= 0:
                break
            tile_data = raw[offset: offset + data_size]
            offset += data_size
            self._parse_tile(tile_data, all_min, all_max)

        if not self.triangles:
            raise ValueError(f"No walkable triangles parsed from {self.path}")

        self.bounds_min = all_min
        self.bounds_max = all_max

    def _parse_tile(self, tile_data: bytes, all_min: np.ndarray, all_max: np.ndarray) -> None:
        header = struct.unpack_from("<15i10f", tile_data, 0)
        magic = header[0]
        version = header[1]
        if magic != DT_NAVMESH_MAGIC or version != 7:
            return

        poly_count = int(header[6])
        vert_count = int(header[7])
        max_link_count = int(header[8])
        detail_mesh_count = int(header[9])
        detail_vert_count = int(header[10])
        detail_tri_count = int(header[11])
        bv_node_count = int(header[12])
        off_mesh_con_count = int(header[13])

        offset = 100

        verts = np.frombuffer(tile_data, dtype="<f4", count=vert_count * 3, offset=offset).reshape(vert_count, 3)
        offset += vert_count * 3 * 4

        polys = []
        for i in range(poly_count):
            poly_offset = offset + i * 32
            first_link = struct.unpack_from("<I", tile_data, poly_offset)[0]
            verts_idx = struct.unpack_from("<6H", tile_data, poly_offset + 4)
            neis = struct.unpack_from("<6H", tile_data, poly_offset + 16)
            flags = struct.unpack_from("<H", tile_data, poly_offset + 28)[0]
            vert_count_poly = tile_data[poly_offset + 30]
            area_and_type = tile_data[poly_offset + 31]
            polys.append((first_link, verts_idx, neis, flags, vert_count_poly, area_and_type))
        offset += poly_count * 32

        offset += max_link_count * 12

        details = []
        for i in range(detail_mesh_count):
            detail_offset = offset + i * 12
            vert_base, tri_base, vert_count_detail, tri_count_detail = struct.unpack_from("<IIBB", tile_data, detail_offset)
            details.append((vert_base, tri_base, vert_count_detail, tri_count_detail))
        offset += detail_mesh_count * 12

        detail_verts = np.frombuffer(tile_data, dtype="<f4", count=detail_vert_count * 3, offset=offset).reshape(detail_vert_count, 3)
        offset += detail_vert_count * 3 * 4

        detail_tris = np.frombuffer(tile_data, dtype=np.uint8, count=detail_tri_count * 4, offset=offset).reshape(detail_tri_count, 4)
        offset += detail_tri_count * 4

        offset += bv_node_count * 16
        offset += off_mesh_con_count * 36

        for poly_idx, poly in enumerate(polys):
            _, verts_idx, _, _, vert_count_poly, area_and_type = poly
            poly_type = area_and_type >> 6
            if poly_type != DT_POLYTYPE_GROUND or vert_count_poly < 3:
                continue

            base_verts = verts[np.array(verts_idx[:vert_count_poly], dtype=np.int32)]
            detail = details[poly_idx] if poly_idx < len(details) else None

            if detail is None or detail[3] == 0:
                self._append_poly_fan(base_verts, all_min, all_max)
                continue

            vert_base, tri_base, _, tri_count_detail = detail
            for tri_local_idx in range(tri_count_detail):
                tri_idx = tri_base + tri_local_idx
                tri = detail_tris[tri_idx]
                tri_verts = []
                for corner in tri[:3]:
                    idx = int(corner)
                    if idx < vert_count_poly:
                        nav_v = base_verts[idx]
                    else:
                        nav_v = detail_verts[vert_base + idx - vert_count_poly]
                    tri_verts.append(_nav_to_engine(nav_v))
                tri_verts3d = np.stack(tri_verts, axis=0)
                self._append_triangle(tri_verts3d, all_min, all_max)

    def _append_poly_fan(self, base_verts: np.ndarray, all_min: np.ndarray, all_max: np.ndarray) -> None:
        if base_verts.shape[0] < 3:
            return
        anchor = _nav_to_engine(base_verts[0])
        for idx in range(1, base_verts.shape[0] - 1):
            tri_verts3d = np.stack(
                [anchor, _nav_to_engine(base_verts[idx]), _nav_to_engine(base_verts[idx + 1])],
                axis=0,
            )
            self._append_triangle(tri_verts3d, all_min, all_max)

    def _append_triangle(self, tri_verts3d: np.ndarray, all_min: np.ndarray, all_max: np.ndarray) -> None:
        tri_verts2d = tri_verts3d[:, [0, 2]].astype(np.float32)
        edge0 = tri_verts2d[1] - tri_verts2d[0]
        edge1 = tri_verts2d[2] - tri_verts2d[0]
        area2 = abs(edge0[0] * edge1[1] - edge0[1] * edge1[0])
        if area2 < 1e-4:
            return

        bbox_min = np.min(tri_verts2d, axis=0)
        bbox_max = np.max(tri_verts2d, axis=0)
        all_min[:] = np.minimum(all_min, bbox_min)
        all_max[:] = np.maximum(all_max, bbox_max)
        self.triangles.append(
            Triangle(
                verts3d=tri_verts3d.astype(np.float32),
                verts2d=tri_verts2d,
                bbox_min=bbox_min.astype(np.float32),
                bbox_max=bbox_max.astype(np.float32),
                avg_height=float(np.mean(tri_verts3d[:, 1])),
            )
        )


class MajestroNavMeshEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        navmesh_path: str | Path | None = None,
        move_step_size: float = 120.0,
        tactical_target_radius: float = 600.0,
        num_other_agents: int = 4,
        observed_other_agents: int = 3,
        agent_radius: float = 90.0,
        success_radius: float = 120.0,
        goal_spawn_min_scale: float = 0.001,
        agent_spawn_min_scale: float = 0.00001,
        agent_spawn_max_scale: float = 0.0001,
        max_steps: int = 512,
        seed: int = 1,
        grid_resolution: int = 256,
        ray_count: int = 8,
        ray_length: float = 600.0,
        sense_radius: float | None = None,
        collision_penalty: float = 0.35,
        stall_penalty: float = 0.05,
        stall_patience: int = 20,
        time_penalty: float = 0.01,
        success_reward: float = 50.0,
        role_rule: str = "fixed",
        agent_role_rules: Optional[Sequence[str]] = None,
        role_selector: Optional[Callable[["MajestroNavMeshEnv"], Sequence[int]]] = None,
        dynamic_horizon: bool = True,
        dynamic_horizon_kappa: float = 1.8,
        dynamic_horizon_Tmin: int = 96,
        dynamic_horizon_Tmax: int = 1024,
        dynamic_horizon_use_geodesic: bool = True,
    ):
        super().__init__()
        if navmesh_path is None:
            navmesh_path = Path(__file__).resolve().parent / "Resources" / "NavMesh" / "all_tiles_navmesh.bin"
        self.navmesh_path = str(navmesh_path)
        self._nav = MajestroNavMeshData(self.navmesh_path)
        self.triangles = self._nav.triangles

        self.step_size = float(move_step_size)
        self.tactical_target_radius = float(tactical_target_radius)
        self.num_other_agents = int(max(0, num_other_agents))
        self.observed_other_agents = int(max(0, observed_other_agents))
        self.agent_radius = float(agent_radius)
        self.success_radius = float(success_radius)
        self.goal_spawn_min_scale = float(goal_spawn_min_scale)
        self.agent_spawn_min_scale = float(agent_spawn_min_scale)
        self.agent_spawn_max_scale = float(max(agent_spawn_max_scale, self.agent_spawn_min_scale))
        self.max_steps = int(max_steps)
        self.base_max_steps = int(max_steps)
        self.grid_resolution = int(grid_resolution)
        self.ray_count = int(ray_count)
        self.ray_length = float(ray_length)
        self.sense_radius = float(ray_length if sense_radius is None else sense_radius)
        self.collision_penalty = float(collision_penalty)
        self.stall_penalty = float(stall_penalty)
        self.stall_patience = int(stall_patience)
        self.time_penalty = float(time_penalty)
        self._R_SUCCESS = float(success_reward)
        self._R_SUCCESS_ENTRY = 20.0
        self._R_SUCCESS_SUSTAIN = 2.0
        self._R_SUCCESS_DROP = 3.0
        self.role_rule = str(role_rule).strip().lower()
        self.agent_role_rules = None if agent_role_rules is None else [str(x).strip().lower() for x in agent_role_rules]
        self.role_selector = role_selector

        self.dynamic_horizon = bool(dynamic_horizon)
        self.dynamic_horizon_kappa = float(dynamic_horizon_kappa)
        self.dynamic_horizon_Tmin = int(dynamic_horizon_Tmin)
        self.dynamic_horizon_Tmax = int(dynamic_horizon_Tmax)
        self.dynamic_horizon_use_geodesic = bool(dynamic_horizon_use_geodesic)

        self._seed_value = seed
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        self.bounds_min = self._nav.bounds_min.copy()
        self.bounds_max = self._nav.bounds_max.copy()
        self.map_center = 0.5 * (self.bounds_min + self.bounds_max)
        self.map_size = np.maximum(self.bounds_max - self.bounds_min, 1.0)
        self.map_range = float(np.max(self.map_size) * 0.5)

        self._grid_origin = None
        self._grid_cell_size = None
        self._walkable = None
        self._height_map = None
        self._geo_map = None
        self._geo_origin = None
        self._geo_cell_size = None
        self._geo_goal_rc = None
        self._grid_tri_indices: List[List[int]] = []
        self._free_cells = None
        self._detour_wrapper = None
        self._detour_enabled = False
        self._detour_last_error = ""

        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.agent_height = 0.0
        self.goal_height = 0.0
        self.num_agents = 1 + self.num_other_agents
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_heights = np.zeros((self.num_agents,), dtype=np.float32)
        self.agent_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_target_offsets = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_role_ids = np.zeros((self.num_agents,), dtype=np.int32)
        self.role_targets = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._stall_best = None
        self._stall_wait = None
        self._prev_geo = None
        self._prev_success_mask = np.zeros((self.num_agents,), dtype=bool)
        self.steps = 0

        self._build_raster_cache()
        self._init_detour_wrapper()

        obs_dim = 3 + 3 + 3 + 2 + self.observed_other_agents * 4 + 1
        self.single_agent_obs_dim = int(obs_dim)
        self.single_agent_act_dim = 2
        self._apply_agent_config(self.num_agents, self.agent_role_rules)

    def _normalize_agent_role_rules(self, rules: Optional[Sequence[str]], num_agents: int) -> Optional[List[str]]:
        if rules is None:
            return None
        out = [str(x).strip().lower() for x in rules if str(x).strip()]
        if not out:
            return None
        if len(out) == 1:
            out = out * int(num_agents)
        elif len(out) != int(num_agents):
            raise ValueError(
                f"agent_role_rules length must be 1 or num_agents ({int(num_agents)}), got {len(out)}"
            )
        return out

    def _apply_agent_config(self, num_agents: int, agent_role_rules: Optional[Sequence[str]]) -> None:
        self.num_agents = int(max(1, num_agents))
        self.num_other_agents = int(max(0, self.num_agents - 1))
        self.agent_role_rules = self._normalize_agent_role_rules(agent_role_rules, self.num_agents)
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_heights = np.zeros((self.num_agents,), dtype=np.float32)
        self.agent_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_target_offsets = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_role_ids = np.zeros((self.num_agents,), dtype=np.int32)
        self.role_targets = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._prev_success_mask = np.zeros((self.num_agents,), dtype=bool)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self.single_agent_obs_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, self.single_agent_act_dim),
            dtype=np.float32,
        )

    def configure_agent_group(self, agent_role_rules: Optional[Sequence[str]]) -> None:
        rules = None if agent_role_rules is None else [str(x).strip().lower() for x in agent_role_rules if str(x).strip()]
        num_agents = 1 if not rules else len(rules)
        self._apply_agent_config(num_agents, rules)

    def _init_detour_wrapper(self) -> None:
        self._detour_wrapper = None
        self._detour_enabled = False
        self._detour_last_error = ""

        module_name = "detour_navmesh_py"
        module_dir_env = os.environ.get("DETOUR_MODULE_DIR", "").strip()
        search_dirs = []
        if module_dir_env:
            search_dirs.append(module_dir_env)
        search_dirs.extend(
            [
                str(Path(__file__).resolve().parent / "native" / "runtime"),
                str(Path(__file__).resolve().parent / "native"),
                "/tmp/sac_detour_build",
            ]
        )

        added_paths = []
        try:
            for module_dir in search_dirs:
                if not module_dir:
                    continue
                if not Path(module_dir).exists():
                    continue
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                    added_paths.append(module_dir)

            module = importlib.import_module(module_name)
            wrapper = module.DetourNavMeshWrapper()
            if not wrapper.load_navmesh(self.navmesh_path):
                self._detour_last_error = str(wrapper.last_error())
                return
            self._detour_wrapper = wrapper
            self._detour_enabled = True
        except Exception as exc:
            self._detour_last_error = str(exc)
        finally:
            for added in added_paths:
                if added in sys.path:
                    sys.path.remove(added)

    def _world3_to_detour_xyz(self, world3: np.ndarray) -> Tuple[float, float, float]:
        return (
            float(world3[2]) / 100.0,
            float(world3[1]) / 100.0,
            float(world3[0]) / 100.0,
        )

    def _detour_xyz_to_world3(self, x: float, y: float, z: float) -> np.ndarray:
        return np.array([float(z) * 100.0, float(y) * 100.0, float(x) * 100.0], dtype=np.float32)

    def _build_raster_cache(self) -> None:
        span = self.bounds_max - self.bounds_min
        max_span = float(np.max(span))
        cell = max(max_span / float(self.grid_resolution), 1.0)
        cols = max(8, int(math.ceil(span[0] / cell)))
        rows = max(8, int(math.ceil(span[1] / cell)))

        self._grid_cell_size = float(cell)
        self._grid_origin = self.bounds_min.copy()
        self._geo_origin = self._grid_origin.copy()
        self._geo_cell_size = np.array([cell, cell], dtype=np.float32)
        self._grid_rows = rows
        self._grid_cols = cols
        self._walkable = np.zeros((rows, cols), dtype=np.uint8)
        self._height_map = np.full((rows, cols), -np.inf, dtype=np.float32)
        self._grid_tri_indices = [[] for _ in range(rows * cols)]

        for tri_idx, tri in enumerate(self.triangles):
            c0 = max(0, int(math.floor((tri.bbox_min[0] - self._grid_origin[0]) / cell)))
            c1 = min(cols - 1, int(math.floor((tri.bbox_max[0] - self._grid_origin[0]) / cell)))
            r0 = max(0, int(math.floor((tri.bbox_min[1] - self._grid_origin[1]) / cell)))
            r1 = min(rows - 1, int(math.floor((tri.bbox_max[1] - self._grid_origin[1]) / cell)))

            for r in range(r0, r1 + 1):
                cz = self._grid_origin[1] + (r + 0.5) * cell
                for c in range(c0, c1 + 1):
                    cx = self._grid_origin[0] + (c + 0.5) * cell
                    p = np.array([cx, cz], dtype=np.float32)
                    if not _point_in_triangle_2d(p, tri.verts2d):
                        continue
                    bucket = self._grid_tri_indices[r * cols + c]
                    bucket.append(tri_idx)
                    self._walkable[r, c] = 1
                    h = _barycentric_height(p, tri.verts2d, tri.verts3d)
                    if h is not None and h > self._height_map[r, c]:
                        self._height_map[r, c] = h

        self._free_cells = np.argwhere(self._walkable > 0)
        if self._free_cells.size == 0:
            raise ValueError("No walkable raster cells created from navmesh.")

    def _world_to_grid_rc(self, pos: np.ndarray) -> Tuple[int, int]:
        c = int(math.floor((float(pos[0]) - float(self._grid_origin[0])) / self._grid_cell_size))
        r = int(math.floor((float(pos[1]) - float(self._grid_origin[1])) / self._grid_cell_size))
        r = max(0, min(self._grid_rows - 1, r))
        c = max(0, min(self._grid_cols - 1, c))
        return r, c

    def _grid_rc_to_world(self, r: int, c: int) -> np.ndarray:
        return np.array(
            [
                self._grid_origin[0] + (c + 0.5) * self._grid_cell_size,
                self._grid_origin[1] + (r + 0.5) * self._grid_cell_size,
            ],
            dtype=np.float32,
        )

    def _nearest_valid_point(self, pos: np.ndarray, max_radius: int = 4) -> Optional[Tuple[np.ndarray, float]]:
        r, c = self._world_to_grid_rc(pos)
        best = None
        best_d2 = None
        for rad in range(max_radius + 1):
            r0 = max(0, r - rad)
            r1 = min(self._grid_rows - 1, r + rad)
            c0 = max(0, c - rad)
            c1 = min(self._grid_cols - 1, c + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if self._walkable[rr, cc] == 0:
                        continue
                    world = self._grid_rc_to_world(rr, cc)
                    d2 = float(np.sum((world - pos) ** 2))
                    if best_d2 is None or d2 < best_d2:
                        best = world
                        best_d2 = d2
            if best is not None:
                break
        if best is None:
            return None
        height = self._sample_height(best)
        if height is None:
            height = 0.0
        return best, float(height)

    def _sample_height(self, pos: np.ndarray) -> Optional[float]:
        r, c = self._world_to_grid_rc(pos)
        cells = [(r, c)]
        for rr in range(max(0, r - 1), min(self._grid_rows - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(self._grid_cols - 1, c + 1) + 1):
                if (rr, cc) != (r, c):
                    cells.append((rr, cc))

        best_h = None
        best_abs = None
        for rr, cc in cells:
            for tri_idx in self._grid_tri_indices[rr * self._grid_cols + cc]:
                tri = self.triangles[tri_idx]
                if not _point_in_triangle_2d(pos, tri.verts2d):
                    continue
                h = _barycentric_height(pos, tri.verts2d, tri.verts3d)
                if h is None:
                    continue
                abs_h = abs(h)
                if best_abs is None or abs_h < best_abs:
                    best_h = h
                    best_abs = abs_h
        if best_h is not None:
            return float(best_h)

        if self._walkable[r, c]:
            h = float(self._height_map[r, c])
            if np.isfinite(h):
                return h
        return None

    def _is_walkable_point(self, pos: np.ndarray) -> Tuple[bool, Optional[float]]:
        h = self._sample_height(pos)
        return h is not None, h

    def _move_along_navmesh(self, start: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        delta = target - start
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6:
            return start.copy(), float(self.agent_height), False

        sample_step = max(self._grid_cell_size * 0.5, 10.0)
        steps = max(1, int(math.ceil(dist / sample_step)))

        last_valid = start.copy()
        last_height = float(self.agent_height)
        collided = False

        for i in range(1, steps + 1):
            t = float(i) / float(steps)
            probe = start + delta * t
            valid, height = self._is_walkable_point(probe)
            if valid:
                last_valid = probe
                last_height = float(height)
                continue
            collided = True
            lo = last_valid.copy()
            hi = probe.copy()
            for _ in range(6):
                mid = (lo + hi) * 0.5
                mid_valid, mid_height = self._is_walkable_point(mid)
                if mid_valid:
                    last_valid = mid
                    last_height = float(mid_height)
                    lo = mid
                else:
                    hi = mid
            break

        return last_valid.astype(np.float32), float(last_height), collided

    def _collides_with_other_agents(self, pos: np.ndarray, ignore_index: Optional[int] = None) -> bool:
        if self.agent_positions.size == 0:
            return False
        min_dist = self.agent_radius * 2.0
        min_dist_sq = min_dist * min_dist
        for idx, other in enumerate(self.agent_positions):
            if ignore_index is not None and idx == ignore_index:
                continue
            dx = float(pos[0] - other[0])
            dz = float(pos[1] - other[1])
            if dx * dx + dz * dz < min_dist_sq:
                return True
        return False

    def _separation_penalty_value(self, pos: np.ndarray) -> float:
        if self.agent_positions.size == 0:
            return 0.0
        penalty = 0.0
        min_dist = self.agent_radius * 2.0
        for other in self.agent_positions:
            dist = float(np.linalg.norm(pos - other))
            if dist < min_dist:
                penalty += (min_dist - dist) / max(min_dist, 1.0)
        return penalty

    def _move_with_agent_avoidance(
        self,
        start: np.ndarray,
        target: np.ndarray,
        ignore_index: Optional[int] = None,
        start_height: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        saved_height = self.agent_height
        if start_height is not None:
            self.agent_height = float(start_height)

        moved, moved_height, nav_collided = self._move_along_navmesh(start, target)
        agent_blocked = False
        if self._collides_with_other_agents(moved, ignore_index=ignore_index):
            agent_blocked = True
            direction = moved - start
            dist = float(np.linalg.norm(direction))
            if dist > 1e-6:
                lo = start.copy()
                hi = moved.copy()
                best = start.copy()
                best_height = float(start_height if start_height is not None else moved_height)
                for _ in range(7):
                    mid = (lo + hi) * 0.5
                    mid_h = self._sample_height(mid)
                    if mid_h is None or self._collides_with_other_agents(mid, ignore_index=ignore_index):
                        hi = mid
                    else:
                        best = mid
                        best_height = float(mid_h)
                        lo = mid
                moved = best.astype(np.float32)
                moved_height = best_height
            else:
                moved = start.copy()
                moved_height = float(start_height if start_height is not None else saved_height)

        self.agent_height = saved_height
        return moved, float(moved_height), bool(nav_collided or agent_blocked)

    def _compute_geodesic_map(self, goal_rc: Tuple[int, int]) -> np.ndarray:
        rows, cols = self._walkable.shape
        dist = np.full((rows, cols), np.inf, dtype=np.float32)
        gr, gc = goal_rc
        if self._walkable[gr, gc] == 0:
            return dist

        dist[gr, gc] = 0.0
        pq = [(0.0, gr, gc)]
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != float(dist[r, c]):
                continue
            for dr, dc, w in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or self._walkable[nr, nc] == 0:
                    continue
                nd = d + w
                if nd < float(dist[nr, nc]):
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
        return dist

    def _geo_distance(self, pos: np.ndarray, search: int = 3) -> Optional[float]:
        if self._geo_map is None:
            return None
        r, c = self._world_to_grid_rc(pos)
        best = None
        for rad in range(search + 1):
            r0 = max(0, r - rad)
            r1 = min(self._grid_rows - 1, r + rad)
            c0 = max(0, c - rad)
            c1 = min(self._grid_cols - 1, c + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    d = float(self._geo_map[rr, cc])
                    if not math.isfinite(d):
                        continue
                    if best is None or d < best:
                        best = d
            if best is not None:
                break
        if best is None:
            return None
        return float(best * self._grid_cell_size)

    def _geo_distance_robust(self, pos: np.ndarray, max_search: int = 3) -> Optional[float]:
        return self._geo_distance(pos, search=max_search)

    def _pos_to_geo_rc(self, pos: np.ndarray) -> Tuple[int, int]:
        return self._world_to_grid_rc(pos)

    def _compute_dynamic_horizon(self) -> int:
        if self.dynamic_horizon_use_geodesic:
            d_world = self._geo_distance(self.agent_pos)
        else:
            d_world = None
        if d_world is None:
            d_world = float(np.linalg.norm(self.goal_pos - self.agent_pos))
        t_min = int(math.ceil(d_world / max(self.step_size, 1e-6)))
        return int(np.clip(math.ceil(self.dynamic_horizon_kappa * t_min), self.dynamic_horizon_Tmin, self.dynamic_horizon_Tmax))

    def _other_agent_observation(self, agent_index: int, scale: float, max_distance: Optional[float] = None) -> np.ndarray:
        if self.observed_other_agents <= 0:
            return np.zeros((0,), dtype=np.float32), 0
        if self.agent_positions.shape[0] <= 1:
            return np.zeros((self.observed_other_agents * 4,), dtype=np.float32), 0
        other_indices = np.array([idx for idx in range(self.agent_positions.shape[0]) if idx != agent_index], dtype=np.int32)
        others = self.agent_positions[other_indices]
        rel = others - self.agent_positions[agent_index][None, :]
        dists = np.linalg.norm(rel, axis=1)
        if max_distance is not None:
            mask = dists <= float(max_distance)
            other_indices = other_indices[mask]
            rel = rel[mask]
            dists = dists[mask]
        if dists.size == 0:
            return np.zeros((self.observed_other_agents * 4,), dtype=np.float32), 0
        order = np.argsort(dists)[: self.observed_other_agents]
        obs = np.zeros((self.observed_other_agents, 4), dtype=np.float32)
        for out_idx, agent_idx in enumerate(order):
            obs[out_idx, 0:2] = rel[agent_idx] / scale
            obs[out_idx, 2] = dists[agent_idx] / scale
            obs[out_idx, 3] = self._heuristic_value(int(other_indices[agent_idx]))[0]
        return obs.reshape(-1), int(len(order))

    def _sense_local_space(self, agent_index: int, scale: float) -> Tuple[np.ndarray, float]:
        sense_limit = max(1.0, float(self.sense_radius))
        other_obs, sensed_agents = self._other_agent_observation(agent_index, scale, max_distance=sense_limit)
        goal_dist = float(np.linalg.norm(self.goal_pos - self.agent_positions[agent_index]))
        goal_in_sense = bool(goal_dist <= sense_limit)
        fail_code = 1.0 if (sensed_agents == 0 and not goal_in_sense) else 0.0
        return other_obs, fail_code

    def _geo_next_waypoint(self, pos: np.ndarray, max_search: int = 3) -> Optional[np.ndarray]:
        if self._geo_map is None:
            return None
        rc = self._pos_to_geo_rc(pos)
        rows, cols = self._geo_map.shape

        def find_valid_start(r: int, c: int, radius: int = 3) -> Optional[Tuple[int, int]]:
            if 0 <= r < rows and 0 <= c < cols and np.isfinite(self._geo_map[r, c]):
                return r, c
            for rad in range(1, radius + 1):
                r0 = max(0, r - rad)
                r1 = min(rows - 1, r + rad)
                c0 = max(0, c - rad)
                c1 = min(cols - 1, c + rad)
                best = None
                best_val = np.inf
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        v = float(self._geo_map[rr, cc])
                        if np.isfinite(v) and v < best_val:
                            best = (rr, cc)
                            best_val = v
                if best is not None:
                    return best
            return None

        start = find_valid_start(rc[0], rc[1], radius=max_search)
        if start is None:
            return None

        cur_r, cur_c = start
        cur_val = float(self._geo_map[cur_r, cur_c])
        best = None
        best_val = cur_val
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in neighbors:
            rr = cur_r + dr
            cc = cur_c + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            v = float(self._geo_map[rr, cc])
            if np.isfinite(v) and v + 1e-6 < best_val:
                best = (rr, cc)
                best_val = v

        if best is None:
            return self.goal_pos.copy()
        return self._grid_rc_to_world(best[0], best[1]).astype(np.float32)

    def _detour_next_waypoint_to_target(
        self,
        pos: np.ndarray,
        target_pos: np.ndarray,
        height: Optional[float] = None,
        target_height: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if not self._detour_enabled or self._detour_wrapper is None:
            return None

        cur_h = float(self.agent_height if height is None else height)
        start3 = np.array([float(pos[0]), cur_h, float(pos[1])], dtype=np.float32)
        if target_height is None:
            sampled_h = self._sample_height(np.asarray(target_pos, dtype=np.float32).reshape(-1)[:2])
            goal_h = float(self.goal_height if sampled_h is None else sampled_h)
        else:
            goal_h = float(target_height)
        goal3 = np.array([float(target_pos[0]), goal_h, float(target_pos[1])], dtype=np.float32)
        sx, sy, sz = self._world3_to_detour_xyz(start3)
        gx, gy, gz = self._world3_to_detour_xyz(goal3)

        try:
            waypoint = self._detour_wrapper.find_next_waypoint(sx, sy, sz, gx, gy, gz)
        except Exception as exc:
            self._detour_last_error = str(exc)
            return None
        if waypoint is None:
            return None

        wx, wy, wz = waypoint
        world3 = self._detour_xyz_to_world3(wx, wy, wz)
        return world3[[0, 2]].astype(np.float32)

    def _detour_next_waypoint(self, pos: np.ndarray, height: Optional[float] = None) -> Optional[np.ndarray]:
        return self._detour_next_waypoint_to_target(
            pos,
            self.goal_pos,
            height=height,
            target_height=float(self.goal_height),
        )

    def recover_fallback_path_world(
        self,
        start_pos: np.ndarray,
        start_height: Optional[float] = None,
        max_len: int = 256,
    ) -> List[np.ndarray]:
        pos = np.asarray(start_pos, dtype=np.float32).reshape(-1)[:2].copy()
        cur_height = float(self.agent_height if start_height is None else start_height)

        if self._detour_enabled:
            pts: List[np.ndarray] = [pos.copy()]
            seen = set()
            step_eps = max(self._grid_cell_size * 0.25, 1.0)
            for _ in range(max_len):
                key = (round(float(pos[0]), 2), round(float(pos[1]), 2))
                if key in seen:
                    break
                seen.add(key)

                waypoint = self._detour_next_waypoint(pos, height=cur_height)
                if waypoint is None:
                    break
                if float(np.linalg.norm(waypoint - pos)) <= step_eps:
                    break
                pts.append(waypoint.copy())
                pos = waypoint
                h = self._sample_height(pos)
                if h is not None:
                    cur_height = float(h)
                if float(np.linalg.norm(self.goal_pos - pos)) <= max(self.step_size, self._grid_cell_size):
                    pts.append(self.goal_pos.copy())
                    break
            return pts

        if self._geo_map is None:
            return []

        rc = self._pos_to_geo_rc(pos)
        rows, cols = self._geo_map.shape

        def find_valid_start(r: int, c: int, radius: int = 3) -> Optional[Tuple[int, int]]:
            if 0 <= r < rows and 0 <= c < cols and np.isfinite(self._geo_map[r, c]):
                return r, c
            for rad in range(1, radius + 1):
                r0 = max(0, r - rad)
                r1 = min(rows - 1, r + rad)
                c0 = max(0, c - rad)
                c1 = min(cols - 1, c + rad)
                best = None
                best_val = np.inf
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        v = float(self._geo_map[rr, cc])
                        if np.isfinite(v) and v < best_val:
                            best = (rr, cc)
                            best_val = v
                if best is not None:
                    return best
            return None

        cur = find_valid_start(rc[0], rc[1], radius=3)
        if cur is None:
            return []

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        pts = [pos.copy()]
        cur_val = float(self._geo_map[cur[0], cur[1]])

        for _ in range(max_len):
            if self._geo_goal_rc is not None and cur == self._geo_goal_rc:
                break
            best = None
            best_val = cur_val
            for dr, dc in neighbors:
                rr = cur[0] + dr
                cc = cur[1] + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                v = float(self._geo_map[rr, cc])
                if np.isfinite(v) and v + 1e-6 < best_val:
                    best = (rr, cc)
                    best_val = v
            if best is None:
                break
            cur = best
            cur_val = best_val
            pts.append(self._grid_rc_to_world(cur[0], cur[1]))
        return pts

    def _role_value(self, agent_index: int) -> np.ndarray:
        denom = max(1, ROLE_COUNT - 1)
        role_id = int(np.clip(self.agent_role_ids[agent_index], 0, ROLE_COUNT - 1))
        return np.array([float(role_id) / float(denom)], dtype=np.float32)

    def _heuristic_value(self, agent_index: int) -> np.ndarray:
        rule_name = self._heuristic_name_for_agent(agent_index)
        heuristic_id = int(HEURISTIC_NAME_TO_ID.get(rule_name, 0))
        denom = max(1, HEURISTIC_COUNT - 1)
        return np.array([float(heuristic_id) / float(denom)], dtype=np.float32)

    def _heuristic_name_for_agent(self, agent_index: int) -> str:
        if self.agent_role_rules is not None:
            return str(self.agent_role_rules[agent_index]).strip().lower()
        return str(self.role_rule).strip().lower()

    def _has_nearby_actor_with_heuristic(self, agent_index: int, names: Sequence[str]) -> bool:
        wanted = {str(name).strip().lower() for name in names}
        sense_limit_sq = max(1.0, float(self.sense_radius)) ** 2
        my_pos = self.agent_positions[agent_index]
        for idx, other_pos in enumerate(self.agent_positions):
            if idx == agent_index:
                continue
            rule_name = self._heuristic_name_for_agent(idx)
            if rule_name not in wanted:
                continue
            rel = other_pos - my_pos
            if float(np.dot(rel, rel)) <= sense_limit_sq:
                return True
        return False

    def _normalize_vec(self, v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= eps:
            return np.zeros_like(v, dtype=np.float32)
        return (v / n).astype(np.float32)

    def _closest_cover_anchor(self, agent_index: int) -> Optional[np.ndarray]:
        if self.num_agents <= 1:
            return None
        my_pos = self.agent_positions[agent_index]
        best = None
        best_score = np.inf
        to_goal = self.goal_pos - my_pos
        to_goal_dir = self._normalize_vec(to_goal)
        for idx, other in enumerate(self.agent_positions):
            if idx == agent_index:
                continue
            rel = other - my_pos
            dist = float(np.linalg.norm(rel))
            if dist <= 1e-6:
                continue
            rel_dir = self._normalize_vec(rel)
            align = float(np.dot(rel_dir, to_goal_dir))
            score = dist - 80.0 * align
            if score < best_score:
                best_score = score
                best = other.copy()
        return best

    def _surround_stats(self, agent_index: int, pos: Optional[np.ndarray] = None) -> Tuple[float, float, int]:
        center = self.goal_pos
        my_pos = self.agent_positions[agent_index] if pos is None else np.asarray(pos, dtype=np.float32)
        my_rel = my_pos - center
        my_dist = float(np.linalg.norm(my_rel))
        if my_dist <= 1e-6:
            my_angle = 0.0
        else:
            my_angle = math.atan2(float(my_rel[1]), float(my_rel[0]))

        ally_angles: List[float] = []
        for idx, other in enumerate(self.agent_positions):
            if idx == agent_index:
                continue
            rel = other - center
            dist = float(np.linalg.norm(rel))
            if dist <= self.success_radius * 0.8:
                continue
            ally_angles.append(math.atan2(float(rel[1]), float(rel[0])))

        if not ally_angles:
            return my_dist, math.pi, 0

        min_gap = min(
            abs(((my_angle - ang + math.pi) % (2.0 * math.pi)) - math.pi)
            for ang in ally_angles
        )
        return my_dist, float(min_gap), len(ally_angles)

    def _role_target_for(self, agent_index: int) -> np.ndarray:
        role_id = int(self.agent_role_ids[agent_index])
        my_pos = self.agent_positions[agent_index]
        to_goal = self.goal_pos - my_pos
        goal_dir = self._normalize_vec(to_goal)
        if float(np.linalg.norm(goal_dir)) <= 1e-6:
            goal_dir = np.array([1.0, 0.0], dtype=np.float32)
        side_dir = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)

        if role_id == ROLE_FRONT:
            target = self.goal_pos.copy()
        elif role_id == ROLE_COVER:
            cover_anchor = self._closest_cover_anchor(agent_index)
            if cover_anchor is None:
                target = self.goal_pos - goal_dir * max(self.agent_radius * 3.0, self.success_radius * 1.2)
            else:
                cover_dir = self._normalize_vec(cover_anchor - self.goal_pos)
                if float(np.linalg.norm(cover_dir)) <= 1e-6:
                    cover_dir = -goal_dir
                target = cover_anchor + cover_dir * max(self.agent_radius * 2.2, self.success_radius * 0.9)
        elif role_id == ROLE_BASE_MOVE:
            target = self.goal_pos.copy()
        elif role_id == ROLE_SURROUND:
            surround_radius = max(self.success_radius * 1.6, self.agent_radius * 2.2)
            rel = my_pos - self.goal_pos
            rel_dir = self._normalize_vec(rel)
            if float(np.linalg.norm(rel_dir)) <= 1e-6:
                rel_dir = side_dir
            target = self.goal_pos + rel_dir * surround_radius
        else:
            kiting_radius = max(1.0, float(self.sense_radius) - 50.0)
            rel = my_pos - self.goal_pos
            rel_dir = self._normalize_vec(rel)
            if float(np.linalg.norm(rel_dir)) <= 1e-6:
                rel_dir = -goal_dir
            target = self.goal_pos + rel_dir * kiting_radius

        snapped = self._nearest_valid_point(target, max_radius=8)
        if snapped is not None:
            return snapped[0].astype(np.float32)
        return target.astype(np.float32)

    def _fixed_role_for_agent(self, agent_index: int) -> int:
        base_roles = [
            ROLE_FRONT,
            ROLE_COVER,
            ROLE_BASE_MOVE,
            ROLE_SURROUND,
            ROLE_KITING,
        ]
        if agent_index == 0:
            return ROLE_FRONT
        return int(base_roles[(agent_index - 1) % len(base_roles)])

    def _agent_rule_stats(self, agent_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float]:
        positions = self.agent_positions.copy()
        dists = np.linalg.norm(self.goal_pos[None, :] - positions, axis=1)
        my_pos = positions[agent_index]
        team_center = np.mean(positions, axis=0) if len(positions) > 0 else my_pos
        approach_dir = self._normalize_vec(self.goal_pos - team_center)
        if float(np.linalg.norm(approach_dir)) <= 1e-6:
            approach_dir = self._normalize_vec(self.goal_pos - my_pos)
        if float(np.linalg.norm(approach_dir)) <= 1e-6:
            approach_dir = np.array([1.0, 0.0], dtype=np.float32)
        side_dir = np.array([-approach_dir[1], approach_dir[0]], dtype=np.float32)
        side_score = float(np.dot(my_pos - self.goal_pos, side_dir))
        my_dist = float(dists[agent_index])
        median_dist = float(np.median(dists)) if len(dists) > 0 else my_dist
        p30 = float(np.percentile(dists, 30)) if len(dists) > 0 else my_dist
        p70 = float(np.percentile(dists, 70)) if len(dists) > 0 else my_dist
        return positions, dists, side_dir, side_score, my_dist, median_dist, p30, p70

    def _choose_role_for_agent(self, agent_index: int, rule_name: str) -> int:
        rule = str(rule_name).strip().lower()
        if rule == "fixed":
            return self._fixed_role_for_agent(agent_index)
        if rule in ("melee_dps", "meleedps"):
            my_dist = float(np.linalg.norm(self.goal_pos - self.agent_positions[agent_index]))
            goal_in_sense = bool(my_dist <= max(1.0, float(self.sense_radius)))
            has_nearby_melee = self._has_nearby_actor_with_heuristic(agent_index, ("melee_dps", "meleedps"))
            if not goal_in_sense:
                return ROLE_BASE_MOVE
            if has_nearby_melee:
                return ROLE_SURROUND
            return ROLE_FRONT
        if rule in ("ranged_dps", "rangeddps"):
            my_dist = float(np.linalg.norm(self.goal_pos - self.agent_positions[agent_index]))
            goal_in_sense = bool(my_dist <= max(1.0, float(self.sense_radius)))
            if not goal_in_sense:
                return ROLE_BASE_MOVE
            if self._has_nearby_actor_with_heuristic(agent_index, ("melee_dps", "meleedps")):
                return ROLE_COVER
            return ROLE_KITING

        return self._fixed_role_for_agent(agent_index)

    def _compute_assigned_roles(self) -> np.ndarray:
        if callable(self.role_selector):
            try:
                selected = np.asarray(self.role_selector(self), dtype=np.int32).reshape(-1)
                if selected.shape[0] == self.num_agents:
                    return np.clip(selected, 0, ROLE_COUNT - 1).astype(np.int32)
            except Exception:
                pass

        out = np.zeros((self.num_agents,), dtype=np.int32)
        for idx in range(self.num_agents):
            if self.agent_role_rules is not None:
                rule = self.agent_role_rules[idx]
            else:
                rule = self.role_rule
            out[idx] = int(self._choose_role_for_agent(idx, rule))
        return out

    def _assign_roles(self) -> None:
        self.agent_role_ids[:] = self._compute_assigned_roles()

    def _update_role_targets(self) -> None:
        for idx in range(self.num_agents):
            self.role_targets[idx] = self._role_target_for(idx)

    def _role_progress_delta(self, old_pos: np.ndarray, new_pos: np.ndarray, target_pos: np.ndarray) -> float:
        old_d = float(np.linalg.norm(target_pos - old_pos))
        new_d = float(np.linalg.norm(target_pos - new_pos))
        return old_d - new_d

    def _role_reward_bonus(
        self,
        agent_index: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
        collided: bool,
        detour_used: bool,
        old_path_dist: Optional[float] = None,
        new_path_dist: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float], bool]:
        role_id = int(self.agent_role_ids[agent_index])
        target = self.role_targets[agent_index]
        terms: Dict[str, float] = {}
        bonus = 0.0
        tactical_success = False

        role_progress = self._role_progress_delta(old_pos, new_pos, target)
        role_progress_reward = 0.03 * role_progress
        bonus += role_progress_reward
        terms["role_progress"] = role_progress_reward

        to_goal = self.goal_pos - new_pos
        goal_dist = float(np.linalg.norm(to_goal))
        goal_dir = self._normalize_vec(to_goal)
        side_dir = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32)

        if role_id == ROLE_FRONT:
            directness = float(np.dot(self._normalize_vec(self.agent_velocities[agent_index]), goal_dir))
            direct_bonus = 0.12 * max(0.0, directness)
            bonus += direct_bonus
            terms["role_front"] = direct_bonus
            tactical_success = bool(goal_dist <= self.success_radius * 1.15 and directness >= 0.45)
        elif role_id == ROLE_COVER:
            anchor = self._closest_cover_anchor(agent_index)
            cover_bonus = 0.0
            cover_goal_cap = max(1.0, float(self.sense_radius))
            if anchor is not None:
                anchor_to_goal = self.goal_pos - anchor
                me_to_anchor = new_pos - anchor
                behind_score = float(np.dot(self._normalize_vec(me_to_anchor), -self._normalize_vec(anchor_to_goal)))
                distance_gate = max(0.0, 1.0 - goal_dist / max(cover_goal_cap, 1.0))
                cover_bonus += 0.10 * max(0.0, behind_score) * distance_gate
                line_gap = float(np.linalg.norm((new_pos - self.goal_pos) - anchor_to_goal))
                cover_bonus -= 0.02 * min(line_gap / max(self.agent_radius * 4.0, 1.0), 1.5)
                anchor_dist = float(np.linalg.norm(me_to_anchor))
                if goal_dist > cover_goal_cap:
                    cover_bonus -= 0.08 * min((goal_dist - cover_goal_cap) / max(self.success_radius, 1.0), 2.0)
                tactical_success = bool(
                    behind_score >= 0.55
                    and anchor_dist <= self.agent_radius * 3.5
                    and goal_dist <= cover_goal_cap
                )
            if goal_dist < self.success_radius * 0.6:
                cover_bonus -= 0.05
            bonus += cover_bonus
            terms["role_cover"] = cover_bonus
        elif role_id == ROLE_BASE_MOVE:
            old_goal_dist = float(np.linalg.norm(self.goal_pos - old_pos))
            old_path = old_path_dist if old_path_dist is not None else self._geo_distance_robust(old_pos, max_search=3)
            new_path = new_path_dist if new_path_dist is not None else self._geo_distance_robust(new_pos, max_search=3)
            if old_path is None:
                old_path = float(np.linalg.norm(self.goal_pos - old_pos))
            if new_path is None:
                new_path = float(np.linalg.norm(self.goal_pos - new_pos))

            path_progress = max(0.0, float(old_path) - float(new_path))
            path_progress_bonus = 0.08 * min(path_progress / max(self.step_size, 1.0), 1.5)
            base_bonus = path_progress_bonus
            terms["role_base_path_progress"] = path_progress_bonus
            entered_sense_radius = bool(old_goal_dist > float(self.sense_radius) and goal_dist <= float(self.sense_radius))
            if entered_sense_radius:
                base_bonus += 0.12
                terms["role_base_enter_radius"] = 0.12
            if detour_used:
                base_bonus += 0.02
            if collided:
                base_bonus -= 0.03
            speed = float(np.linalg.norm(self.agent_velocities[agent_index]))
            speed_bonus = 0.04 * min(speed / max(self.step_size, 1.0), 1.0)
            base_bonus += speed_bonus
            terms["role_base_speed"] = speed_bonus
            bonus += base_bonus
            terms["role_base_move"] = base_bonus
            tactical_success = bool(
                entered_sense_radius
                and (not collided)
                and speed >= self.step_size * 0.45
            )
        elif role_id == ROLE_SURROUND:
            surround_radius = max(self.success_radius * 1.6, self.agent_radius * 2.2)
            my_dist, angle_gap, ally_count = self._surround_stats(agent_index, pos=new_pos)
            ring_error = abs(my_dist - surround_radius)
            surround_bonus = 0.10 * max(0.0, 1.0 - ring_error / max(surround_radius, 1.0))
            surround_bonus += 0.08 * min(angle_gap / (math.pi / 2.0), 1.0)
            if ally_count >= 2:
                surround_bonus += 0.04
            bonus += surround_bonus
            terms["role_surround"] = surround_bonus
            tactical_success = bool(
                ally_count >= 2
                and ring_error <= self.agent_radius * 1.5
                and angle_gap >= 0.70
            )
        else:
            kiting_min_dist = max(0.0, float(self.sense_radius) - 100.0)
            kiting_max_dist = max(kiting_min_dist + 1.0, float(self.sense_radius))
            kiting_radius = 0.5 * (kiting_min_dist + kiting_max_dist)
            band_half_width = max(1.0, 0.5 * (kiting_max_dist - kiting_min_dist))
            ring_error = abs(goal_dist - kiting_radius)
            in_kiting_band = bool(kiting_min_dist <= goal_dist <= kiting_max_dist)
            velocity_dir = self._normalize_vec(self.agent_velocities[agent_index])
            retreat_alignment = float(np.dot(velocity_dir, -goal_dir))
            kiting_bonus = 0.12 * max(0.0, 1.0 - ring_error / band_half_width)
            if in_kiting_band:
                kiting_bonus += 0.08
            kiting_bonus += 0.05 * max(0.0, retreat_alignment)
            if goal_dist < kiting_min_dist:
                kiting_bonus -= 0.10 * min((kiting_min_dist - goal_dist) / max(self.success_radius, 1.0), 1.5)
            bonus += kiting_bonus
            terms["role_kiting"] = kiting_bonus
            tactical_success = bool(
                in_kiting_band
                and retreat_alignment >= 0.15
            )

        terms["tactical_success"] = 1.0 if tactical_success else 0.0
        return float(bonus), terms, tactical_success

    def _pack_single_observation(self, agent_index: int) -> np.ndarray:
        scale = max(self.map_range, 1.0)
        pos2 = self.agent_positions[agent_index]
        height = self.agent_heights[agent_index]
        agent_pos3 = np.array([pos2[0], height, pos2[1]], dtype=np.float32)
        goal_pos3 = np.array([self.goal_pos[0], self.goal_height, self.goal_pos[1]], dtype=np.float32)
        delta3 = goal_pos3 - agent_pos3
        center3 = np.array([self.map_center[0], 0.0, self.map_center[1]], dtype=np.float32)

        agent_norm = (agent_pos3 - center3) / scale
        goal_norm = (goal_pos3 - center3) / scale
        delta_norm = delta3 / scale
        vel_norm = self.agent_velocities[agent_index] / max(self.step_size, 1.0)

        other_obs, fail_code = self._sense_local_space(agent_index, scale)
        obs = np.concatenate(
            [
                agent_norm.astype(np.float32),
                goal_norm.astype(np.float32),
                delta_norm.astype(np.float32),
                vel_norm.astype(np.float32),
                other_obs,
                np.array([fail_code], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _pack_observation(self) -> np.ndarray:
        return np.stack([self._pack_single_observation(i) for i in range(self.num_agents)], axis=0).astype(np.float32)

    def _sample_spawn_point(
            self,
            avoid_points: List[np.ndarray],
            min_dist: float,
            tries: int = 128,
            anchor: Optional[np.ndarray] = None,
            max_dist: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        min_dist_sq = min_dist * min_dist
        max_dist_sq = None
        anchor_pos = None
        if anchor is not None and max_dist is not None and float(max_dist) > 0.0:
            max_dist_sq = float(max_dist) * float(max_dist)
            anchor_pos = np.asarray(anchor, dtype=np.float32).reshape(-1)[:2]
        for _ in range(tries):
            idx = int(self.rng.randrange(len(self._free_cells)))
            rc = tuple(int(x) for x in self._free_cells[idx])
            pos = self._grid_rc_to_world(rc[0], rc[1])
            if max_dist_sq is not None and anchor_pos is not None:
                adx = float(pos[0] - anchor_pos[0])
                adz = float(pos[1] - anchor_pos[1])
                if adx * adx + adz * adz > max_dist_sq:
                    continue
            ok = True
            for avoid in avoid_points:
                dx = float(pos[0] - avoid[0])
                dz = float(pos[1] - avoid[1])
                if dx * dx + dz * dz < min_dist_sq:
                    ok = False
                    break
            if ok:
                return pos, float(self._height_map[rc[0], rc[1]])

        if anchor_pos is not None and max_dist_sq is not None:
            best_pos = None
            best_height = None
            best_d2 = None
            for rc in self._free_cells:
                pos = self._grid_rc_to_world(int(rc[0]), int(rc[1]))
                adx = float(pos[0] - anchor_pos[0])
                adz = float(pos[1] - anchor_pos[1])
                anchor_d2 = adx * adx + adz * adz
                if anchor_d2 > max_dist_sq:
                    continue
                ok = True
                for avoid in avoid_points:
                    dx = float(pos[0] - avoid[0])
                    dz = float(pos[1] - avoid[1])
                    if dx * dx + dz * dz < min_dist_sq:
                        ok = False
                        break
                if not ok:
                    continue
                if best_d2 is None or anchor_d2 < best_d2:
                    best_pos = pos
                    best_height = float(self._height_map[int(rc[0]), int(rc[1])])
                    best_d2 = anchor_d2
            if best_pos is not None:
                return best_pos.astype(np.float32), float(best_height)

        idx = int(self.rng.randrange(len(self._free_cells)))
        rc = tuple(int(x) for x in self._free_cells[idx])
        return self._grid_rc_to_world(rc[0], rc[1]), float(self._height_map[rc[0], rc[1]])

    def _reset_all_agents(self) -> None:
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.agent_heights = np.zeros((self.num_agents,), dtype=np.float32)
        self.agent_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.last_target_offsets = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.role_targets = np.zeros((self.num_agents, 2), dtype=np.float32)

        self.agent_positions[0] = self.agent_pos.copy()
        self.agent_heights[0] = self.agent_height
        avoid = [self.agent_pos.copy(), self.goal_pos.copy()]
        min_dist = max(self.agent_radius * 1.0, self.success_radius * self.agent_spawn_min_scale)
        max_dist = max(min_dist, self.success_radius * self.agent_spawn_max_scale)
        for idx in range(1, self.num_agents):
            pos, height = self._sample_spawn_point(
                avoid,
                min_dist=min_dist,
                anchor=self.agent_pos,
                max_dist=max_dist,
            )
            self.agent_positions[idx] = pos
            self.agent_heights[idx] = height
            avoid.append(pos.copy())

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        options = options or {}
        if seed is not None:
            self._seed_value = seed
            self.rng = random.Random(seed)
            self.nprng = np.random.default_rng(seed)

        start_idx = int(self.rng.randrange(len(self._free_cells)))
        start_rc = tuple(int(x) for x in self._free_cells[start_idx])
        self.agent_pos = self._grid_rc_to_world(start_rc[0], start_rc[1])
        self.agent_height = float(self._height_map[start_rc[0], start_rc[1]])

        self._geo_map = None
        self.goal_pos = self.agent_pos.copy()
        self.goal_height = self.agent_height

        min_goal_dist = max(self.step_size * 2.0, self.success_radius * self.goal_spawn_min_scale)
        for _ in range(128):
            goal_idx = int(self.rng.randrange(len(self._free_cells)))
            goal_rc = tuple(int(x) for x in self._free_cells[goal_idx])
            goal_pos = self._grid_rc_to_world(goal_rc[0], goal_rc[1])
            if float(np.linalg.norm(goal_pos - self.agent_pos)) < min_goal_dist:
                continue
            geo_map = self._compute_geodesic_map(goal_rc)
            if not math.isfinite(float(geo_map[start_rc[0], start_rc[1]])):
                continue
            self.goal_pos = goal_pos
            self.goal_height = float(self._height_map[goal_rc[0], goal_rc[1]])
            self._geo_map = geo_map
            self._geo_goal_rc = goal_rc
            break

        if self._geo_map is None:
            candidate_indices = list(range(len(self._free_cells)))
            self.rng.shuffle(candidate_indices)
            best_goal = None
            best_geo = None
            best_goal_dist = -1.0
            for idx in candidate_indices:
                goal_rc = tuple(int(x) for x in self._free_cells[idx])
                goal_pos = self._grid_rc_to_world(goal_rc[0], goal_rc[1])
                goal_dist = float(np.linalg.norm(goal_pos - self.agent_pos))
                if goal_dist < min_goal_dist:
                    continue
                geo_map = self._compute_geodesic_map(goal_rc)
                start_geo = float(geo_map[start_rc[0], start_rc[1]])
                if not math.isfinite(start_geo):
                    continue
                if goal_dist > best_goal_dist:
                    best_goal = goal_rc
                    best_geo = geo_map
                    best_goal_dist = goal_dist
            if best_goal is None:
                best_goal = start_rc
                best_geo = self._compute_geodesic_map(best_goal)
            self.goal_pos = self._grid_rc_to_world(best_goal[0], best_goal[1])
            self.goal_height = float(self._height_map[best_goal[0], best_goal[1]])
            self._geo_map = best_geo
            self._geo_goal_rc = best_goal

        self._reset_all_agents()
        self._assign_roles()
        self._update_role_targets()

        self.steps = 0
        self._prev_geo = np.full((self.num_agents,), np.nan, dtype=np.float32)
        self._stall_best = np.full((self.num_agents,), np.nan, dtype=np.float32)
        self._stall_wait = np.zeros((self.num_agents,), dtype=np.int32)
        self._prev_success_mask = np.zeros((self.num_agents,), dtype=bool)
        for idx in range(self.num_agents):
            geo = self._geo_distance(self.agent_positions[idx])
            self._prev_geo[idx] = np.nan if geo is None else geo
            self._stall_best[idx] = geo if geo is not None else float(np.linalg.norm(self.goal_pos - self.agent_positions[idx]))

        if self.dynamic_horizon:
            self.max_steps = self._compute_dynamic_horizon()
        else:
            self.max_steps = self.base_max_steps

        return self._pack_observation(), {}

    def step(self, action):
        acts = np.asarray(action, dtype=np.float32)
        if acts.ndim == 1:
            acts = np.repeat(acts.reshape(1, -1), self.num_agents, axis=0)
        if acts.shape != (self.num_agents, 2):
            acts = np.zeros((self.num_agents, 2), dtype=np.float32)

        old_positions = self.agent_positions.copy()
        step_role_ids = self.agent_role_ids.copy()
        old_geos = np.array([self._geo_distance(p) for p in old_positions], dtype=object)
        old_role_targets = self.role_targets.copy()
        sensor_fail_codes = np.zeros((self.num_agents,), dtype=np.float32)
        detour_used_codes = np.zeros((self.num_agents,), dtype=np.float32)
        detour_attempt_codes = np.zeros((self.num_agents,), dtype=np.float32)
        detour_targets = np.zeros_like(self.agent_positions)
        detour_waypoints = np.full_like(self.agent_positions, np.nan, dtype=np.float32)
        tactical_targets = np.zeros_like(self.agent_positions)
        requested_targets = np.zeros_like(self.agent_positions)
        collisions = np.zeros((self.num_agents,), dtype=bool)
        rewards = np.full((self.num_agents,), -self.time_penalty, dtype=np.float32)
        terms_list = [{"time_penalty": -self.time_penalty} for _ in range(self.num_agents)]

        for idx in range(self.num_agents):
            old_pos = old_positions[idx]
            _, fail_code = self._sense_local_space(idx, max(self.map_range, 1.0))
            sensor_fail_codes[idx] = fail_code
            target_offset = np.clip(acts[idx], -1.0, 1.0) * self.tactical_target_radius
            if fail_code > 0.5:
                desired_target = self.goal_pos.copy()
            else:
                desired_target = old_pos + target_offset
            snapped = self._nearest_valid_point(desired_target, max_radius=6)
            snapped_target = old_pos.copy() if snapped is None else snapped[0]
            snapped_height = None if snapped is None else float(snapped[1])
            detour_targets[idx] = snapped_target

            if self._detour_enabled:
                detour_attempt_codes[idx] = 1.0
                waypoint = self._detour_next_waypoint_to_target(
                    old_pos,
                    snapped_target,
                    height=float(self.agent_heights[idx]),
                    target_height=snapped_height,
                )
                detour_used_codes[idx] = 1.0 if waypoint is not None else 0.0
                if waypoint is not None:
                    detour_waypoints[idx] = waypoint
                tactical_target = snapped_target if waypoint is None else waypoint
            elif fail_code > 0.5:
                waypoint = self._geo_next_waypoint(old_pos, max_search=3)
                tactical_target = snapped_target if waypoint is None else waypoint
            else:
                tactical_target = snapped_target

            to_target = tactical_target - old_pos
            target_dist = float(np.linalg.norm(to_target))
            step_size = self.step_size
            if target_dist > step_size and target_dist > 1e-6:
                movement_target = old_pos + (to_target / target_dist) * step_size
            else:
                movement_target = tactical_target

            new_pos, new_height, collided = self._move_with_agent_avoidance(
                old_pos,
                movement_target,
                ignore_index=idx,
                start_height=float(self.agent_heights[idx]),
            )
            self.agent_positions[idx] = new_pos
            self.agent_heights[idx] = new_height
            self.agent_velocities[idx] = new_pos - old_pos
            self.last_target_offsets[idx] = target_offset.astype(np.float32)
            tactical_targets[idx] = tactical_target
            requested_targets[idx] = desired_target
            collisions[idx] = collided

        self.agent_pos = self.agent_positions[0].copy()
        self.agent_height = float(self.agent_heights[0])
        self.steps += 1

        dists = np.linalg.norm(self.goal_pos[None, :] - self.agent_positions, axis=1).astype(np.float32)
        success_mask = np.zeros((self.num_agents,), dtype=bool)
        truncated = self.steps >= self.max_steps
        terminated = False

        geo_dists = []
        for idx in range(self.num_agents):
            old_geo = old_geos[idx]
            new_geo = self._geo_distance(self.agent_positions[idx])
            geo_dists.append(new_geo)
            if old_geo is not None and new_geo is not None:
                cur_metric = float(new_geo)
            else:
                cur_metric = float(dists[idx])

            if collisions[idx]:
                rewards[idx] -= self.collision_penalty
                terms_list[idx]["collision_penalty"] = -self.collision_penalty

            self.agent_role_ids[idx] = step_role_ids[idx]
            self.role_targets[idx] = old_role_targets[idx]
            if sensor_fail_codes[idx] > 0.5:
                terms_list[idx]["role_none"] = 1.0
                terms_list[idx]["tactical_success"] = 0.0
                success_mask[idx] = False
            else:
                role_bonus, role_terms, tactical_success = self._role_reward_bonus(
                    idx,
                    old_positions[idx],
                    self.agent_positions[idx],
                    bool(collisions[idx]),
                    bool(detour_used_codes[idx] > 0.5),
                    old_path_dist=(float(old_geo) if old_geo is not None else None),
                    new_path_dist=(float(new_geo) if new_geo is not None else None),
                )
                self.role_targets[idx] = self._role_target_for(idx)
                rewards[idx] += role_bonus
                terms_list[idx].update(role_terms)
                success_mask[idx] = bool(tactical_success)

            best = float(self._stall_best[idx])
            if np.isnan(best) or cur_metric < best - 1.0:
                self._stall_best[idx] = cur_metric
                self._stall_wait[idx] = 0
            else:
                self._stall_wait[idx] += 1
                if self._stall_wait[idx] >= self.stall_patience:
                    rewards[idx] -= self.stall_penalty
                    terms_list[idx]["stall_penalty"] = -self.stall_penalty

            self._prev_geo[idx] = np.nan if new_geo is None else float(new_geo)

            was_success = bool(self._prev_success_mask[idx])
            is_success = bool(success_mask[idx])
            if is_success and not was_success:
                rewards[idx] += self._R_SUCCESS_ENTRY
                terms_list[idx]["success_entry"] = self._R_SUCCESS_ENTRY
            elif is_success:
                rewards[idx] += self._R_SUCCESS_SUSTAIN
                terms_list[idx]["success_sustain"] = self._R_SUCCESS_SUSTAIN
            elif was_success:
                rewards[idx] -= self._R_SUCCESS_DROP
                terms_list[idx]["success_drop"] = -self._R_SUCCESS_DROP

        terminated = False

        self.agent_pos = self.agent_positions[0].copy()
        self.agent_height = float(self.agent_heights[0])

        info_role_ids = step_role_ids.copy()
        info_role_ids[sensor_fail_codes > 0.5] = ROLE_NONE

        info = {
            "dist_to_goal": dists.copy(),
            "collided": collisions.copy(),
            "geo_dist": np.array([np.nan if g is None else float(g) for g in geo_dists], dtype=np.float32),
            "reward_terms": terms_list,
            "agent_heights": self.agent_heights.copy(),
            "goal_height": float(self.goal_height),
            "tactical_target": tactical_targets.copy(),
            "requested_target": requested_targets.copy(),
            "agent_positions": self.agent_positions.copy(),
            "success_mask": success_mask.copy(),
            "role_ids": info_role_ids.copy(),
            "agent_role_rules": list(self.agent_role_rules) if self.agent_role_rules is not None else [self.role_rule] * self.num_agents,
            "role_targets": self.role_targets.copy(),
            "sensor_fail_code": sensor_fail_codes.copy(),
            "detour_used": detour_used_codes.copy(),
            "detour_attempted": detour_attempt_codes.copy(),
            "detour_target": detour_targets.copy(),
            "detour_waypoint": detour_waypoints.copy(),
            "detour_enabled": bool(self._detour_enabled),
            "detour_error": self._detour_last_error,
        }
        self._prev_success_mask = success_mask.copy()
        self.agent_role_ids[:] = self._compute_assigned_roles()
        self._update_role_targets()
        return self._pack_observation(), rewards.astype(np.float32), terminated, truncated, info

    def render(self, mode: str = "human"):
        print(
            f"[MajestroNavMeshEnv] pos={self.agent_pos} h={self.agent_height:.2f} "
            f"goal={self.goal_pos} gh={self.goal_height:.2f} steps={self.steps}/{self.max_steps}"
        )

    @property
    def R_SUCCESS(self):
        return self._R_SUCCESS

    @R_SUCCESS.setter
    def R_SUCCESS(self, value):
        self._R_SUCCESS = float(value)
