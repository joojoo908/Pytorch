# arena_env_ma.py
import numpy as np


class MultiAgentArenaEnv:
    """
    2D 사각형 아레나 환경 (Multi-Agent)

    - arena 범위: [-map_range, map_range]^2
    - 플레이어 n_players, 몬스터 N(랜덤: monsters_min~monsters_max)
    - 액션(연속 3차원):
        a0: 목표 x 좌표 정규화 [-1,1] -> [-map_range, map_range]
        a1: 목표 y 좌표 정규화 [-1,1] -> [-map_range, map_range]
        a2: 공격 트리거 [-1,1] -> >0 이면 공격 시도
    - 관측:
        base(9) = (pos2, vel2, hp_norm, alive, team_alive_ratio2, time_frac)
        + 가까운 적 K명 (rel_pos2, hp_norm) => K*3
        + 가까운 아군 K명 (rel_pos2, hp_norm) => K*3

    [추가/수정 요약]
      1) 플레이어 원거리 공격(히트스캔/자동 타겟) 추가
      2) 맵 경계 밖으로 "나가려는 시도"에 패널티 추가
      3) viewer/train 코드 호환을 위해:
         - reset() -> (obs, info) 반환
         - k_nearest 인자 지원(k_near로 매핑)
         - obs_dim/act_dim/hp_*_max 속성 제공
      4) viewer에서 원거리 샷을 그릴 수 있도록 last_shots 이벤트 리스트 제공
         - step마다 초기화되며, 원거리 공격 시 {"start","end","hit","shooter","target","kind"}를 push
      5) [수정] 직접 이동 방식 제거
         - 액터가 "어떻게 움직일지"가 아니라 "어디로 갈지"를 정하면
           환경이 그 목표점을 향해 자동 이동
    """

    def __init__(
        self,
        seed=1,
        n_players=3,
        monsters_min=10,
        monsters_max=20,
        map_range=8.0,
        max_speed=0.25,
        # collision
        radius_player=0.15,
        radius_monster=0.12,
        # combat (근접 - 기존)
        attack_range=0.9,
        attack_damage=10.0,
        attack_cooldown_steps=6,
        # player ranged combat (추가)
        ranged_attack_range_player=3.5,
        ranged_attack_damage_player=6.0,
        ranged_attack_cooldown_steps_player=10,
        ranged_attack_requires_los=True,
        # reward
        time_penalty=0.01,
        penalty_out_of_bounds=0.2,  # 맵 경계를 넘어가는 목표를 찍으려는 시도 패널티
        reward_damage=0.02,         # 준 데미지에 비례 보상
        penalty_damage=0.02,        # 받은 데미지에 비례 패널티(현재는 미적용)
        reward_kill=4.0,
        reward_win=15.0,
        reward_lose=-15.0,

        # 몬스터 proximity 보너스(플레이어에 가까울수록 +)
        monster_proximity_bonus=0.02,      # step당 최대 보너스
        monster_proximity_dist=6.0,        # 거리 스케일(클수록 완만)
        monster_proximity_use_alive_only=True,  # 살아있는 플레이어만 거리 계산

        # obs
        k_near=3,
        k_nearest=None,  # 구버전 코드 호환: k_nearest 인자 지원
        max_steps=300,
        # obstacles
        obstacles=None,
    ):
        self.rng = np.random.RandomState(seed)

        self.n_players = int(n_players)
        self.monsters_min = int(monsters_min)
        self.monsters_max = int(monsters_max)
        self.map_range = float(map_range)
        self.max_speed = float(max_speed)

        self.radius_player = float(radius_player)
        self.radius_monster = float(radius_monster)

        # 근접(기존)
        self.attack_range = float(attack_range)
        self.attack_damage = float(attack_damage)
        self.attack_cd = int(attack_cooldown_steps)

        # 플레이어 원거리 공격 파라미터
        self.ranged_attack_range_p = float(ranged_attack_range_player)
        self.ranged_attack_damage_p = float(ranged_attack_damage_player)
        self.ranged_attack_cd_p = int(ranged_attack_cooldown_steps_player)
        self.ranged_attack_requires_los = bool(ranged_attack_requires_los)

        self.time_penalty = float(time_penalty)
        self.penalty_out_of_bounds = float(penalty_out_of_bounds)
        self.reward_damage = float(reward_damage)
        self.penalty_damage = float(penalty_damage)
        self.reward_kill = float(reward_kill)
        self.reward_win = float(reward_win)
        self.reward_lose = float(reward_lose)

        # 몬스터 proximity 보너스 파라미터
        self.monster_proximity_bonus = float(monster_proximity_bonus)
        self.monster_proximity_dist = float(monster_proximity_dist)
        self.monster_proximity_use_alive_only = bool(monster_proximity_use_alive_only)

        # 구버전 호환: k_nearest -> k_near
        if k_nearest is not None:
            k_near = k_nearest
        self.k_near = int(k_near)

        self.max_steps = int(max_steps) if max_steps is not None else None

        # obstacles: list of (x, y, radius)
        self.obstacles = obstacles if obstacles is not None else [
            (0.0, 0.0, 0.7),
            (1.8, -1.0, 0.6),
            (-1.5, 1.4, 0.55),
        ]

        # viewer/train 호환용 속성
        self.act_dim = 3
        # base(9) + enemy(K*3) + ally(K*3)
        self.obs_dim = 9 + 6 * self.k_near
        self.hp_player_max = 100.0
        self.hp_monster_max = 40.0

        # step 단위 발사 이벤트(뷰어 트레이서용)
        self.last_shots = []

        # state
        self.steps = 0
        self.monsters_n = 0

        # arrays
        self.p_pos = None
        self.p_vel = None
        self.p_hp = None
        self.p_cd = None

        self.m_pos = None
        self.m_vel = None
        self.m_hp = None
        self.m_cd = None

        self.reset()

    def _agent_ids(self):
        ids = [f"P{i}" for i in range(self.n_players)]
        ids += [f"M{i}" for i in range(self.monsters_n)]
        return ids

    def reset(self):
        self.steps = 0
        self.monsters_n = int(self.rng.randint(self.monsters_min, self.monsters_max + 1))

        # 발사 이벤트 초기화
        self.last_shots = []

        # players
        self.p_pos = np.zeros((self.n_players, 2), dtype=np.float32)
        self.p_vel = np.zeros((self.n_players, 2), dtype=np.float32)
        self.p_hp = np.ones((self.n_players,), dtype=np.float32) * self.hp_player_max
        self.p_cd = np.zeros((self.n_players,), dtype=np.int32)

        # monsters
        self.m_pos = np.zeros((self.monsters_n, 2), dtype=np.float32)
        self.m_vel = np.zeros((self.monsters_n, 2), dtype=np.float32)
        self.m_hp = np.ones((self.monsters_n,), dtype=np.float32) * self.hp_monster_max
        self.m_cd = np.zeros((self.monsters_n,), dtype=np.int32)

        left_x = -0.6 * self.map_range
        right_x = 0.6 * self.map_range

        # 플레이어는 왼쪽에 세로로 배치
        ys = np.linspace(-0.3 * self.map_range, 0.3 * self.map_range, self.n_players)
        for i in range(self.n_players):
            self.p_pos[i] = np.array([left_x, ys[i]], dtype=np.float32)

        # 몬스터는 오른쪽에 y 랜덤 배치
        for i in range(self.monsters_n):
            y = self.rng.uniform(-0.8 * self.map_range, 0.8 * self.map_range)
            self.m_pos[i] = np.array([right_x, y], dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action_dict):
        self.steps += 1

        # 이번 step의 원거리 발사 이벤트 초기화(뷰어 트레이서용)
        self.last_shots = []

        # 기본 보상: 시간 패널티
        rew = {aid: -self.time_penalty for aid in self._agent_ids()}

        # cd 감소
        self.p_cd = np.maximum(self.p_cd - 1, 0)
        self.m_cd = np.maximum(self.m_cd - 1, 0)

        # 1) 이동
        self._apply_moves(action_dict, rew)

        # 몬스터만 플레이어 근접 보너스
        self._apply_monster_proximity_bonus(rew)

        # 2) 공격
        self._apply_attacks(action_dict, rew)

        # 종료 판정
        done = False
        trunc = False

        players_alive = int(np.sum(self.p_hp > 0))
        monsters_alive = int(np.sum(self.m_hp > 0))

        if monsters_alive <= 0:
            done = True
            # 플레이어 승
            for i in range(self.n_players):
                rew[f"P{i}"] += self.reward_win
            for i in range(self.monsters_n):
                rew[f"M{i}"] += self.reward_lose
        elif players_alive <= 0:
            done = True
            # 몬스터 승
            for i in range(self.n_players):
                rew[f"P{i}"] += self.reward_lose
            for i in range(self.monsters_n):
                rew[f"M{i}"] += self.reward_win

        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            trunc = True

        obs = self._get_obs()
        info = {aid: {} for aid in self._agent_ids()}
        term = {aid: done for aid in self._agent_ids()}
        truncs = {aid: trunc for aid in self._agent_ids()}

        return obs, rew, term, truncs, info

    # -----------------------
    # Observation
    # -----------------------
    def _get_obs(self):
        obs = {}
        # 전역 정보
        p_alive = float(np.sum(self.p_hp > 0)) / float(self.n_players)
        m_alive = float(np.sum(self.m_hp > 0)) / float(max(1, self.monsters_n))
        time_frac = float(self.steps) / float(max(1, self.max_steps if self.max_steps else 1))

        for i in range(self.n_players):
            obs[f"P{i}"] = self._build_obs_for_player(i, p_alive, m_alive, time_frac)
        for i in range(self.monsters_n):
            obs[f"M{i}"] = self._build_obs_for_monster(i, p_alive, m_alive, time_frac)
        return obs

    def _build_obs_for_player(self, idx, p_alive, m_alive, time_frac):
        me_pos = self.p_pos[idx]
        me_vel = self.p_vel[idx]
        me_hp = self.p_hp[idx]
        me_alive = 1.0 if me_hp > 0 else 0.0

        enemy_feats = self._nearest_entities(me_pos, self.m_pos, self.m_hp, self.k_near)
        ally_feats = self._nearest_entities(me_pos, self.p_pos, self.p_hp, self.k_near, exclude_idx=idx)

        base = np.array([
            me_pos[0], me_pos[1],
            me_vel[0], me_vel[1],
            me_hp / self.hp_player_max,
            me_alive,
            p_alive, m_alive, time_frac
        ], dtype=np.float32)

        return np.concatenate([base, enemy_feats, ally_feats], axis=0)

    def _build_obs_for_monster(self, idx, p_alive, m_alive, time_frac):
        me_pos = self.m_pos[idx]
        me_vel = self.m_vel[idx]
        me_hp = self.m_hp[idx]
        me_alive = 1.0 if me_hp > 0 else 0.0

        enemy_feats = self._nearest_entities(me_pos, self.p_pos, self.p_hp, self.k_near)
        ally_feats = self._nearest_entities(me_pos, self.m_pos, self.m_hp, self.k_near, exclude_idx=idx)

        base = np.array([
            me_pos[0], me_pos[1],
            me_vel[0], me_vel[1],
            me_hp / self.hp_monster_max,
            me_alive,
            m_alive, p_alive, time_frac  # monster perspective
        ], dtype=np.float32)

        return np.concatenate([base, enemy_feats, ally_feats], axis=0)

    def _nearest_entities(self, me_pos, all_pos, all_hp, k, exclude_idx=None):
        alive = all_hp > 0
        if exclude_idx is not None:
            alive = alive.copy()
            alive[exclude_idx] = False

        idxs = np.nonzero(alive)[0]
        if len(idxs) == 0:
            return np.zeros((k * 3,), dtype=np.float32)

        pos = all_pos[idxs]
        hp = all_hp[idxs]

        d = np.linalg.norm(pos - me_pos[None, :], axis=1)
        order = np.argsort(d)[:k]

        out = []
        for j in order:
            rel = pos[j] - me_pos
            # 기존 코드 호환 위해 hp는 100 기준으로 정규화
            out.extend([float(rel[0]), float(rel[1]), float(hp[j]) / 100.0])

        while len(out) < k * 3:
            out.extend([0.0, 0.0, 0.0])

        return np.asarray(out, dtype=np.float32)

    # -----------------------
    # Action decode
    # -----------------------
    def _decode_target_pos(self, a, radius):
        """
        [수정]
        a[0], a[1]을 목표 위치로 해석한다.

        - a0: 목표 x 정규화 [-1,1]
        - a1: 목표 y 정규화 [-1,1]

        반환:
            raw_target    : 액션이 가리킨 원래 목표점
            target_clamped: 맵 내부로 보정된 목표점
            overflow      : 경계 바깥으로 지정한 양
        """
        raw_target = np.array([
            float(np.clip(a[0], -1.0, 1.0)) * self.map_range,
            float(np.clip(a[1], -1.0, 1.0)) * self.map_range,
        ], dtype=np.float32)

        min_x = -self.map_range + radius
        max_x = self.map_range - radius
        min_y = -self.map_range + radius
        max_y = self.map_range - radius

        target_clamped = raw_target.copy()
        target_clamped[0] = np.clip(target_clamped[0], min_x, max_x)
        target_clamped[1] = np.clip(target_clamped[1], min_y, max_y)

        overflow = float(np.linalg.norm(raw_target - target_clamped))
        return raw_target, target_clamped, overflow

    def _compute_delta_towards_target(self, cur_pos, target_pos):
        """
        [추가]
        현재 위치에서 목표 위치까지 자동 이동 벡터를 계산한다.
        한 step당 최대 self.max_speed 만큼만 이동한다.
        목표가 가까우면 정확히 목표 지점에서 멈춘다.
        """
        to_target = target_pos - cur_pos
        dist = float(np.linalg.norm(to_target))

        if dist < 1e-8:
            return np.zeros((2,), dtype=np.float32)

        step_len = min(self.max_speed, dist)
        d = to_target / dist * step_len
        return d.astype(np.float32)

    # -----------------------
    # Movement & collision
    # -----------------------
    def _apply_moves(self, action_dict, rew_dict):
        # 플레이어
        for i in range(self.n_players):
            if self.p_hp[i] <= 0:
                self.p_vel[i] = np.zeros((2,), dtype=np.float32)
                continue

            a = np.asarray(action_dict.get(f"P{i}", np.zeros(3, dtype=np.float32)), dtype=np.float32)

            # [수정] 기존의 각도/속도 기반 직접 이동 제거
            #       목표 위치를 해석한 뒤 그쪽으로 자동 이동
            raw_target, target_pos, overflow = self._decode_target_pos(a, self.radius_player)
            d = self._compute_delta_towards_target(self.p_pos[i], target_pos)

            # [수정] 맵 밖 목표를 지정하면 패널티
            if overflow > 0.0:
                rew_dict[f"P{i}"] -= self.penalty_out_of_bounds * (overflow / (self.max_speed + 1e-6))

            new_pos, new_vel = self._move_with_collision(self.p_pos[i], d, self.radius_player)
            self.p_vel[i] = new_vel
            self.p_pos[i] = new_pos

        # 몬스터
        for i in range(self.monsters_n):
            if self.m_hp[i] <= 0:
                self.m_vel[i] = np.zeros((2,), dtype=np.float32)
                continue

            a = np.asarray(action_dict.get(f"M{i}", np.zeros(3, dtype=np.float32)), dtype=np.float32)

            # [수정] 몬스터도 동일하게 목표 위치 기반 자동 이동
            raw_target, target_pos, overflow = self._decode_target_pos(a, self.radius_monster)
            d = self._compute_delta_towards_target(self.m_pos[i], target_pos)

            if overflow > 0.0:
                rew_dict[f"M{i}"] -= self.penalty_out_of_bounds * (overflow / (self.max_speed + 1e-6))

            new_pos, new_vel = self._move_with_collision(self.m_pos[i], d, self.radius_monster)
            self.m_vel[i] = new_vel
            self.m_pos[i] = new_pos

    def _move_with_collision(self, pos, delta, radius):
        # 1) 경계 충돌(사각형)
        p = pos + delta
        p[0] = np.clip(p[0], -self.map_range + radius, self.map_range - radius)
        p[1] = np.clip(p[1], -self.map_range + radius, self.map_range - radius)

        # 2) 장애물 충돌(원형) - 간단하게 밀어내기
        for (ox, oy, orad) in self.obstacles:
            c = np.array([ox, oy], dtype=np.float32)
            v = p - c
            dist = float(np.linalg.norm(v) + 1e-8)
            min_d = float(orad + radius)
            if dist < min_d:
                # obstacle 밖으로 push
                n = v / dist
                p = c + n * min_d

        vel = (p - pos).astype(np.float32)
        return p.astype(np.float32), vel

    # -----------------------
    # Combat helpers
    # -----------------------
    def _has_line_of_sight(self, p0, p1):
        """
        p0->p1 선분이 원형 장애물에 의해 막히는지 검사.
        - True: 시야 확보
        - False: 장애물이 가로막음
        """
        if len(self.obstacles) == 0:
            return True

        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        d = p1 - p0
        denom = float(np.dot(d, d)) + 1e-8

        for (ox, oy, orad) in self.obstacles:
            c = np.array([ox, oy], dtype=np.float32)

            t = float(np.dot(c - p0, d) / denom)
            t = float(np.clip(t, 0.0, 1.0))
            closest = p0 + d * t

            dist = float(np.linalg.norm(closest - c))
            if dist < float(orad):
                return False

        return True

    def _segment_first_circle_hit(self, p0, p1):
        """
        p0->p1 선분이 장애물 원과 교차하면 가장 가까운 교차점 반환.
        교차가 없으면 None 반환.
        viewer에서 LOS가 막혔을 때 tracer 끝점을 찍기 위해 사용.
        """
        if len(self.obstacles) == 0:
            return None

        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        d = p1 - p0
        a = float(np.dot(d, d))
        if a < 1e-8:
            return None

        best_t = None
        best_pt = None

        for (ox, oy, orad) in self.obstacles:
            c = np.array([ox, oy], dtype=np.float32)
            f = p0 - c

            b = 2.0 * float(np.dot(f, d))
            cc = float(np.dot(f, f)) - float(orad) * float(orad)

            disc = b * b - 4.0 * a * cc
            if disc < 0.0:
                continue

            sqrt_disc = float(np.sqrt(disc))
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)

            for t in (t1, t2):
                if 0.0 <= t <= 1.0:
                    if best_t is None or t < best_t:
                        best_t = t
                        best_pt = p0 + d * t

        if best_pt is None:
            return None
        return best_pt.astype(np.float32)

    # -----------------------
    # Monster proximity reward
    # -----------------------
    def _apply_monster_proximity_bonus(self, rew_dict):
        """
        몬스터에게만:
        - 가장 가까운 플레이어까지의 거리가 가까울수록 보너스 지급
        - 보너스 = monster_proximity_bonus * exp(-dist / monster_proximity_dist)
        """
        if self.monster_proximity_bonus <= 0.0:
            return
        if self.monster_proximity_dist <= 1e-6:
            return

        if self.monster_proximity_use_alive_only:
            alive_p = (self.p_hp > 0)
        else:
            alive_p = np.ones((self.n_players,), dtype=bool)

        if not np.any(alive_p):
            return

        ppos = self.p_pos[alive_p]

        for mi in range(self.monsters_n):
            if self.m_hp[mi] <= 0:
                continue

            mpos = self.m_pos[mi]
            dists = np.linalg.norm(ppos - mpos[None, :], axis=1)
            dist = float(np.min(dists))

            bonus = self.monster_proximity_bonus * float(np.exp(-dist / self.monster_proximity_dist))
            rew_dict[f"M{mi}"] += bonus

    # -----------------------
    # Combat
    # -----------------------
    def _apply_attacks(self, action_dict, rew_dict):
        # 플레이어 공격 -> 몬스터 피격
        for pi in range(self.n_players):
            if self.p_hp[pi] <= 0:
                continue
            if self.p_cd[pi] > 0:
                continue

            a = np.asarray(action_dict.get(f"P{pi}", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            if a[2] <= 0.0:
                continue

            alive = self.m_hp > 0
            if not np.any(alive):
                continue

            epos = self.m_pos[alive]
            idx_map = np.nonzero(alive)[0]
            dists = np.linalg.norm(epos - self.p_pos[pi][None, :], axis=1)
            j = int(np.argmin(dists))
            dist = float(dists[j])
            tid = int(idx_map[j])

            did_attack = False

            # 2-A) 근접 공격
            if dist <= (self.attack_range + self.radius_monster):
                killed, dmg = self._try_attack(
                    attacker_pos=self.p_pos[pi],
                    target_pos=self.m_pos,
                    target_hp=self.m_hp,
                    target_radius=self.radius_monster,
                    damage=self.attack_damage,
                    range_=self.attack_range,
                )
                if dmg > 0.0:
                    rew_dict[f"P{pi}"] += self.reward_damage * float(dmg)
                if killed:
                    rew_dict[f"P{pi}"] += self.reward_kill

                self.p_cd[pi] = self.attack_cd
                did_attack = True

            # 2-B) 원거리 공격
            elif dist <= (self.ranged_attack_range_p + self.radius_monster):
                p0 = self.p_pos[pi].copy()
                p1 = self.m_pos[tid].copy()

                hit = True
                if self.ranged_attack_requires_los and (not self._has_line_of_sight(p0, p1)):
                    hit = False

                end_pt = p1
                if not hit:
                    hit_pt = self._segment_first_circle_hit(p0, p1)
                    if hit_pt is not None:
                        end_pt = hit_pt

                self.last_shots.append({
                    "kind": "ranged",
                    "shooter": f"P{pi}",
                    "target": f"M{tid}",
                    "start": p0,
                    "end": end_pt.copy(),
                    "hit": bool(hit),
                })

                if hit:
                    before = float(self.m_hp[tid])
                    self.m_hp[tid] = max(0.0, self.m_hp[tid] - float(self.ranged_attack_damage_p))
                    after = float(self.m_hp[tid])

                    dealt = before - after
                    killed = (after <= 0.0)

                    if dealt > 0.0:
                        rew_dict[f"P{pi}"] += self.reward_damage * float(dealt)
                    if killed:
                        rew_dict[f"P{pi}"] += self.reward_kill

                # LOS에 막혀도 발사 시도는 했으므로 쿨다운 소비
                self.p_cd[pi] = self.ranged_attack_cd_p
                did_attack = True

            if not did_attack:
                pass

        # 몬스터 공격 -> 플레이어 피격 (근접만)
        for mi in range(self.monsters_n):
            if self.m_hp[mi] <= 0:
                continue
            if self.m_cd[mi] > 0:
                continue

            a = np.asarray(action_dict.get(f"M{mi}", np.zeros(3, dtype=np.float32)), dtype=np.float32)
            if a[2] <= 0.0:
                continue

            killed, dmg = self._try_attack(
                attacker_pos=self.m_pos[mi],
                target_pos=self.p_pos,
                target_hp=self.p_hp,
                target_radius=self.radius_player,
                damage=self.attack_damage,
                range_=self.attack_range,
            )
            if dmg > 0.0:
                rew_dict[f"M{mi}"] += self.reward_damage * float(dmg)
            if killed:
                rew_dict[f"M{mi}"] += self.reward_kill

            self.m_cd[mi] = self.attack_cd

        # NOTE:
        # penalty_damage(피격 패널티)는 누가 맞았는지 추적하거나,
        # step 시작 시 hp 스냅샷을 저장했다가 hp diff로 계산해야 정확히 적용 가능.
        # 현재 버전은 구현 편의상 생략.

    def _try_attack(self, attacker_pos, target_pos, target_hp, target_radius, damage, range_):
        alive = target_hp > 0
        if not np.any(alive):
            return False, 0.0

        epos = target_pos[alive]
        idx_map = np.nonzero(alive)[0]

        d = np.linalg.norm(epos - attacker_pos[None, :], axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= float(range_) + float(target_radius):
            tid = int(idx_map[j])
            before = float(target_hp[tid])
            target_hp[tid] = max(0.0, float(target_hp[tid]) - float(damage))
            after = float(target_hp[tid])
            dealt = before - after
            killed = (after <= 0.0)
            return killed, float(dealt)

        return False, 0.0