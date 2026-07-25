import math
from dataclasses import dataclass

import numpy as np


PLAYER_ROLES = ("bass", "drum", "guitar")

BRASS_SKILL4_MIN_X = -2625.0
BRASS_SKILL4_MAX_X = 2625.0
BRASS_SKILL4_MIN_Z = -1250.0
BRASS_SKILL4_MAX_Z = 1250.0
BRASS_SKILL4_TILE_SIZE = 250.0
BRASS_SKILL4_DAMAGE = 50.0

DRAGON_SKILL4_CENTER_COUNT = 5
DRAGON_SKILL4_SPAWN_RADIUS = 2000.0
DRAGON_SKILL4_EXPLOSION_RADIUS = 700.0
DRAGON_SKILL4_DAMAGE = 50.0


@dataclass
class BossEnvConfig:
    n_players: int = 3
    max_steps: int = 80
    map_size: float = 3000.0
    boss_move_speed: float = 220.0
    boss_max_hp: float = 0.0
    player_max_hps: tuple[float, float, float] = (200.0, 150.0, 250.0)
    underused_skill_bonus: float = 0.75
    overused_skill_penalty: float = 0.65
    immediate_repeat_penalty: float = 0.45
    recent_repeat_penalty: float = 0.15
    skill_cycle_bonus: float = 1.0
    terminal_uniformity_bonus: float = 2.0
    randomize_party_composition: bool = False
    full_party_probability: float = 0.50
    two_player_probability: float = 0.25
    seed: int = 7
    boss_kind: str = "brass"


class BossPatternEnv:
    """
    Boss pattern selection toy environment.

    - step 1회 = 보스가 대상 1명과 이동/스킬 행동 1개 선택
    - action: (target, choice), choice 0=이동만, 1~4=보스 스킬 1~4
    - observation: global 5 + player당 5차원

    이 환경은 실제 서버를 그대로 복제하지는 않지만,
    "3인 파티 상태를 보고 4개 패턴 중 하나를 고른다"는 학습 구조를
    빠르게 검증하기 위한 샌드박스 역할을 한다.
    """

    def __init__(self, config: BossEnvConfig | None = None):
        self.cfg = config or BossEnvConfig()
        if self.cfg.boss_max_hp <= 0.0:
            self.cfg.boss_max_hp = 5000.0 if self.cfg.boss_kind.lower() == "brass" else 1200.0
        self.rng = np.random.RandomState(self.cfg.seed)
        self.skill_option_count = 5
        self.target_dim = self.cfg.n_players
        self.choice_dim = self.skill_option_count
        self.action_dims = (self.target_dim, self.choice_dim)
        self.obs_dim = 5 + self.cfg.n_players * 5
        self.reset()

    def reset(self):
        self.step_count = 0
        self.boss_hp = self.cfg.boss_max_hp
        self.last_target = 0
        self.last_skill_choice = 0
        self.skill_usage_counts = np.zeros(4, dtype=np.float32)
        self.recent_skill_choices = []
        self.recent_damage_taken = 0.0
        self.dragon_dot_ticks = np.zeros((self.cfg.n_players,), dtype=np.int32)
        self.summon_pressure_ticks = 0

        self.boss_pos = np.zeros(2, dtype=np.float32)
        self.player_pos = np.zeros((self.cfg.n_players, 2), dtype=np.float32)
        self.player_max_hp = np.asarray(self.cfg.player_max_hps, dtype=np.float32)
        if self.player_max_hp.shape[0] != self.cfg.n_players:
            raise ValueError("player_max_hps length must match n_players")
        self.player_hp = self.player_max_hp.copy()
        self.player_roles = PLAYER_ROLES[: self.cfg.n_players]
        self.player_alive = np.ones((self.cfg.n_players,), dtype=np.float32)
        self.player_recent_damage = np.zeros((self.cfg.n_players,), dtype=np.float32)
        self.player_recent_boss_dps = np.zeros((self.cfg.n_players,), dtype=np.float32)
        self.player_recent_heal = np.zeros((self.cfg.n_players,), dtype=np.float32)
        self.player_threat_score = np.zeros((self.cfg.n_players,), dtype=np.float32)

        base_angles = np.linspace(-1.15, 1.15, self.cfg.n_players)
        for i, angle in enumerate(base_angles):
            radius = self.rng.uniform(900.0, 1400.0)
            self.player_pos[i, 0] = math.cos(angle) * radius
            self.player_pos[i, 1] = math.sin(angle) * radius

        if self.cfg.randomize_party_composition:
            roll = float(self.rng.rand())
            if roll < self.cfg.full_party_probability:
                alive_count = self.cfg.n_players
            elif roll < self.cfg.full_party_probability + self.cfg.two_player_probability:
                alive_count = max(1, self.cfg.n_players - 1)
            else:
                alive_count = 1

            survivor_indices = self.rng.choice(
                self.cfg.n_players, size=alive_count, replace=False
            )
            self.player_alive.fill(0.0)
            self.player_alive[survivor_indices] = 1.0
            self.player_hp.fill(0.0)
            for idx in survivor_indices:
                self.player_hp[idx] = self.player_max_hp[idx] * self.rng.uniform(0.25, 1.0)
            self.boss_hp = self.cfg.boss_max_hp * self.rng.uniform(0.25, 1.0)

        self._clip_positions_to_map()

        return self._build_obs(), {}

    def step(self, action):
        if not isinstance(action, (tuple, list, np.ndarray)) or len(action) != 2:
            raise ValueError("action must be (target_idx, skill_choice)")
        target_idx = int(np.clip(action[0], 0, self.target_dim - 1))
        skill_choice = int(np.clip(action[1], 0, self.choice_dim - 1))
        self.step_count += 1
        self.player_recent_damage.fill(0.0)
        self.player_recent_boss_dps.fill(0.0)
        self.player_recent_heal.fill(0.0)
        self.player_threat_score = self._compute_player_threat_scores()
        ongoing_damage, ongoing_note = self._apply_ongoing_effects()

        target_valid = bool(self.player_alive[target_idx] > 0.5)
        if skill_choice == 0 and target_valid:
            move_distance = self._move_boss_toward(target_idx)
            skill_note = f"move_toward_p{target_idx}:{move_distance:.1f}"
        elif skill_choice == 0:
            move_distance = 0.0
            skill_note = f"move_failed_dead_target_p{target_idx}"
        else:
            move_distance = 0.0
            skill_note = ""

        players_to_boss, support_heal = self._players_attack_boss()
        self.boss_hp = max(0.0, self.boss_hp - players_to_boss)
        self.recent_damage_taken = players_to_boss

        boss_hp_before_pattern = self.boss_hp
        if self.boss_hp <= 0.0 or skill_choice == 0:
            instant_damage = 0.0
            weighted_damage = float(getattr(self, "_ongoing_weighted_damage", 0.0))
            kill_count = int(getattr(self, "_ongoing_kill_count", 0))
            kill_reward = float(getattr(self, "_ongoing_kill_reward", 0.0))
        else:
            instant_damage, weighted_damage, kill_count, kill_reward, skill_note = (
                self._apply_boss_pattern(skill_choice - 1, target_idx)
            )
        boss_self_heal = max(0.0, self.boss_hp - boss_hp_before_pattern)
        self._update_player_motion()
        boss_damage = ongoing_damage + instant_damage

        balance_reward, cycle_completed = self._record_skill_choice(skill_choice)

        reward = 0.0
        reward += 0.03 * boss_damage
        reward += 0.012 * weighted_damage
        reward += kill_reward
        reward -= 0.015 * players_to_boss
        reward += 0.02 * boss_self_heal
        reward += 0.001 * move_distance if skill_choice == 0 else 0.0
        reward -= 0.25 if not target_valid else 0.0
        reward += balance_reward
        if skill_choice == 0 and self.last_skill_choice == 0:
            reward -= 0.10

        done = False
        if self.boss_hp <= 0.0:
            reward -= 4.0
            done = True

        if np.sum(self.player_alive) <= 0:
            reward += 6.0
            done = True

        if self.step_count >= self.cfg.max_steps:
            done = True

        skill_uniformity = self._skill_uniformity()
        if done and np.sum(self.skill_usage_counts) >= 4.0:
            reward += self.cfg.terminal_uniformity_bonus * skill_uniformity

        self.last_target = target_idx
        self.last_skill_choice = skill_choice
        obs = self._build_obs()
        info = {
            "action": (target_idx, skill_choice),
            "target_idx": target_idx,
            "target_role": self.player_roles[target_idx],
            "target_valid": target_valid,
            "skill_choice": skill_choice,
            "move_distance": move_distance,
            "skill_balance_reward": balance_reward,
            "skill_cycle_completed": cycle_completed,
            "skill_usage_counts": self.skill_usage_counts.copy(),
            "skill_uniformity": skill_uniformity,
            "boss_position": self.boss_pos.copy(),
            "boss_damage_dealt": boss_damage,
            "instant_boss_damage": instant_damage,
            "ongoing_boss_damage": ongoing_damage,
            "weighted_boss_damage": weighted_damage,
            "boss_damage_taken": players_to_boss,
            "boss_self_heal": boss_self_heal,
            "player_heal_done": support_heal,
            "players_alive": int(np.sum(self.player_alive)),
            "kill_count": kill_count,
            "kill_reward": kill_reward,
            "skill_note": skill_note,
            "ongoing_note": ongoing_note,
            "dragon_dot_ticks": self.dragon_dot_ticks.copy(),
            "player_roles": list(self.player_roles),
            "player_threat_score": self.player_threat_score.copy(),
            "player_recent_boss_dps": self.player_recent_boss_dps.copy(),
            "player_recent_heal": self.player_recent_heal.copy(),
        }
        return obs, reward, done, False, info

    def _record_skill_choice(self, skill_choice: int):
        if skill_choice == 0:
            return 0.0, False

        skill_idx = skill_choice - 1
        selected_count = float(self.skill_usage_counts[skill_idx])
        minimum_count = float(np.min(self.skill_usage_counts))
        count_gap = selected_count - minimum_count

        reward = (
            self.cfg.underused_skill_bonus
            if count_gap <= 1e-6
            else -self.cfg.overused_skill_penalty * count_gap
        )
        if self.recent_skill_choices and self.recent_skill_choices[-1] == skill_choice:
            reward -= self.cfg.immediate_repeat_penalty
        elif skill_choice in self.recent_skill_choices[-2:]:
            reward -= self.cfg.recent_repeat_penalty

        self.skill_usage_counts[skill_idx] += 1.0
        self.recent_skill_choices.append(skill_choice)
        self.recent_skill_choices = self.recent_skill_choices[-3:]

        cycle_completed = float(np.min(self.skill_usage_counts)) > minimum_count
        if cycle_completed:
            reward += self.cfg.skill_cycle_bonus
        return reward, cycle_completed

    def _skill_uniformity(self):
        total = float(np.sum(self.skill_usage_counts))
        if total <= 0.0:
            return 0.0
        distribution = self.skill_usage_counts / total
        l1_distance = float(np.sum(np.abs(distribution - 0.25)))
        return float(np.clip(1.0 - l1_distance / 1.5, 0.0, 1.0))

    def _players_attack_boss(self) -> float:
        total = 0.0
        total_heal = 0.0
        alive_indices = np.where(self.player_alive > 0.5)[0]
        for idx in alive_indices:
            dps, heal = self._player_profile_output(int(idx))
            self.player_recent_boss_dps[idx] = dps
            self.player_recent_heal[idx] = heal
            total += dps
            total_heal += heal
        if total_heal > 0.0:
            self._apply_player_healing(total_heal)
        return total, total_heal

    def _player_profile_output(self, idx: int):
        dist = self._distance_to_player(idx)
        role = self.player_roles[idx]
        if role == "bass":
            return 26.0, 10.0
        if role == "drum":
            if dist <= 250.0:
                return 78.0, 0.0
            if dist <= 550.0:
                return 42.0, 0.0
            if dist <= 900.0:
                return 18.0, 0.0
            return 8.0, 0.0
        if dist <= 280.0:
            return 48.0, 0.0
        if dist <= 700.0:
            return 72.0, 0.0
        if dist <= 950.0:
            return 26.0, 0.0
        return 10.0, 0.0

    def _apply_player_healing(self, total_heal: float):
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if alive_indices.size == 0:
            return
        injured = [int(idx) for idx in alive_indices if self.player_hp[idx] < self.player_max_hp[idx]]
        if not injured:
            return
        ordered = sorted(injured, key=lambda idx: self.player_hp[idx] / self.player_max_hp[idx])
        remaining = total_heal
        for idx in ordered:
            if remaining <= 0.0:
                break
            missing = float(self.player_max_hp[idx] - self.player_hp[idx])
            healed = min(missing, remaining)
            self.player_hp[idx] += healed
            remaining -= healed

    def _compute_player_threat_scores(self):
        scores = np.zeros((self.cfg.n_players,), dtype=np.float32)
        for idx in range(self.cfg.n_players):
            if self.player_alive[idx] <= 0.5:
                continue
            role = self.player_roles[idx]
            dps, heal = self._player_profile_output(idx)
            role_bonus = 1.10 if role == "bass" else 1.20 if role == "guitar" else 1.00
            scores[idx] = dps + 0.8 * heal
            scores[idx] *= role_bonus
        return scores

    def _apply_boss_pattern(self, action: int, target_idx: int):
        if self.cfg.boss_kind.lower() == "brass":
            return self._apply_brass_pattern(action, target_idx)
        return self._apply_dragon_pattern(action, target_idx)

    def _apply_ongoing_effects(self):
        total_damage = 0.0
        weighted_damage = 0.0
        kill_count = 0
        kill_reward = 0.0
        notes = []
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if self.summon_pressure_ticks > 0:
            self.summon_pressure_ticks -= 1
            if alive_indices.size > 0:
                target_idx = int(self.rng.choice(alive_indices))
                summon_damage = 18.0 if self.cfg.boss_kind.lower() == "brass" else 14.0
                damage_result = self._damage_player(target_idx, summon_damage)
                total_damage += damage_result["damage"]
                weighted_damage += damage_result["weighted_damage"]
                kill_count += damage_result["kill_count"]
                kill_reward += damage_result["kill_reward"]
                notes.append(f"summon_pressure_p{target_idx}")

        for idx in range(self.cfg.n_players):
            if self.dragon_dot_ticks[idx] <= 0 or self.player_alive[idx] <= 0.5:
                continue
            self.dragon_dot_ticks[idx] -= 1
            damage_result = self._damage_player(idx, 10.0)
            total_damage += damage_result["damage"]
            weighted_damage += damage_result["weighted_damage"]
            kill_count += damage_result["kill_count"]
            kill_reward += damage_result["kill_reward"]
            notes.append(f"dragon_dot_p{idx}")

        self._ongoing_weighted_damage = weighted_damage
        self._ongoing_kill_count = kill_count
        self._ongoing_kill_reward = kill_reward
        return total_damage, ",".join(notes)

    def _nearest_alive_player(self):
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if alive_indices.size == 0:
            return None
        distances = np.linalg.norm(self.player_pos[alive_indices] - self.boss_pos, axis=1)
        return int(alive_indices[np.argmin(distances)])

    def _forward_to_player(self, target_idx: int):
        if self.player_alive[target_idx] <= 0.5:
            return np.array([1.0, 0.0], dtype=np.float32)
        direction = (self.player_pos[target_idx] - self.boss_pos).astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return np.array([1.0, 0.0], dtype=np.float32)
        return direction / norm

    def _distance_to_player(self, idx: int):
        return float(np.linalg.norm(self.player_pos[idx] - self.boss_pos))

    def _move_boss_toward(self, target_idx: int):
        offset = self.player_pos[target_idx] - self.boss_pos
        distance = float(np.linalg.norm(offset))
        if distance <= 1e-6:
            return 0.0
        move_distance = min(float(self.cfg.boss_move_speed), distance)
        self.boss_pos += offset / distance * move_distance
        self._clip_positions_to_map()
        return move_distance

    def _angle_diff_deg(self, vec_a, vec_b):
        dot = float(np.dot(vec_a, vec_b))
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def _apply_brass_pattern(self, action: int, target_idx: int):
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if alive_indices.size == 0:
            return 0.0, 0.0, 0, 0.0, ""

        boss_damage = 0.0
        weighted_damage = float(getattr(self, "_ongoing_weighted_damage", 0.0))
        kill_count = int(getattr(self, "_ongoing_kill_count", 0))
        kill_reward = float(getattr(self, "_ongoing_kill_reward", 0.0))
        distances = np.linalg.norm(self.player_pos - self.boss_pos, axis=1)

        if action == 0:
            # Server: spawn 8 adds, no direct heal and no direct damage.
            self.summon_pressure_ticks = max(self.summon_pressure_ticks, 5)
            return 0.0, weighted_damage, kill_count, kill_reward, "spawn_pianoman2_bongoman1_hornman2_fly2_slime1"

        elif action == 1:
            # Server bullet: damage 100, speed 500, size 50, lifetime 5.
            if self.player_alive[target_idx] > 0.5 and distances[target_idx] <= 2500.0:
                damage_result = self._damage_player(target_idx, 100.0)
                boss_damage += damage_result["damage"]
                weighted_damage += damage_result["weighted_damage"]
                kill_count += damage_result["kill_count"]
                kill_reward += damage_result["kill_reward"]
                return boss_damage, weighted_damage, kill_count, kill_reward, f"projectile_target_p{target_idx}"
            return boss_damage, weighted_damage, kill_count, kill_reward, f"projectile_miss_p{target_idx}"

        elif action == 2:
            # Server: 16 shots over 8 beats, each 25 damage with 8-way spread repeated twice.
            base_dir = self._forward_to_player(target_idx)
            hit_counts = {int(idx): 0 for idx in alive_indices}
            for shot_idx in range(16):
                pattern_idx = shot_idx % 8
                shot_yaw_deg = -70.0 + 45.0 * pattern_idx
                shot_yaw_rad = math.radians(shot_yaw_deg)
                c = math.cos(shot_yaw_rad)
                s = math.sin(shot_yaw_rad)
                shot_dir = np.array(
                    [
                        base_dir[0] * c - base_dir[1] * s,
                        base_dir[0] * s + base_dir[1] * c,
                    ],
                    dtype=np.float32,
                )
                shot_dir /= max(np.linalg.norm(shot_dir), 1e-6)

                best_idx = None
                best_angle = 999.0
                for idx in alive_indices:
                    target_dir = (self.player_pos[idx] - self.boss_pos).astype(np.float32)
                    target_dist = float(np.linalg.norm(target_dir))
                    if target_dist > 2420.0 or target_dist <= 1e-6:
                        continue
                    target_dir /= target_dist
                    angle = self._angle_diff_deg(shot_dir, target_dir)
                    if angle <= 25.0 and angle < best_angle:
                        best_idx = int(idx)
                        best_angle = angle
                if best_idx is not None:
                    damage_result = self._damage_player(best_idx, 25.0)
                    boss_damage += damage_result["damage"]
                    weighted_damage += damage_result["weighted_damage"]
                    kill_count += damage_result["kill_count"]
                    kill_reward += damage_result["kill_reward"]
                    hit_counts[best_idx] += 1

            summary = ",".join(
                f"p{idx}x{count}" for idx, count in hit_counts.items() if count > 0
            )
            return boss_damage, weighted_damage, kill_count, kill_reward, f"spread_shots:{summary or 'no_hit'}"

        else:
            # Server: a fixed 21x10 checkerboard (250x250 tiles), then the
            # opposite parity, then the original parity. Each explosion deals 50.
            # The server board is translated so its center is the local origin;
            # observations remain boss-relative and keep the server's normalization.
            active_parity = int(self.rng.randint(0, 2))
            hit_counts = {}
            for idx in alive_indices:
                local_pos = self.player_pos[idx] - self.boss_pos
                x = float(local_pos[0])
                z = float(local_pos[1])
                if (
                    x < BRASS_SKILL4_MIN_X
                    or x >= BRASS_SKILL4_MAX_X
                    or z < BRASS_SKILL4_MIN_Z
                    or z >= BRASS_SKILL4_MAX_Z
                ):
                    continue

                x_index = int((x - BRASS_SKILL4_MIN_X) / BRASS_SKILL4_TILE_SIZE)
                z_index = int((z - BRASS_SKILL4_MIN_Z) / BRASS_SKILL4_TILE_SIZE)
                player_parity = (x_index + z_index) & 1
                explosion_hits = 2 if player_parity == active_parity else 1
                hit_counts[int(idx)] = explosion_hits
                for _ in range(explosion_hits):
                    damage_result = self._damage_player(int(idx), BRASS_SKILL4_DAMAGE)
                    boss_damage += damage_result["damage"]
                    weighted_damage += damage_result["weighted_damage"]
                    kill_count += damage_result["kill_count"]
                    kill_reward += damage_result["kill_reward"]

            summary = ",".join(f"p{idx}x{count}" for idx, count in hit_counts.items())
            return (
                boss_damage,
                weighted_damage,
                kill_count,
                kill_reward,
                f"checkerboard_A-B-A_4beat_damage50:{summary or 'no_hit'}",
            )

    def _apply_dragon_pattern(self, action: int, target_idx: int):
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if alive_indices.size == 0:
            return 0.0, 0.0, 0, 0.0, ""

        boss_damage = 0.0
        weighted_damage = float(getattr(self, "_ongoing_weighted_damage", 0.0))
        kill_count = int(getattr(self, "_ongoing_kill_count", 0))
        kill_reward = float(getattr(self, "_ongoing_kill_reward", 0.0))
        forward = self._forward_to_player(target_idx)
        attack_center = self.boss_pos + forward * 4.0

        if action == 0:
            # Server melee: damage 30, radius 1000, full circle, forward distance 4.
            for idx in alive_indices:
                if np.linalg.norm(self.player_pos[idx] - attack_center) <= 1000.0:
                    damage_result = self._damage_player(idx, 30.0)
                    boss_damage += damage_result["damage"]
                    weighted_damage += damage_result["weighted_damage"]
                    kill_count += damage_result["kill_count"]
                    kill_reward += damage_result["kill_reward"]
            return boss_damage, weighted_damage, kill_count, kill_reward, "circular_aoe"
        elif action == 1:
            # Server: heal 100 and summon one enemy group.
            self.boss_hp = min(self.cfg.boss_max_hp, self.boss_hp + 100.0)
            self.summon_pressure_ticks = max(self.summon_pressure_ticks, 5)
            return 0.0, weighted_damage, kill_count, kill_reward, "heal_100_and_spawn_group"
        elif action == 2:
            # Server melee: damage 50, center 200 forward, radius 300, knockback 600.
            attack_center = self.boss_pos + forward * 200.0
            for idx in alive_indices:
                offset = self.player_pos[idx] - attack_center
                if np.linalg.norm(offset) <= 300.0:
                    direction = (self.player_pos[idx] - self.boss_pos).astype(np.float32)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction /= norm
                        self.player_pos[idx] = self.player_pos[idx] + direction * 600.0
                        self._clip_positions_to_map()
                    damage_result = self._damage_player(idx, 50.0)
                    boss_damage += damage_result["damage"]
                    weighted_damage += damage_result["weighted_damage"]
                    kill_count += damage_result["kill_count"]
                    kill_reward += damage_result["kill_reward"]
            return boss_damage, weighted_damage, kill_count, kill_reward, "knockback_600_and_silence"
        else:
            # Server: five random telegraphed centers within radius 2000;
            # all explode four beats later with radius 700 and damage 50.
            hit_counts = {int(idx): 0 for idx in alive_indices}
            for _ in range(DRAGON_SKILL4_CENTER_COUNT):
                angle = self.rng.uniform(-math.pi, math.pi)
                radius = math.sqrt(self.rng.uniform(0.0, 1.0)) * DRAGON_SKILL4_SPAWN_RADIUS
                center = self.boss_pos + np.array(
                    [math.cos(angle) * radius, math.sin(angle) * radius],
                    dtype=np.float32,
                )
                for idx in alive_indices:
                    if np.linalg.norm(self.player_pos[idx] - center) <= DRAGON_SKILL4_EXPLOSION_RADIUS:
                        damage_result = self._damage_player(int(idx), DRAGON_SKILL4_DAMAGE)
                        boss_damage += damage_result["damage"]
                        weighted_damage += damage_result["weighted_damage"]
                        kill_count += damage_result["kill_count"]
                        kill_reward += damage_result["kill_reward"]
                        hit_counts[int(idx)] += 1

            summary = ",".join(
                f"p{idx}x{count}" for idx, count in hit_counts.items() if count > 0
            )
            return (
                boss_damage,
                weighted_damage,
                kill_count,
                kill_reward,
                f"five_random_explosions_4beat_damage50:{summary or 'no_hit'}",
            )

    def _damage_player(self, idx: int, amount: float):
        if self.player_alive[idx] <= 0.5:
            return {"damage": 0.0, "weighted_damage": 0.0, "kill_count": 0, "kill_reward": 0.0}
        prev = self.player_hp[idx]
        new_hp = max(0.0, self.player_hp[idx] - amount)
        actual_damage = float(prev - new_hp)
        self.player_hp[idx] = new_hp
        self.player_recent_damage[idx] += actual_damage
        threat = float(self.player_threat_score[idx])
        weighted_damage = actual_damage * (1.0 + threat / 100.0)
        kill_count = 0
        kill_reward = 0.0
        if prev > 0.0 and self.player_hp[idx] <= 0.0:
            self.player_alive[idx] = 0.0
            kill_count = 1
            kill_reward = 1.25 + 0.03 * threat
        return {
            "damage": actual_damage,
            "weighted_damage": weighted_damage,
            "kill_count": kill_count,
            "kill_reward": kill_reward,
        }

    def _update_player_motion(self):
        preferred_ranges = {"bass": 1100.0, "drum": 180.0, "guitar": 500.0}
        move_speeds = {"bass": 150.0, "drum": 170.0, "guitar": 140.0}
        for idx in range(self.cfg.n_players):
            if self.player_alive[idx] <= 0.5:
                continue
            role = self.player_roles[idx]
            offset = self.player_pos[idx] - self.boss_pos
            distance = float(np.linalg.norm(offset))
            if distance <= 1e-6:
                angle = self.rng.uniform(-math.pi, math.pi)
                radial = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            else:
                radial = offset / distance

            preferred = preferred_ranges[role]
            speed = move_speeds[role]
            if distance > preferred + 80.0:
                move_dir = -radial
            elif distance < preferred - 80.0:
                move_dir = radial
            else:
                tangent_sign = -1.0 if self.rng.rand() < 0.5 else 1.0
                move_dir = np.array([-radial[1], radial[0]], dtype=np.float32) * tangent_sign

            jitter = self.rng.uniform(-0.12, 0.12, size=2).astype(np.float32)
            move_dir = move_dir + jitter
            move_norm = float(np.linalg.norm(move_dir))
            if move_norm > 1e-6:
                self.player_pos[idx] += move_dir / move_norm * speed
        self._clip_positions_to_map()

    def _clip_positions_to_map(self):
        half_extent = self.cfg.map_size * 0.5
        self.boss_pos[:] = np.clip(self.boss_pos, -half_extent, half_extent)
        self.player_pos[:] = np.clip(self.player_pos, -half_extent, half_extent)

    def _build_obs(self):
        relative_positions = self.player_pos - self.boss_pos
        distances = np.linalg.norm(relative_positions, axis=1)
        max_distance = self.cfg.map_size * math.sqrt(2.0)
        obs = [
            self.boss_hp / self.cfg.boss_max_hp,
            self.step_count / self.cfg.max_steps,
            min(self.recent_damage_taken / 180.0, 1.0),
            self.last_skill_choice / float(max(1, self.skill_option_count - 1)),
            self.last_target / float(max(1, self.cfg.n_players - 1)),
        ]
        for idx in range(self.cfg.n_players):
            distance = float(distances[idx])
            if distance > 1e-6:
                direction = relative_positions[idx] / distance
                direction_x = float(direction[0])
                direction_z = float(direction[1])
            else:
                direction_x = 0.0
                direction_z = 0.0
            dist = float(np.clip(distance / max_distance, 0.0, 1.0))
            hp_ratio = float(np.clip(self.player_hp[idx] / self.player_max_hp[idx], 0.0, 1.0))
            alive = float(self.player_alive[idx])
            obs.extend([direction_x, direction_z, dist, hp_ratio, alive])

        return np.asarray(obs, dtype=np.float32)

    def _player_spread_norm(self):
        alive_indices = np.where(self.player_alive > 0.5)[0]
        if alive_indices.size <= 1:
            return 0.0
        alive_pos = self.player_pos[alive_indices]
        centroid = alive_pos.mean(axis=0)
        spread = np.mean(np.linalg.norm(alive_pos - centroid, axis=1))
        return float(np.clip(spread / self.cfg.map_size, 0.0, 1.0))
