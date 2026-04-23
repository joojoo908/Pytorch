# SAC_PathFind_3d 환경 개요

## 요약

현재 환경은 navmesh 위에서 동작하는 멀티에이전트 전술 이동 환경입니다.

- 환경: `majestro_navmesh_env.py`
- 학습: `Model.py`
- 학습 실행: `Test.py`
- 평가 실행: `ModelTest.py`

기본값 기준:

- 기본 에이전트 수: `5`
- 단일 에이전트 관측 크기: `24`
- 단일 에이전트 행동 크기: `2`
- 관측 shape: `(num_agents, 24)`
- 행동 shape: `(num_agents, 2)`

주의:

- 학습과 평가는 episode마다 선택된 heuristic 조합 길이에 따라 실제 `num_agents`가 달라질 수 있습니다.
- `agent_role_rule_pool`을 쓰면 `1`, `3`, `4`, `5`명처럼 가변 agent 수로 episode가 돌아갈 수 있습니다.

## 좌표와 상태

이 환경은 수평 평면 `(x, z)` 기준으로 이동을 처리합니다.

- 실제 이동 위치: `(x, z)`
- 높이 정보: `y`

주요 내부 상태:

- `agent_positions`: `(num_agents, 2)`
- `agent_heights`: `(num_agents,)`
- `agent_velocities`: `(num_agents, 2)`
- `goal_pos`: `(2,)`
- `goal_height`: 스칼라

## 역할과 heuristic

현재 역할 ID는 다음과 같습니다.

- `0`: `front`
- `1`: `cover`
- `2`: `base_move`
- `3`: `surround`
- `4`: `kiting`
- `-1`: `none` (`sensor_fail_code == 1.0`인 fallback step)

현재 heuristic 규칙은 다음 세 가지입니다.

- `fixed`
- `melee_dps`
- `ranged_dps`

역할은 heuristic과 현재 배치에 따라 정해집니다.

### `fixed`

에이전트 순서대로 아래 역할을 반복합니다.

- `front`
- `cover`
- `base_move`
- `surround`
- `kiting`

### `melee_dps`

- goal이 `sense_radius` 밖이면 `base_move`
- goal이 `sense_radius` 안이고 주변에 다른 `melee_dps` actor가 있으면 `surround`
- 아니면 `front`

### `ranged_dps`

- goal이 `sense_radius` 밖이면 `base_move`
- goal이 `sense_radius` 안이고 주변에 `melee_dps` actor가 있으면 `cover`
- 아니면 `kiting`

## 관측 구조

단일 에이전트 관측은 총 `24`개입니다.

### 0-2: 내 위치

- `agent_norm_x`
- `agent_norm_y`
- `agent_norm_z`

### 3-5: goal 위치

- `goal_norm_x`
- `goal_norm_y`
- `goal_norm_z`

### 6-8: goal까지 상대 벡터

- `delta_norm_x`
- `delta_norm_y`
- `delta_norm_z`

### 9-10: 현재 속도

- `velocity_x`
- `velocity_z`

### 11-22: 주변 에이전트 정보

최대 `observed_other_agents`명, 기본 `3`명까지 들어갑니다.

각 슬롯은:

- `rel_x`
- `rel_z`
- `dist`
- `heuristic_id`

형태입니다.

### 23: 센서 실패 코드

- `0.0`: 반경 안에 다른 에이전트가 있음
- `1.0`: 반경 안에 다른 에이전트가 없음

## 행동 구조

행동은 `(dx, dz)` 두 값입니다.

```python
target_offset = clip(action, -1, 1) * tactical_target_radius
desired_target = current_position + target_offset
```

즉 행동은 즉시 이동량이 아니라 로컬 tactical target offset입니다.

## 센서 실패와 fallback

`sensor_fail_code == 1.0`이면 해당 step은 역할 actor를 직접 쓰지 않고 fallback 이동으로 처리됩니다.

동작:

- `desired_target`은 `goal_pos`
- Detour가 켜져 있으면 Detour waypoint 사용
- 아니면 geodesic waypoint fallback 사용
- `info["role_ids"]`에는 `-1` (`none`)으로 들어감
- 이 step은 role success 집계에서 빠짐

`ModelTest.py`에서는 이런 step에 대해 fallback 경로를 화면에 따로 그립니다.

## 보상 구조

보상은 에이전트별로 계산됩니다.

- `rewards.shape == (num_agents,)`

각 step 보상은 대략 다음 항목들의 합입니다.

### 1. 시간 패널티

모든 에이전트에 기본 적용:

```python
-time_penalty
```

기본값:

- `time_penalty = 0.01`

### 2. 충돌 패널티

충돌 시:

```python
-collision_penalty
```

기본값:

- `collision_penalty = 0.35`

### 3. Stall 패널티

geodesic 거리 기준 진전이 충분히 없으면:

```python
-stall_penalty
```

기본값:

- `stall_penalty = 0.05`
- `stall_patience = 20`

### 4. 공통 role progress 보상

모든 역할 공통으로 현재 role target에 가까워지면 보상을 받습니다.

```python
role_progress_reward = 0.03 * (old_dist_to_role_target - new_dist_to_role_target)
```

### 5. 역할별 shaping

#### `front`

- goal 방향 직접 압박
- `directness`가 좋을수록 가산

#### `cover`

- 다른 actor 뒤쪽 엄폐 형성
- goal이 `sense_radius` 밖이면 불리

#### `base_move`

- goal 직선 거리가 아니라 `경로 잔여 길이` 감소를 기준으로 보상
- geodesic 경로 길이 감소량을 `role_base_path_progress`로 반영
- 실제 이동 속도도 `role_base_speed`로 보상
- 충돌은 전역 패널티 외에 role 내부에서도 약하게 감점

#### `surround`

- goal 주위 적정 반경 형성
- ally와의 각도 분산 확보

#### `kiting`

- `sense_radius - 100`부터 `sense_radius` 사이 거리 band 유지
- goal 반대 방향 이동 성분 보상

## 성공 보상 구조

현재 성공 보상은 한 값 반복 지급이 아니라 `entry / sustain / drop` 구조입니다.

### 1. 성공 첫 진입

```python
success_entry = +20
```

### 2. 성공 유지

```python
success_sustain = +2
```

### 3. 성공 상태 이탈

```python
success_drop = -3
```

즉:

- 처음 전술 성공 상태에 들어가면 크게 보상
- 유지 중에는 작은 보상
- 성공 상태였다가 이탈하면 작은 패널티

예전의 `step마다 +50 반복 지급` 구조는 제거되었습니다.

## 역할별 성공 기준

현재 성공은 goal 반경 도달이 아니라 역할 전술 성공입니다.

대략:

- `front`: goal 근처에서 정면 압박 + 이동 방향성 확보
- `cover`: anchor 뒤 엄폐 위치 형성
- `base_move`: 충돌 없이 빠르게 이동하면서 goal 근처 진입
- `surround`: 반경/각도 분산 만족
- `kiting`: 거리 band 유지 + 후퇴 방향성 확보

## 종료 조건

역할 성공으로는 조기 종료하지 않습니다.

```python
terminated = False
truncated = (steps >= max_steps)
```

즉:

- 성공해도 episode는 계속 진행
- 시간 제한으로만 episode가 끝납니다

## step 처리 순서

한 step은 대략 다음 순서입니다.

1. 행동 입력 수신
2. tactical target 계산
3. navmesh 유효 위치 스냅
4. Detour 또는 geodesic waypoint 선택
5. 이동량 제한 적용
6. 실제 이동 및 agent 회피
7. 충돌/role shaping/stall/success 전이 보상 계산
8. `info` 구성
9. 다음 step용 역할 재할당

## `info` 주요 항목

`env.step()`은 다음과 같은 정보를 돌려줍니다.

- `dist_to_goal`
- `geo_dist`
- `collided`
- `reward_terms`
- `tactical_target`
- `requested_target`
- `agent_positions`
- `success_mask`
- `role_ids`
- `agent_role_rules`
- `role_targets`
- `sensor_fail_code`
- `detour_used`
- `detour_attempted`
- `detour_target`
- `detour_waypoint`
- `detour_enabled`
- `detour_error`

`reward_terms`에는 예를 들면 아래와 같은 key가 들어갑니다.

- `time_penalty`
- `collision_penalty`
- `stall_penalty`
- `role_progress`
- `role_front`
- `role_cover`
- `role_base_move`
- `role_base_path_progress`
- `role_base_speed`
- `role_surround`
- `role_kiting`
- `success_entry`
- `success_sustain`
- `success_drop`

## 학습 로그 의미

학습 로그는 episode 전체 success율이 아니라 최근 100 episode 기준 role별 step 성공률을 중심으로 봅니다.

예:

```text
role_step=front:12.0/cover:25.0/base_move:48.0/surround:8.0/kiting:30.0
role_step_n=front:12/100/cover:25/100/base_move:48/100/surround:8/100/kiting:30/100
succ_growth=front:10/cover:4/base_move:30/surround:2/kiting:7
```

의미:

- `role_step`: 해당 role step 중 성공 비율
- `role_step_n`: 성공 step 수 / 시도 step 수
- `succ_growth`: 최근 구간에서 role별 success replay buffer 증가량

## best 저장 방식

현재 best 저장은 role별로 따로 갱신됩니다.

- 어떤 role의 success buffer 증가량이 자기 최고 기록보다 좋아지면
- 그 role snapshot만 best 묶음 안에서 갱신
- 저장 파일은 여전히 전체 multi-role 파일 형태를 유지

즉 `sac_actor_best.pth`는 전체 actor 묶음이지만, 내부적으로는 role별 최신 best 조합입니다.

## 현재 핵심 변경점

현재 코드 기준으로 예전 문서와 달라진 핵심은 다음입니다.

- `flank_left`, `flank_right` 삭제
- goal progress reward 제거
- separation penalty 미사용
- `base_move`는 경로 잔여 길이 감소 기준
- 성공 보상은 `entry / sustain / drop`
- role pool로 episode마다 heuristic 조합 샘플링 가능
- role pool 길이에 따라 실제 agent 수가 episode마다 달라질 수 있음
- best 저장은 role별 갱신 방식
