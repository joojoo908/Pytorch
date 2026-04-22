# SAC_PathFind_3d 환경 개요

## 요약

현재 환경은 navmesh 위에서 동작하는 shared-policy 멀티에이전트 환경입니다.

- 환경 파일: `majestro_navmesh_env.py`
- 학습 파일: `Model.py`
- 학습 실행 파일: `Test.py`
- 평가 실행 파일: `ModelTest.py`

현재 기준:

- 에이전트 1명 기준 관측 크기: `24`
- 에이전트 1명 기준 행동 크기: `2`
- 전체 관측 shape: `(num_agents, 24)`
- 전체 행동 shape: `(num_agents, 2)`

기본값 기준:

- `num_other_agents = 4`
- `num_agents = 5`

## 기본 구조

이 환경은 이동을 수평 평면 기준으로 처리합니다.

- 실제 이동 위치: `(x, z)`
- 높이 정보: `y`

내부 상태:

- `agent_positions`: `(num_agents, 2)`
- `agent_heights`: `(num_agents,)`
- `goal_pos`: `(2,)`
- `goal_height`: 스칼라

즉 모델은 수평 이동만 제어하고, 높이는 navmesh에서 정해집니다.

## 역할 시스템

각 에이전트는 아래 역할 중 하나를 가집니다.

- `0`: `front`
- `1`: `cover`
- `2`: `base_move`
- `3`: `surround`
- `4`: `kiting`

역할은:

- 관측에 `role_id`로 들어가고
- 역할별 보상에 사용되고
- 역할별 성공 판정에도 사용됩니다

휴리스틱 규칙도 에이전트별로 따로 가질 수 있습니다.

- 공통 규칙: `role_rule`
- 개별 규칙: `agent_role_rules`

현재 지원 규칙:

- `fixed`
- `melee_dps`
- `ranged_dps`

## 로컬 탐지

현재 로컬 탐지는 `sense_radius` 안의 주변 몹만 봅니다.

- `sense_radius` 안의 다른 에이전트만 관측에 포함
- 반경 안에 주변 몹이 하나도 없으면 탐지 실패

기본값:

- `sense_radius = 600.0`

실행 시 조절:

```bash
python Test.py --sense-radius 500
python ModelTest.py --sense-radius 500
```

## 입력 24개

### 0-2: 현재 위치

- `agent_norm_x`
- `agent_norm_y`
- `agent_norm_z`

### 3-5: 목표 위치

- `goal_norm_x`
- `goal_norm_y`
- `goal_norm_z`

### 6-8: 목표까지 상대 벡터

- `delta_norm_x`
- `delta_norm_y`
- `delta_norm_z`

### 9-10: 현재 속도

- `velocity_x`
- `velocity_z`

### 11-22: 주변 몹 정보 + 휴리스틱

- `other_0_rel_x`
- `other_0_rel_z`
- `other_0_dist`
- `other_0_heuristic`
- `other_1_rel_x`
- `other_1_rel_z`
- `other_1_dist`
- `other_1_heuristic`
- `other_2_rel_x`
- `other_2_rel_z`
- `other_2_dist`
- `other_2_heuristic`

의미:

- `sense_radius` 안의 다른 에이전트 중 가장 가까운 최대 3명의 상대 위치/거리/휴리스틱
- 부족한 슬롯은 `0`으로 채워짐

### 23: 센서 실패 코드

- `sensor_fail_code`
- `0.0`: 반경 안에 주변 몹 있음
- `1.0`: 반경 안에 주변 몹 없음

현재는 자기 자신의 휴리스틱을 따로 넣지 않고, 관측한 다른 에이전트 슬롯에 그 에이전트의 휴리스틱을 함께 넣습니다.

휴리스틱 정규화 규칙:

- `fixed -> 0.00`
- `melee_dps -> 0.50`
- `ranged_dps -> 1.00`

즉 정책은 "내가 무슨 heuristic인가"보다 "내 주변에 보이는 다른 에이전트들이 어떤 heuristic인가"를 입력으로 받습니다.

`melee_dps`의 역할 선택 규칙은 현재 다음과 같습니다.

- goal이 sense radius 밖이면 `base_move`
- goal이 sense radius 안이고 주변에 다른 `melee_dps` actor가 있으면 `surround`
- goal이 sense radius 안이고 주변에 다른 `melee_dps` actor가 없으면 `front`

`ranged_dps`의 역할 선택 규칙은 현재 다음과 같습니다.

- goal이 sense radius 밖이면 `base_move`
- goal이 sense radius 안이고 주변에 `melee_dps` actor가 있으면 `cover`
- goal이 sense radius 안이고 주변에 `melee_dps` actor가 없으면 `kiting`

## 출력 2개

- `action[0]`: `dx`
- `action[1]`: `dz`

의미:

- navmesh 평면에서의 로컬 이동 오프셋

기본 해석:

```python
target_offset = clip(action, -1, 1) * tactical_target_radius
desired_target = current_position + target_offset
```

## F=1 fallback

현재는 `sensor_fail_code == 1.0`이면 명시적인 `none` 상태로 처리합니다.

즉:

- 평소: role actor의 정책 action 기반 이동
- `F=1`: role actor를 선택하지 않고 action은 zero로 둠
- 환경 이동은 goal 방향 Detour waypoint 또는 geodesic waypoint fallback을 사용
- `info["role_ids"]`에는 해당 agent role이 `-1`, 즉 `none`으로 표시됨
- 이 step은 role replay buffer와 success replay buffer에 저장되지 않음

이 fallback은 정책이 주변 몹이 안 보이는 상황에서 장거리 추적을 직접 학습하지 않아도 목표 쪽으로 우회 이동할 수 있게 하기 위한 장치이며, role 학습 샘플과는 분리됩니다.

## 보상 구조

보상은 에이전트별로 계산됩니다.

- `rewards.shape == (num_agents,)`

구성:

### 1. 시간 패널티

```python
-time_penalty
```

기본값:

- `0.01`

### 2. 충돌 패널티

```python
-collision_penalty
```

기본값:

- `0.35`

### 3. Stall 패널티

진전이 없으면 감점합니다.

기본값:

- `stall_penalty = 0.05`
- `stall_patience = 20`

### 4. 역할 보상

역할별 shaping reward를 추가합니다.

- `front`: 정면 압박
- `cover`: 다른 액터 뒤 엄폐, 단 goal이 `sense_radius` 밖이면 감점 및 성공 실패
- `base_move`: 빠르게 접근
- `surround`: 포위 반경과 각도 분산
- `kiting`: goal과 `sense_radius - 100`에서 `sense_radius` 사이 거리를 유지하며 이탈 방향 움직임

### 5. 성공 보상

큰 성공 보상은 goal 도달이 아니라 역할별 전술 위치 형성 시 지급됩니다.

```python
if success_mask[idx]:
    rewards[idx] += self._R_SUCCESS
```

기본값:

- `success_reward = 50.0`

## 성공 기준

현재 성공은 goal 반경 도달 기준이 아닙니다.

`success_mask[idx]`는 “그 에이전트가 자기 역할 전술을 만족했는가”를 뜻합니다.

대략:

- `front`: goal 근처에서 정면 압박 형성
- `cover`: goal이 `sense_radius` 안에 있는 상태에서 다른 몹 뒤 엄폐 위치 형성
- `base_move`: 충돌 없이 빠르게 접근하고 goal 근처까지 진입
- `surround`: goal 주위 반경과 각도 분산 형성
- `kiting`: goal과 `sense_radius - 100`에서 `sense_radius` 사이 거리 유지

## 종료 조건

현재 환경은 역할 성공으로 조기 종료하지 않습니다.

```python
terminated = False
```

즉:

- 각 에이전트는 개별 전술 성공 상태를 가짐
- 성공한 에이전트는 `success_reward`를 받지만 episode는 계속 진행
- episode는 시간 제한에 걸릴 때 종료

시간 초과는:

```python
steps >= max_steps
```

## 한 턴 처리 순서

### 1. 행동 입력 받기

- shape: `(num_agents, 2)`
- 1차원 입력이면 전원에 복제

### 2. 이전 상태 저장

- 이전 위치
- 이전 geodesic 거리
- 이전 역할 목표점

### 3. 주변 몹 탐지

- `sense_radius` 안의 주변 몹 최대 3개 수집
- 하나도 없으면 `sensor_fail_code = 1.0`

### 4. 목표점 결정

- 기본: `old_pos + action offset`
- `F=1`: geodesic map 기준 다음 waypoint

### 5. Navmesh 유효 위치로 스냅

- invalid target이면 유효한 근처 점으로 보정

### 6. 한 step 이동량 제한

- 모든 에이전트는 같은 `move_step_size` 사용

### 7. 실제 이동

- navmesh 유효성 검사
- 높이 샘플링
- 에이전트 간 충돌 회피

### 8. 기본 보상 계산

- 시간 패널티
- 충돌/stall 패널티

### 9. 역할 보상 및 역할 성공 판정

- 역할별 전술 평가
- 역할 shaping reward 추가
- `success_mask[idx]` 결정

### 10. 성공 보상 지급

- 전술 성공한 에이전트만 큰 보상 지급

### 11. 종료 조건 계산

- 역할 성공과 무관하게 `terminated = False`
- 시간 제한이면 `truncated = True`

### 12. 다음 관측 반환

```python
obs, rewards, terminated, truncated, info
```

`info`에는 다음 같은 값이 들어갑니다.

- `agent_positions`
- `tactical_target`
- `role_targets`
- `role_ids`: `sensor_fail_code == 1.0`인 agent는 `-1`/`none`
- `success_mask`
- `sensor_fail_code`
- `reward_terms`

## 학습 성공률 의미

현재 `Model.py` 학습 로그는 episode 전체 성공률이 아니라 최근 100 episode 기준 역할별 step 성공률을 중심으로 봅니다.

예:

```text
role_step=front:0.0/cover:12.5/base_move:30.0/surround:8.0/kiting:20.0
role_step_n=front:0/0/cover:5/40/base_move:30/100/surround:4/50/kiting:10/50
```

의미:

- `role_step`: 해당 role로 움직인 step 중 `success_mask=True`가 된 비율
- `role_step_n`: 해당 role의 `성공 step 수 / 시도 step 수`
- 시도 step이 없는 role은 분모가 0으로 표시될 수 있음
- `none` fallback step은 어떤 role에도 집계되지 않음

best 저장은 최근 `best_min_episodes` 범위에서 `succ_replay_buffer` 총량이 얼마나 증가했는지를 기준으로 합니다.

alpha freeze는 `succ_replay_buffer` 총량이 `alpha_freeze_succbuf` 이상이 되었는지만 봅니다.

## 현재 코드 핵심 변경점

현재 코드 기준 핵심은 다음입니다.

- 장애물 ray 관측 제거
- 주변 몹 반경 탐지 도입
- `sensor_fail_code` 도입
- `F=1`일 때 geodesic waypoint fallback
- 모든 에이전트 이동 속도 동일
- goal 도달이 아니라 역할 전술 형성이 성공 기준
- 역할 성공으로 episode를 조기 종료하지 않음
- 학습 로그는 episode success 대신 역할별 step 성공률 중심으로 집계
