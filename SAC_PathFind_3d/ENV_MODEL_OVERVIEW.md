# SAC_PathFind_3d 환경 개요

## 요약

현재 환경은 navmesh 위에서 동작하는 shared-policy 멀티에이전트 환경입니다.

- 환경 파일: `majestro_navmesh_env.py`
- 학습 파일: `Model.py`
- 학습 실행 파일: `Test.py`
- 평가 실행 파일: `ModelTest.py`

현재 기준:

- 에이전트 1명 기준 관측 크기: `22`
- 에이전트 1명 기준 행동 크기: `2`
- 전체 관측 shape: `(num_agents, 22)`
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
- `1`: `flank_left`
- `2`: `flank_right`
- `3`: `cover`

역할은:

- 관측에 `role_id`로 들어가고
- 역할별 보상에 사용되고
- 역할별 성공 판정에도 사용됩니다

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

## 입력 22개

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

### 11-19: 주변 몹 정보

- `other_0_rel_x`
- `other_0_rel_z`
- `other_0_dist`
- `other_1_rel_x`
- `other_1_rel_z`
- `other_1_dist`
- `other_2_rel_x`
- `other_2_rel_z`
- `other_2_dist`

의미:

- `sense_radius` 안의 다른 에이전트 중 가장 가까운 최대 3명의 상대 위치/거리
- 부족한 슬롯은 `0`으로 채워짐

### 20: 역할 ID

- 정규화된 `role_id`

### 21: 센서 실패 코드

- `sensor_fail_code`
- `0.0`: 반경 안에 주변 몹 있음
- `1.0`: 반경 안에 주변 몹 없음

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

현재는 `sensor_fail_code == 1.0`이면 단순 직선 추적이 아니라, geodesic map 기준 다음 waypoint를 따라갑니다.

즉:

- 평소: 정책 action 기반 이동
- `F=1`: goal 방향 직선 추적이 아니라, geodesic 하강 경로 기반 waypoint 이동

이 fallback은 정책이 주변 몹이 안 보이는 상황에서 장거리 추적을 직접 학습하지 않아도, 목표 쪽으로 우회 이동할 수 있게 하기 위한 장치입니다.

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

### 2. 목표 접근 보상

가능하면 geodesic 거리 감소량, 아니면 유클리드 거리 감소량을 씁니다.

```python
progress_coef * (old_dist - new_dist)
```

기본값:

- `progress_coef = 0.02`

주의:

- 이 값은 여전히 `goal_pos` 기준 거리 감소량입니다
- 하지만 이것만으로 성공은 아닙니다

### 3. 충돌 패널티

```python
-collision_penalty
```

기본값:

- `0.35`

### 4. Separation 패널티

다른 에이전트와 너무 가까우면 감점합니다.

### 5. Stall 패널티

진전이 없으면 감점합니다.

기본값:

- `stall_penalty = 0.05`
- `stall_patience = 20`

### 6. 역할 보상

역할별 shaping reward를 추가합니다.

- `front`: 정면 압박
- `flank_left`: 왼쪽 측면 점유
- `flank_right`: 오른쪽 측면 점유
- `cover`: 다른 몹 뒤 위치

### 7. 성공 보상

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
- `flank_left`: 왼쪽 측면 밴드 형성
- `flank_right`: 오른쪽 측면 밴드 형성
- `cover`: 다른 몹 뒤 엄폐 위치 형성

## 종료 조건

환경 종료는 팀 기준입니다.

```python
terminated = np.all(success_mask)
```

즉:

- 각 에이전트는 개별 전술 성공 상태를 가짐
- 전원이 역할 전술을 만족해야 episode 종료

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
- 목표 접근 보상
- 충돌/분리/stall 패널티

### 9. 역할 보상 및 역할 성공 판정

- 역할별 전술 평가
- 역할 shaping reward 추가
- `success_mask[idx]` 결정

### 10. 성공 보상 지급

- 전술 성공한 에이전트만 큰 보상 지급

### 11. 종료 조건 계산

- 전원이 성공하면 `terminated = True`

### 12. 다음 관측 반환

```python
obs, rewards, terminated, truncated, info
```

`info`에는 다음 같은 값이 들어갑니다.

- `agent_positions`
- `tactical_target`
- `role_targets`
- `role_ids`
- `success_mask`
- `sensor_fail_code`
- `reward_terms`

## 학습 성공률 의미

현재 `Model.py`에서 episode success는 느슨한 “한 명 성공” 기준이 아닙니다.

전술 다양성 조건:

- `front` 성공 1개 이상
- `flank_left` 또는 `flank_right` 중 하나 이상 성공
- `cover` 성공 1개 이상

즉:

- 정면 압박
- 측면 포위
- 엄폐

이 최소 조합이 만들어져야 episode success입니다.

학습 로그에는 역할별 성공률도 같이 표시됩니다.

예:

```text
role(front/fl/flr/cov)=...
```

의미:

- `front` 역할 성공률
- `flank_left` 역할 성공률
- `flank_right` 역할 성공률
- `cover` 역할 성공률

## 현재 코드 핵심 변경점

현재 코드 기준 핵심은 다음입니다.

- 장애물 ray 관측 제거
- 주변 몹 반경 탐지 도입
- `sensor_fail_code` 도입
- `F=1`일 때 geodesic waypoint fallback
- 모든 에이전트 이동 속도 동일
- goal 도달이 아니라 역할 전술 형성이 성공 기준
- 학습 성공률도 전술 다양성 기준으로 집계
