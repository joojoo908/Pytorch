## SAC_Tactical_BaseMove

이 프로젝트는 `SAC_PathFind_3d`를 복사한 뒤 `base_move` 전용 학습용으로 정리한 버전이다.

### 목적

- 반경 밖에서 goal 쪽으로 접근
- navmesh 경로를 따라 이동
- 다른 agent와 충돌을 피하면서 `sense_radius` 안으로 진입

### 역할 구조

- 학습과 평가는 `base_move` 단일 role만 사용한다.
- env 내부에는 기존 role 상수가 남아 있지만, 기본 heuristic은 항상 `base_move_only`를 사용한다.
- 모델 bundle도 `base_move` 하나만 생성하고 저장한다.

### 성공 조건

`base_move` 성공은 다음을 만족할 때다.

- 이전 step에서 goal이 `sense_radius` 밖
- 현재 step에서 goal이 `sense_radius` 안

즉 성공은 상태 유지가 아니라 `밖 -> 안` 진입 이벤트다.

### 현재 보상 구조

- 매 step 기본 시간 패널티 `-time_penalty`
- 충돌 시 `-collision_penalty`
- 정체가 길어지면 `-stall_penalty`
- role target 쪽으로 가까워지면 `role_progress` 보상
- geodesic/path distance가 줄어들면 `path_progress` 보상
- `sense_radius` 안으로 처음 진입하면 작은 진입 보너스
- detour waypoint를 사용하면 작은 보너스
- 이동 속도에 따른 작은 속도 보너스
- 성공 진입 이벤트가 발생한 step에는 큰 성공 보상 `+20`
- `sense_radius` 안에 있던 agent가 밖으로 나가면 강한 이탈 패널티 `-12`

초기부터 `sense_radius` 안에 있던 agent는 성공으로 세지지 않는다.
다만 이후 밖으로 나갔다가 다시 안으로 들어오면 다시 성공 이벤트가 발생할 수 있다.

### success buffer

- 일반 replay buffer에는 `base_move` step이 계속 쌓인다.
- success replay buffer에는 성공 순간만이 아니라, 성공 직전 최근 `base_move` 구간도 함께 적재된다.
- 기본 최근 구간 길이는 `8 step`이다.

### 학습 로그 기준

학습 로그는 10 episode마다 다음 항목을 출력한다.

- `start_in_sense`: episode 시작 시 이미 `sense_radius` 안에 있던 agent 수
- `in_sense_end`: episode 종료 시 `sense_radius` 안에 남아 있던 agent 수와 비율
- `recent100`: 최근 100 episode 누적 기준 종료 시 `sense_radius` 안에 있던 agent 비율
- `succ_buf`: success replay buffer에 쌓인 sample 수
- `growth@N`: 최근 기록 창 기준 success buffer 증가량
- `alpha`: 현재 SAC entropy coefficient

### 기본 실행

- 학습: `python Test.py`
- 평가: `python ModelTest.py`

기본 인자는 `base_move_only` 단일 규칙을 사용하도록 바뀌어 있다.
