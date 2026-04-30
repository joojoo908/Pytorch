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

`base_move` 성공은 다음을 동시에 만족할 때다.

- 이전 step에서 goal이 `sense_radius` 밖
- 현재 step에서 goal이 `sense_radius` 안
- 충돌 없음
- 실제 이동 속도가 `step_size * 0.45` 이상

### success buffer

- 일반 replay buffer에는 `base_move` step이 계속 쌓인다.
- success replay buffer에는 성공 순간만이 아니라, 성공 직전 최근 `base_move` 구간도 함께 적재된다.
- 기본 최근 구간 길이는 `8 step`이다.

### 기본 실행

- 학습: `python Test.py`
- 평가: `python ModelTest.py`

기본 인자는 `base_move_only` 단일 규칙을 사용하도록 바뀌어 있다.
