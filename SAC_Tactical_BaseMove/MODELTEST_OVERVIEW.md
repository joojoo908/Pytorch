## ModelTest

`ModelTest.py`는 `base_move` 전용 actor를 평가한다.

기본 시각화는 좌우 비교 방식이다.

- 왼쪽: `Detour Only`
- 오른쪽: `BaseMove Actor`

### 평가 기준

- episode 종료 시점에 `sense_radius` 안에 있는 agent 수를 주요 지표로 본다.
- 같은 초기 상태의 `Detour Only` 종료 수를 함께 계산한다.
- 시작 시 이미 `sense_radius` 안에 있던 agent 수도 함께 기록한다.
- 기존 `success / failed / timeout / blocked` outcome 출력도 유지한다.
- 충돌 횟수도 함께 기록한다.

### 기본 규칙

- 기본 `role_rule`: `base_move_only`
- 기본 `agent_role_rules`: `base_move_only`
- 기본 `agent_role_rule_pool`: 없음

### 체크포인트 형식

- actor 파일은 `base_move_actor` 형식을 사용한다.

### 로그

- `[Eval]` 줄에서 다음을 출력한다.
- 반환값 `return`
- `start_in_sense`: episode 시작 시 `sense_radius` 안에 있던 agent 수
- `in_sense_end`: actor의 episode 종료 시 `sense_radius` 안 agent 수
- `detour_end`: detour-only의 episode 종료 시 `sense_radius` 안 agent 수
- `vs_detour`: `actor / detour` 비율
- `collisions`

- `[Episode ...]` 줄에서 다음을 출력한다.
- episode 반환값
- outcome
- `start_in_sense`
- `in_sense_end`
- `detour_end`
- `vs_detour`
- `collisions`
- 실제 agent 수

- `[Summary]` 줄에서 다음을 출력한다.
- 전체 episode 수
- 기존 success count
- 전체 평가 구간 누적 `start_in_sense`
- 전체 평가 구간 누적 `in_sense_end`와 비율
- 전체 평가 구간 누적 `detour_end`
- 전체 평가 구간 누적 `vs_detour`
- 전체 평가 구간 누적 `collisions`
- 평균 반환값

를 출력한다.

### 시각화

- `Detour Only` 패널에는 episode 시작 시 detour 경로 프리뷰를 회색으로 표시한다.
- 현재 tick에 충돌 중인 agent는 붉은색으로 표시한다.
- goal에 도착한 agent는 충돌 처리와 충돌 횟수 집계에서 제외한다.
- actor 패널에는 현재 `vs_detour` 상대 성능도 함께 표시한다.
