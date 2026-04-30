## ModelTest

`ModelTest.py`는 `base_move` 전용 actor를 평가한다.

### 평가 기준

- episode 성공 여부는 `base_move` 성공이 한 번이라도 발생했는지로 본다.
- 다중 전술 조합 성공 판정은 사용하지 않는다.

### 기본 규칙

- 기본 `role_rule`: `base_move_only`
- 기본 `agent_role_rules`: `base_move_only`
- 기본 `agent_role_rule_pool`: 없음

### 체크포인트 형식

- actor 파일은 `base_move_actor` 형식을 사용한다.

### 로그

- episode별 반환값
- 성공 여부
- 실제 agent 수
- 적용된 규칙 목록

를 출력한다.
