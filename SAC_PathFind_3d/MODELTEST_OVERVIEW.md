# ModelTest.py 설명

## 개요

`ModelTest.py`는 학습된 actor를 불러와 현재 환경에서 평가하고, pygame으로 2D 탑다운 디버그 시각화를 보여주는 스크립트입니다.

역할:

1. 저장된 actor 가중치로 여러 에피소드 평가
2. 멀티에이전트 이동 결과를 시각적으로 확인
3. 역할별 전술 위치 형성, fallback 이동, 감지 반경 동작을 디버깅

관련 파일:

- 평가 스크립트: `ModelTest.py`
- 환경: `majestro_navmesh_env.py`
- 모델 정의: `Model.py`

## 실행 방법

기본 실행:

```bash
python ModelTest.py
```

자주 쓰는 옵션 예시:

```bash
python ModelTest.py --actor-path sac_actor_best.pth
python ModelTest.py --episodes 20
python ModelTest.py --sense-radius 500
python ModelTest.py --goal-spawn-min-scale 2.0 --agent-spawn-min-scale 1.2
python ModelTest.py --no-visualize
```

주요 옵션:

- `--actor-path`: 불러올 actor 가중치 경로
- `--episodes`: 평가 에피소드 수
- `--scale`: 월드 좌표를 화면에 투영할 배율
- `--no-visualize`: pygame 시각화 비활성화
- `--move-step-size`: 모든 에이전트 공통 한 step 최대 이동 거리
- `--tactical-target-radius`: action을 tactical target으로 바꿀 때 쓰는 반경
- `--num-other-agents`: 추가 에이전트 수
- `--observed-other-agents`: 관측에 넣을 주변 몹 최대 수
- `--agent-radius`: 에이전트 반경
- `--sense-radius`: 주변 몹 탐지 반경
- `--goal-spawn-min-scale`: 시작점과 goal의 최소 거리 배수
- `--agent-spawn-min-scale`: 초기 스폰 간 최소 거리 배수

## 이 스크립트가 하는 일

전체 흐름:

1. env 생성
2. actor weight 로드
3. `evaluate_once(...)`로 한 에피소드 평가
4. 에피소드별 reward와 outcome 출력
5. 마지막에는 전체 성공 개수와 평균 return 요약 출력

주요 함수:

- `policy_act(...)`
- `evaluate_once(...)`
- `run_multiple_evaluations(...)`
- `recover_descent_path_world(...)`

## 화면은 3D 렌더링이 아니라 2D 탑다운 뷰

`ModelTest.py`는 실제 3D 메시를 렌더링하지 않습니다.

대신 navmesh 월드의 `(x, z)` 평면을 화면에 투영해서 위에서 내려다보는 디버그 화면을 그립니다.

즉:

- 화면 가로 = `x`
- 화면 세로 = `z`

높이 `y`는 화면 좌표 계산에 직접 쓰지 않고, HUD 텍스트와 환경 내부 계산에만 사용됩니다.

좌표 변환은 `make_world_to_screen(...)`에서 처리합니다.

## 화면에 그려지는 요소

### 1. Navmesh Overlay

- walkable grid를 어두운 점으로 그림
- 현재 월드에서 이동 가능한 영역을 확인하기 위한 레이어

### 2. 역할별 실제 이동 궤적

- 모든 에이전트의 실제 이동 궤적을 선으로 그림
- 색은 에이전트 ID가 아니라 역할 기준으로 고정

즉:

- 같은 역할이면 서로 같은 색 궤적을 그림
- 개체 번호보다 역할군 움직임 비교에 초점이 있음

### 3. 감지 반경

- 각 에이전트 중심에 `sense_radius` 원을 그림
- 역할 색과 같은 색을 사용

의미:

- 현재 로컬 탐지가 적용되는 월드 반경

### 4. Fallback 경로

- `sensor_fail_code == 1.0`인 에이전트만 geodesic descent path를 선으로 그림
- 항상 표시되는 힌트 경로가 아니라, `F=1`일 때만 표시되는 fallback 경로

의미:

- 주변 몹을 감지하지 못한 에이전트가 fallback으로 따라가려는 근사 경로

주의:

- 이 경로는 Detour의 실제 polygon path가 아니라
- 현재 env의 geodesic map을 따라 복원한 근사 경로입니다

### 5. Tactical Target

- 각 step에서 실제로 향하려 한 tactical target을 원으로 표시
- 첫 번째 에이전트는 조금 더 크게, 나머지는 작은 점으로 표시

### 6. Role Target

- 역할 보상 계산에 쓰는 역할 기준 목표 위치를 작은 원으로 표시
- 역할 색을 따름

### 7. 현재 에이전트 위치

- 모든 에이전트를 역할 색 점으로 그림
- 메인 에이전트는 흰색 외곽선으로 한 번 더 표시

### 8. 센서 실패 코드 표시

- 각 에이전트 점 옆에 `F:0` 또는 `F:1` 텍스트 표시

의미:

- `F:0`: `sense_radius` 안에 주변 몹이 있음
- `F:1`: `sense_radius` 안에 주변 몹이 없음, fallback 이동 조건

### 9. Goal

- 공통 목표 위치를 빨간 점으로 그림

### 10. HUD 텍스트

좌측 상단에 다음 정보가 표시됩니다.

- 현재 step
- 누적 return
- 메인 에이전트와 goal의 유클리드 거리
- 메인 에이전트의 `(x, y, z)` 위치
- 현재 각 에이전트 역할 목록

## 역할별 색상

현재 역할 색상:

- `front`: `(255, 110, 110)`
- `flank_l`: `(110, 220, 255)`
- `flank_r`: `(255, 210, 90)`
- `cover`: `(180, 140, 255)`

화면상 의미:

- `front`: 빨강
- `flank_l`: 하늘색
- `flank_r`: 노랑
- `cover`: 보라

이 색은 다음 요소들에 공통으로 사용됩니다.

- 실제 이동 궤적
- 감지 반경 원
- role target
- 현재 에이전트 위치
- `F=1` fallback 경로

## 한 에피소드 처리 순서

`evaluate_once(...)` 기준 한 턴은 대략 다음 순서입니다.

1. `env.reset()`으로 초기 상태 생성
2. actor를 `eval()` 모드로 둠
3. 모든 에이전트의 초기 위치를 궤적 버퍼에 저장
4. 매 step마다 actor가 `(num_agents, obs_dim)` 관측을 받아 `(num_agents, 2)` action 계산
5. `env.step(action)` 호출
6. env가 각 에이전트의 실제 이동과 보상을 계산
7. 반환된 `final_info`에서 `role_ids`, `agent_positions`, `sensor_fail_code`, `tactical_target`, `role_targets`를 읽음
8. 모든 에이전트의 실제 위치를 궤적 버퍼에 추가
9. pygame 화면에 navmesh, 궤적, 감지 반경, fallback 경로, 현재 위치를 다시 그림
10. `terminated` 또는 `truncated`이면 에피소드 종료

## 평가 결과 로그 의미

에피소드 종료 시:

```text
[Eval] success | return=...
```

또는

```text
[Eval] timeout | return=...
```

같은 형식으로 출력됩니다.

여러 에피소드 평가 시:

```text
[Episode i/N] return=... outcome=...
```

마지막에는:

```text
[Summary] episodes=... success=... (...) avg_return=...
```

형태의 요약이 출력됩니다.

## `outcome` 판정 기준

현재 `ModelTest.py`의 `outcome`은 아래 순서로 판정합니다.

1. `reward_terms` 안에 `success` 보상이 하나라도 있으면 `success`
2. `truncated`면 `timeout`
3. `collided`가 하나라도 있으면 `blocked`
4. 그 외는 `failed`

주의:

- 이 평가는 현재 `Model.py` 학습 로그의 전술 다양성 성공률과 완전히 같은 기준이 아닙니다
- `ModelTest.py`는 아직 “적어도 한 에이전트가 success reward를 받았는가”에 더 가까운 outcome 판정을 씁니다

즉:

- 학습 로그의 success rate
- `ModelTest.py`의 outcome

은 같은 의미로 읽으면 안 됩니다.

## CSV 저장

마지막 평가 에피소드에서는 `last_eval_traj.csv`가 저장됩니다.

현재 저장 대상:

- 메인 에이전트(0번)의 실제 궤적만 저장

즉:

- 화면에는 전체 에이전트 궤적이 표시
- CSV에는 메인 에이전트 경로만 기록

## 이 파일을 보는 목적

`ModelTest.py`는 아래 내용을 확인하기 위한 디버그 도구입니다.

- 역할별 이동 패턴이 실제로 분화되는지
- `front / flank / cover` 전술 위치가 만들어지는지
- `sense_radius`가 너무 크거나 작지 않은지
- `F=1`일 때 fallback 경로가 어디로 잡히는지
- 어떤 에이전트가 왜 멈추는지
- 군집 충돌, 병목, 우회 동작이 자연스러운지

## 해석 팁

화면을 볼 때는 보통 이렇게 보면 됩니다.

1. 역할 색 궤적이 서로 다른 패턴으로 갈라지는지 본다
2. 감지 반경 원과 `F:0/F:1`를 함께 본다
3. `F=1`인데 fallback 경로가 막힌 지형 쪽인지 본다
4. tactical target과 실제 궤적 차이를 본다
5. role target 대비 실제 위치가 얼마나 역할답게 형성되는지 본다

## 정리

현재 `ModelTest.py`는:

- 3D 렌더러가 아니라 2D 탑다운 디버그 뷰어
- 모든 에이전트의 실제 이동 궤적을 역할 색으로 그림
- 감지 반경과 `F:0/F:1` 상태를 함께 보여줌
- `F=1`인 에이전트에 대해서만 fallback 경로를 추가로 그림
- goal, tactical target, role target, 현재 위치를 함께 표시

즉 단순 성공 여부만 확인하는 스크립트가 아니라, 역할별 전술 이동과 fallback 동작을 시각적으로 검증하는 평가 도구입니다.
