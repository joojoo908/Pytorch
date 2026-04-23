# ModelTest.py 설명

## 개요

`ModelTest.py`는 학습된 multi-role actor를 불러와 현재 환경에서 평가하고, pygame 기반 2D 탑다운 디버그 화면을 보여주는 스크립트입니다.

관련 파일:

- 평가 스크립트: `ModelTest.py`
- 환경: `majestro_navmesh_env.py`
- 모델 정의: `Model.py`

역할:

1. 저장된 actor 가중치로 여러 episode 평가
2. 역할별 이동과 fallback 경로를 시각적으로 확인
3. heuristic 조합 pool과 가변 agent 수 평가를 디버깅

## 실행 방법

기본 실행:

```bash
python ModelTest.py
```

예시:

```bash
python ModelTest.py --actor-path sac_actor_best.pth
python ModelTest.py --episodes 20 --no-visualize
python ModelTest.py --sense-radius 500
python ModelTest.py --agent-role-rules melee_dps,melee_dps,melee_dps,ranged_dps,ranged_dps
python ModelTest.py --agent-role-rule-pool "rdps,rdps,rdps;mdps,mdps,mdps,mdps;rdps,rdps,mdps,mdps,mdps"
```

## 주요 옵션

- `--actor-path`: 불러올 actor 파일
- `--episodes`: 평가 episode 수
- `--scale`: 월드 좌표를 화면에 투영할 배율
- `--no-visualize`: pygame 시각화 끄기
- `--move-step-size`
- `--tactical-target-radius`
- `--num-other-agents`
- `--observed-other-agents`
- `--agent-radius`
- `--sense-radius`
- `--goal-spawn-min-scale`
- `--agent-spawn-min-scale`
- `--agent-spawn-max-scale`
- `--role-rule`
- `--agent-role-rules`
- `--agent-role-rule-pool`

## heuristic 조합 pool

`ModelTest.py`는 학습과 같은 방식으로 episode마다 heuristic 조합을 랜덤 샘플링할 수 있습니다.

예:

```bash
--agent-role-rule-pool "rdps,rdps,rdps;mdps,mdps,mdps,mdps;mdps;rdps,rdps,mdps,mdps,mdps"
```

의미:

- 각 세미콜론 구간이 하나의 candidate 조합
- episode 시작 전에 하나를 랜덤 선택
- 선택된 조합 길이에 맞춰 실제 agent 수도 바뀜

즉 현재 평가는 1명, 3명, 4명, 5명처럼 가변 agent 수로도 돌 수 있습니다.

로그에는 다음 정보가 함께 출력됩니다.

```text
[ROLE-POOL] 5 sets | ...
[Episode 4/50] return=... outcome=success agents=3 rules=ranged_dps,ranged_dps,ranged_dps
```

## 화면 표현 방식

이 스크립트는 3D 렌더러가 아니라 2D 탑다운 디버그 뷰어입니다.

- 화면 가로축 = `x`
- 화면 세로축 = `z`
- 높이 `y`는 HUD 텍스트와 내부 계산에만 사용

## 화면에 그려지는 요소

### 1. Navmesh overlay

- walkable raster를 점으로 그림

### 2. 에이전트 실제 이동 궤적

- 각 agent의 실제 이동 궤적
- 색은 현재 role 기준

현재 색상:

- `front`: 빨강
- `cover`: 보라
- `base`: 연두
- `surround`: 분홍
- `kiting`: 주황
- `none`: 회색

### 3. 감지 반경

- 각 agent 중심에 `sense_radius` 원 표시

### 4. fallback 경로

- `sensor_fail_code == 1.0`인 agent만 표시
- Detour가 가능하면 Detour waypoint 기반, 아니면 geodesic fallback 경로 복원

### 5. tactical target

- 해당 step에서 실제로 향한 target 점

### 6. role target

- 역할 보상 계산에 쓰는 role target

### 7. 현재 위치

- agent 현재 위치
- 0번 agent는 흰색 외곽선으로 강조

### 8. 센서 실패 표시

- 각 agent 옆에 `F:0` 또는 `F:1`

### 9. goal

- 공통 goal 위치

### 10. HUD 텍스트

- 현재 step
- 누적 return
- 메인 agent와 goal 거리
- 메인 agent 위치
- 현재 role 목록

## episode 처리 순서

`run_multiple_evaluations(...)` 기준:

1. episode 시작 전에 role rule pool에서 조합 샘플링
2. 필요하면 env agent 수 재구성
3. `env.reset()`
4. actor로 action 계산
5. `env.step(action)`
6. 시각화 갱신
7. outcome 판정

## outcome 판정 기준

현재 `outcome`은 아래 순서로 결정됩니다.

1. 사용자가 창을 닫으면 `aborted`
2. `is_diverse_tactical_success(info)`가 참이면 `success`
3. `truncated`면 `timeout`
4. 어떤 agent라도 충돌이면 `blocked`
5. 그 외는 `failed`

여기서 `success`는 단순히 한 agent 성공이 아니라, 현재 `Model.py`와 같은 전술 다양성 성공 판정을 사용합니다.

현재 기준:

- `front` 성공이 하나 이상 있고
- `cover` 성공이 하나 이상 있고
- `surround` 성공이 하나 이상 있으면

`success`

## CSV 저장

마지막 episode는 `last_eval_traj.csv`를 저장합니다.

현재 저장 대상:

- 메인 agent(0번)의 실제 궤적

화면에는 전체 agent가 보이지만, CSV는 메인 agent만 기록합니다.

## 이 파일을 보는 목적

`ModelTest.py`는 주로 아래를 확인하기 위한 도구입니다.

- role별 이동 패턴이 분화되는지
- `front / cover / surround / kiting / base_move`가 의도대로 형성되는지
- fallback 이동이 자연스러운지
- 가변 agent 수 조합에서도 actor가 깨지지 않는지
- 어떤 heuristic 조합에서 실패하는지

## 해석 팁

1. 로그에서 `agents=N rules=...`를 먼저 본다
2. role 색 궤적이 조합별로 어떻게 달라지는지 본다
3. `F:1`인 agent의 fallback 경로를 확인한다
4. role target 대비 실제 위치가 얼마나 맞는지 본다
5. `success`, `timeout`, `blocked`, `failed`를 조합별로 비교한다

## 정리

현재 `ModelTest.py`는:

- 학습과 같은 role pool 샘플링을 지원하고
- episode마다 실제 agent 수가 달라질 수 있으며
- 2D 탑다운 뷰로 전술 이동과 fallback을 시각화하고
- 현재 코드 기준 success 판정을 그대로 사용해 결과를 출력하는

평가 및 디버그 도구입니다.
