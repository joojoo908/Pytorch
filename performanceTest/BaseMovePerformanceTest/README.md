# BaseMovePerformanceTest

`base_move.onnx` 추론 시간과 Detour 계열 이동 비용을 비교하고, `base_move`와 `detour_crowd`가 공용 goal 반경까지 충돌 없이 진입하는 능력을 같이 확인하는 콘솔 벤치마크입니다.

## 포함된 실행 파일

필요한 런타임/데이터 파일을 프로젝트 안으로 복사했습니다.

```text
BaseMovePerformanceTest/
  BaseMovePerformanceTest/
    onnxruntime/
      include/onnxruntime_cxx_api.h
      lib/onnxruntime.lib
      lib/onnxruntime.dll
    native/
      Detour.lib
      Detour-d.lib
      Detour*.h
      DetourCrowd*.h
      DetourCrowd*.cpp
      DetourLocalBoundary.*
      DetourObstacleAvoidance.*
      DetourPathCorridor.*
      DetourPathQueue.*
      DetourProximityGrid.*
      detour_navmesh_wrapper.cpp
      detour_navmesh_wrapper.h
    onnx_roles/base_move.onnx
    Resources/NavMesh/all_tiles_navmesh.bin
```

`x64/Debug` 출력 폴더에도 `onnxruntime.dll`, `onnx_roles/base_move.onnx`, `Resources/NavMesh/all_tiles_navmesh.bin`을 복사해두었습니다.

## 빌드 전제

- Detour는 프로젝트 내부 `native` 폴더의 `Detour.lib` / `Detour-d.lib`와 wrapper를 사용합니다.
- DetourCrowd는 프로젝트 내부 `native` 폴더의 소스 파일을 직접 컴파일합니다.
- ONNX Runtime은 프로젝트 내부 `onnxruntime` 폴더를 자동으로 사용합니다.
- 다른 ONNX Runtime을 쓰고 싶을 때만 Windows 환경 변수 `ONNXRUNTIME_DIR`를 지정하면 됩니다.

예:

```bat
set ONNXRUNTIME_DIR=C:\onnxruntime-win-x64-1.18.1
```

로컬 `onnxruntime` 폴더 또는 `ONNXRUNTIME_DIR`가 있으면 프로젝트가 자동으로 `HAS_ONNXRUNTIME`를 정의하고 `onnxruntime.lib`를 링크합니다. 빌드 후 `onnxruntime.dll`은 exe 출력 폴더로 자동 복사됩니다.

## 실행 예시

```bat
BaseMovePerformanceTest.exe ^
  --onnx onnx_roles\base_move.onnx ^
  --navmesh Resources\NavMesh\all_tiles_navmesh.bin ^
  --obs-dim 24 ^
  --iterations 10000 ^
  --warmup 1000 ^
  --crowd-agents 32 ^
  --crowd-dt 0.0166667 ^
  --arrival-runs 10 ^
  --arrival-steps 128 ^
  --base-move-resolve off ^
  --goal-radius 120
```

프로젝트 디렉터리를 working directory로 실행하면 인자 없이도 기본 경로를 사용합니다.

```bat
BaseMovePerformanceTest.exe
```

Detour 시작/목표점은 Detour 좌표계 기준입니다. 필요하면 직접 지정합니다.

```bat
BaseMovePerformanceTest.exe --detour-start 0 0 0 --detour-goal 10 0 10
```

## 출력 의미

- `onnx_base_move`: `base_move.onnx` actor 1회 추론 비용입니다.
- `onnx_pipeline`: `--crowd-agents` 수만큼 grid 기반 주변 후보 탐색, `obs[24]` 구성, ONNX batch 추론까지 포함한 전체 비용입니다.
- `detour_crowd`: `dtCrowd::update()` 기반 로컬 충돌 회피 비용입니다.
- `detour_query`: `DetourNavMeshQuery` 기반 `find_next_waypoint` 호출 비용입니다. 이 값은 경로/waypoint query 비용이며 crowd 충돌 회피 비용이 아닙니다.
- `base_move_arrival`: `base_move` 규칙으로 다수 agent를 이동시켜 공용 goal의 `sense_radius` 안에 들어온 수, 충돌 없이 진입한 수, 충돌 수, 최종 평균 거리를 출력합니다.
- `detour_arrival`: 같은 시작 배치와 같은 공용 goal을 `dtCrowd`에 넣었을 때의 `sense_radius` 진입/충돌 지표입니다.
- 각 항목은 `total_ms`, `avg_us`, `p50_us`, `p95_us`, `p99_us`를 출력합니다.

### Arrival comparison 출력 형식

`arrival-runs`를 2 이상으로 주면 run별 결과와 평균/최선/최악 비교가 함께 출력됩니다.

예:

```text
run_0
  base_move         arrived=10/20 collision_free=1/20 collided_agents=18 collision_events=320 avg_final_dist=120.5 max_final_dist=410.2 steps=128
  detour            arrived=12/20 collision_free=3/20 collided_agents=16 collision_events=250 avg_final_dist=98.3 max_final_dist=300.1 steps=120

run_1
  base_move         ...
  detour            ...

base_move_arrival   runs=10 avg_arrived=9.80/20 avg_collision_free=1.20/20 avg_collided=17.40 avg_events=301.50 avg_final_dist=130.200 avg_steps=126.40
  best: arrived=13/20 collision_free=3/20 collision_events=210 avg_final_dist=90.500 steps=118
  worst: arrived=6/20 collision_free=0/20 collision_events=420 avg_final_dist=180.300 steps=128

detour_arrival      runs=10 avg_arrived=11.30/20 avg_collision_free=2.60/20 avg_collided=15.80 avg_events=240.10 avg_final_dist=110.900 avg_steps=121.80
  best: arrived=15/20 collision_free=5/20 collision_events=170 avg_final_dist=70.400 steps=110
  worst: arrived=8/20 collision_free=1/20 collision_events=360 avg_final_dist=160.700 steps=128

best_compare
  base_move arrived=13/20 collision_free=3/20 collision_events=210 avg_final_dist=90.500 steps=118
  detour    arrived=15/20 collision_free=5/20 collision_events=170 avg_final_dist=70.400 steps=110

worst_compare
  base_move arrived=6/20 collision_free=0/20 collision_events=420 avg_final_dist=180.300 steps=128
  detour    arrived=8/20 collision_free=1/20 collision_events=360 avg_final_dist=160.700 steps=128
```

### Arrival comparison 항목 해석

- `run_0`, `run_1`, ...: 같은 번호 안의 `base_move`와 `detour`는 동일한 spawn 배치, 동일한 공용 goal, 동일한 파라미터에서 비교한 결과입니다.
- `arrived=a/N`: 공용 goal 중심에서 `sense_radius` 안으로 진입한 agent 수입니다.
- `collision_free=b/N`: 충돌 없이 `sense_radius` 안으로 진입한 agent 수입니다.
- `collided_agents=c`: 이동 중 한 번 이상 충돌 판정을 받은 agent 수입니다.
- `collision_events=d`: 전체 step 동안 누적된 충돌 이벤트 수입니다.
- `avg_final_dist=x`: 종료 시점에 공용 goal 중심까지 남은 평균 거리입니다.
- `max_final_dist=y`: 종료 시점에 가장 멀리 남은 agent의 거리입니다.
- `steps=z`: 해당 run이 실제로 수행한 step 수입니다. 전원이 일찍 성공하면 최대 step보다 작을 수 있습니다.

### 평균 / best / worst 해석

- `base_move_arrival`, `detour_arrival`: 모든 run의 평균 요약입니다.
- `avg_arrived`: 평균적으로 몇 명이 `sense_radius` 안에 들어왔는지 보여줍니다.
- `avg_collision_free`: 평균적으로 몇 명이 충돌 없이 성공했는지 보여줍니다.
- `avg_collided`: 평균 충돌 agent 수입니다.
- `avg_events`: 평균 충돌 이벤트 수입니다.
- `avg_final_dist`: 평균 최종 거리입니다.
- `avg_steps`: 평균 step 수입니다.
- `best`: 각 방식에서 가장 좋았던 run입니다. 현재는 `arrived`가 큰 run을 우선하고, 동률이면 `collision_events`가 작은 run을 고릅니다.
- `worst`: 각 방식에서 가장 나빴던 run입니다. 현재는 `arrived`가 작은 run을 우선하고, 동률이면 `collision_events`가 큰 run을 고릅니다.
- `best_compare`: 두 방식의 best run을 나란히 보여줍니다.
- `worst_compare`: 두 방식의 worst run을 나란히 보여줍니다.

도착 비교에 쓰는 주요 옵션:

- `--arrival-steps <n>`: 최대 시뮬레이션 step 수
- `--arrival-runs <n>`: 다른 spawn 배치로 반복 실행할 횟수
- `--goal-radius <r>`: 기존 호환용 반경 옵션
- `--step-size <v>`: `base_move` 한 step 이동량
- `--tactical-radius <v>`: `base_move` actor 출력이 가리키는 목표 반경
- `--sense-radius <v>`: 주변 agent 관측 반경이자 arrival success 판정 반경
- `--agent-radius <v>`: agent 충돌 반경
- `--base-move-resolve <on|off>`: `base_move`에서 충돌 시 이분 탐색으로 이동량을 줄일지 여부
- `--detour-goal <x> <y> <z>`: arrival comparison에서 모든 agent가 공통으로 진입하려는 goal 중심

`arrival-runs`를 2 이상으로 주면 각 run 결과를 모두 출력하고, 마지막에 평균(`avg_*`)과 최선/최악 run을 같이 요약합니다.
`--base-move-resolve off`를 주면 `base_move`는 충돌이 나도 이동량을 다시 줄여 안전 지점을 찾지 않습니다. 이 경우 ONNX가 제안한 이동 결과가 더 직접적으로 반영되지만 충돌 이벤트 수가 크게 늘 수 있습니다.
Arrival comparison의 거리 파라미터(`sense-radius`, `step-size`, `tactical-radius`, `agent-radius`, `goal-radius`)는 `majestro_navmesh_env.py`와 동일한 1:100 엔진 좌표계 기준으로 입력하고, 내부에서는 Detour 좌표계로 1/100 변환해 사용합니다. `detour-goal`은 이름 그대로 Detour 좌표계 기준입니다.
