# BaseMovePerformanceTest

`base_move.onnx` 추론 시간과 Detour 계열 이동 비용을 비교하는 콘솔 벤치마크입니다.

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
  --crowd-dt 0.0166667
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
- 각 항목은 `total_ms`, `avg_us`, `p50_us`, `p95_us`, `p99_us`를 출력합니다.
