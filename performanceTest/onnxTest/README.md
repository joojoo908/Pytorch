# onnxTest

ONNX Runtime C++ API로 `base_move.onnx`를 실행하는 최소 예제입니다.

## 핵심 함수

`OnnxRunner.h`의 함수가 실제 사용 지점입니다.

```cpp
OnnxRunResult run_onnx_float_model(
    const std::string& model_path,
    const std::vector<float>& input_values,
    const std::vector<std::int64_t>& input_shape);
```

입력은 `float` 벡터와 shape입니다. 출력은 `OnnxRunResult`에 담깁니다.

```cpp
struct OnnxRunResult {
    std::string input_name;
    std::string output_name;
    std::vector<float> input_values;
    std::vector<std::int64_t> input_shape;
    std::vector<float> output_values;
    std::vector<std::int64_t> output_shape;
};
```

`base_move.onnx` 전용 예제는 다음 함수입니다.

```cpp
OnnxRunResult run_base_move_example(const std::string& model_path);
```

이 함수는 `float[1, 24]` 더미 입력을 만들고, actor 출력인 `float[1, 2]` 액션 값을 반환합니다.

## 빌드 설정

프로젝트 폴더 안에 `onnxruntime` 폴더가 있으면 자동으로 사용합니다.

현재 기대 구조:

```text
onnxTest/
  onnxTest/
    onnxruntime/
      include/onnxruntime_cxx_api.h
      lib/onnxruntime.lib
      lib/onnxruntime.dll
```

이 구조가 있으면 별도 환경 변수 없이 빌드할 수 있습니다. 빌드 후 `onnxruntime.dll`은 exe 출력 폴더로 자동 복사됩니다.

다른 위치의 ONNX Runtime C++ SDK를 쓰고 싶으면 Windows 환경 변수로 지정합니다.

```bat
set ONNXRUNTIME_DIR=C:\onnxruntime-win-x64-1.18.1
```

프로젝트는 로컬 `onnxruntime` 폴더 또는 `ONNXRUNTIME_DIR`가 있을 때 자동으로 다음을 설정합니다.

- include: `$(ONNXRUNTIME_DIR)\include`
- lib: `$(ONNXRUNTIME_DIR)\lib`
- dependency: `onnxruntime.lib`
- macro: `HAS_ONNXRUNTIME`

실행 시에는 `onnxruntime.dll`이 exe 옆에 있어야 합니다. 현재 프로젝트는 빌드 후 자동 복사합니다.

## 실행

기본 모델 경로는 `..\..\..\SAC_PathFind_3d\onnx_roles\base_move.onnx`입니다.

```bat
onnxTest.exe
```

다른 모델을 지정하려면:

```bat
onnxTest.exe --model C:\Cooding\Pytorch\SAC_PathFind_3d\onnx_roles\base_move.onnx
```

콘솔에는 입력 이름, 입력 shape, 입력 값, 출력 이름, 출력 shape, 출력 값이 출력됩니다.
