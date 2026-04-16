#include "OnnxRunner.h"

#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

std::int64_t element_count(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        throw std::runtime_error("input_shape must not be empty.");
    }
    return std::accumulate(
        shape.begin(),
        shape.end(),
        std::int64_t{1},
        [](std::int64_t lhs, std::int64_t rhs) {
            if (rhs <= 0) {
                throw std::runtime_error("input_shape values must be positive.");
            }
            return lhs * rhs;
        });
}

std::vector<float> make_base_move_input() {
    constexpr int kObsDim = 24;
    std::vector<float> values(kObsDim, 0.0f);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : values) {
        value = dist(rng);
    }

    values.back() = 0.0f;  // sensor_fail_code: 0 means the role actor can be used.
    return values;
}

#ifdef HAS_ONNXRUNTIME
std::basic_string<ORTCHAR_T> to_ort_path(const std::string& path) {
#ifdef _WIN32
    return std::basic_string<ORTCHAR_T>(path.begin(), path.end());
#else
    return path;
#endif
}
#endif

}  // namespace

bool is_onnx_runtime_enabled() {
#ifdef HAS_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

OnnxRunResult run_onnx_float_model(
    const std::string& model_path,
    const std::vector<float>& input_values,
    const std::vector<std::int64_t>& input_shape) {
    const std::int64_t expected_count = element_count(input_shape);
    if (expected_count != static_cast<std::int64_t>(input_values.size())) {
        throw std::runtime_error("input_values size does not match input_shape element count.");
    }

#ifndef HAS_ONNXRUNTIME
    (void)model_path;
    throw std::runtime_error("ONNX Runtime is disabled. Set ONNXRUNTIME_DIR and rebuild.");
#else
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const auto ort_model_path = to_ort_path(model_path);
    Ort::Session session(env, ort_model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> mutable_input = input_values;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        mutable_input.data(),
        mutable_input.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);

    if (outputs.empty() || !outputs.front().IsTensor()) {
        throw std::runtime_error("ONNX output is empty or not a tensor.");
    }

    Ort::Value& output_tensor = outputs.front();
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<std::int64_t> output_shape = output_info.GetShape();
    const std::size_t output_count = output_info.GetElementCount();
    const float* output_data = output_tensor.GetTensorData<float>();
    std::vector<float> output_values(output_data, output_data + output_count);

    return OnnxRunResult{
        input_name.get(),
        output_name.get(),
        input_values,
        input_shape,
        std::move(output_values),
        std::move(output_shape),
    };
#endif
}

OnnxRunResult run_base_move_example(const std::string& model_path) {
    return run_onnx_float_model(
        model_path,
        make_base_move_input(),
        std::vector<std::int64_t>{1, 24});
}
