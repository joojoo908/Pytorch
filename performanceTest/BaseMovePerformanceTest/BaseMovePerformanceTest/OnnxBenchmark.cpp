#include "OnnxBenchmark.h"

#include <array>
#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> make_observation_batch(int obs_dim) {
    std::vector<float> obs(static_cast<std::size_t>(obs_dim), 0.0f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : obs) {
        v = dist(rng);
    }
    if (!obs.empty()) {
        obs.back() = 0.0f;  // sensor_fail_code: 0 means role actor is allowed to run.
    }
    return obs;
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

Metrics benchmark_onnx_base_move(
    const std::string& onnx_path,
    int obs_dim,
    int iterations,
    int warmup) {
#ifndef HAS_ONNXRUNTIME
    (void)onnx_path;
    (void)obs_dim;
    (void)iterations;
    (void)warmup;
    throw std::runtime_error("ONNX Runtime is disabled. Define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR.");
#else
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "base_move_perf");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const auto model_path = to_ort_path(onnx_path);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    std::vector<float> obs = make_observation_batch(obs_dim);
    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(obs_dim)};
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto run_once = [&]() {
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info,
            obs.data(),
            obs.size(),
            input_shape.data(),
            input_shape.size());
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input,
            1,
            output_names,
            1);
        volatile float sink = outputs.front().GetTensorMutableData<float>()[0];
        (void)sink;
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    const auto total_start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = Clock::now();
        run_once();
        const auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const auto total_end = Clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return compute_metrics(std::move(samples), total_ms);
#endif
}
