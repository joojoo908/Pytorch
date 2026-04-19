#pragma once

#include <string>

#include "BenchmarkTypes.h"

struct OnnxPipelineBenchmarkOptions {
    std::string onnx_path;
    std::string navmesh_path;
    int agents = 32;
    int obs_dim = 24;
    int iterations = 10000;
    int warmup = 1000;
    float agent_radius = 30.0f;
    float sense_radius = 600.0f;
};

bool is_onnx_runtime_enabled();

Metrics benchmark_onnx_base_move(
    const std::string& onnx_path,
    int obs_dim,
    int iterations,
    int warmup);

Metrics benchmark_onnx_pipeline(const OnnxPipelineBenchmarkOptions& options);
