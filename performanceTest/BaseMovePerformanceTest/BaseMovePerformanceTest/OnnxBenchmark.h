#pragma once

#include <string>

#include "BenchmarkTypes.h"

bool is_onnx_runtime_enabled();

Metrics benchmark_onnx_base_move(
    const std::string& onnx_path,
    int obs_dim,
    int iterations,
    int warmup);
