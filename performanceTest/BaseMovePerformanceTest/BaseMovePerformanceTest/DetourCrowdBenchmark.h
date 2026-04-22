#pragma once

#include <string>

#include "BenchmarkTypes.h"

struct DetourCrowdBenchmarkOptions {
    std::string navmesh_path;
    int agents = 32;
    int iterations = 10000;
    int warmup = 1000;
    float dt = 1.0f / 60.0f;
};

bool is_detour_crowd_enabled();

Metrics benchmark_detour_crowd_avoidance(const DetourCrowdBenchmarkOptions& options);
