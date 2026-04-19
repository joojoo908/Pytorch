#include "OnnxBenchmark.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "DetourNavMesh.h"
#include "DetourProximityGrid.h"
#include "detour_navmesh_wrapper.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Vec2 {
    float x = 0.0f;
    float z = 0.0f;
};

struct Bounds {
    float bmin[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    float bmax[3] = {
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
    };
};

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

Bounds compute_navmesh_bounds(const dtNavMesh* navmesh) {
    Bounds bounds;
    bool found = false;
    const int max_tiles = navmesh->getMaxTiles();
    for (int i = 0; i < max_tiles; ++i) {
        const dtMeshTile* tile = navmesh->getTile(i);
        if (tile == nullptr || tile->header == nullptr) {
            continue;
        }
        for (int axis = 0; axis < 3; ++axis) {
            bounds.bmin[axis] = std::min(bounds.bmin[axis], tile->header->bmin[axis]);
            bounds.bmax[axis] = std::max(bounds.bmax[axis], tile->header->bmax[axis]);
        }
        found = true;
    }
    if (!found) {
        throw std::runtime_error("Navmesh has no loaded tiles.");
    }
    return bounds;
}

std::vector<Vec2> make_agent_positions(const Bounds& bounds, int agents) {
    std::vector<Vec2> positions;
    positions.reserve(static_cast<std::size_t>(agents));

    const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<float>(agents)))));
    const float min_x = bounds.bmin[0];
    const float max_x = bounds.bmax[0];
    const float min_z = bounds.bmin[2];
    const float max_z = bounds.bmax[2];
    const float step_x = (max_x - min_x) / static_cast<float>(std::max(1, grid - 1));
    const float step_z = (max_z - min_z) / static_cast<float>(std::max(1, grid - 1));

    for (int z = 0; z < grid && static_cast<int>(positions.size()) < agents; ++z) {
        for (int x = 0; x < grid && static_cast<int>(positions.size()) < agents; ++x) {
            positions.push_back(Vec2{
                min_x + step_x * static_cast<float>(x),
                min_z + step_z * static_cast<float>(z),
            });
        }
    }
    return positions;
}

void add_agent_to_grid(dtProximityGrid& grid, int id, const Vec2& pos, float radius) {
    grid.addItem(
        static_cast<unsigned short>(id),
        pos.x - radius,
        pos.z - radius,
        pos.x + radius,
        pos.z + radius);
}

void build_pipeline_observations(
    const OnnxPipelineBenchmarkOptions& options,
    const std::vector<Vec2>& positions,
    const std::vector<Vec2>& velocities,
    const Vec2& goal,
    const Bounds& bounds,
    dtProximityGrid& grid,
    std::vector<float>& observations) {
    const int agents = static_cast<int>(positions.size());
    const int obs_dim = options.obs_dim;
    constexpr int kObservedOthers = 3;
    constexpr int kCandidateLimit = 64;
    unsigned short ids[kCandidateLimit];

    std::fill(observations.begin(), observations.end(), 0.0f);
    grid.clear();
    for (int i = 0; i < agents; ++i) {
        add_agent_to_grid(grid, i, positions[i], options.agent_radius);
    }

    const float center_x = 0.5f * (bounds.bmin[0] + bounds.bmax[0]);
    const float center_z = 0.5f * (bounds.bmin[2] + bounds.bmax[2]);
    const float scale = std::max({bounds.bmax[0] - bounds.bmin[0], bounds.bmax[2] - bounds.bmin[2], 1.0f});
    const float step_size = 120.0f;

    for (int i = 0; i < agents; ++i) {
        float* obs = observations.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(obs_dim);
        const Vec2& pos = positions[i];
        const Vec2& vel = velocities[i];

        obs[0] = (pos.x - center_x) / scale;
        obs[1] = 0.0f;
        obs[2] = (pos.z - center_z) / scale;

        obs[3] = (goal.x - center_x) / scale;
        obs[4] = 0.0f;
        obs[5] = (goal.z - center_z) / scale;

        obs[6] = (goal.x - pos.x) / scale;
        obs[7] = 0.0f;
        obs[8] = (goal.z - pos.z) / scale;

        obs[9] = vel.x / step_size;
        obs[10] = vel.z / step_size;

        const int nids = grid.queryItems(
            pos.x - options.sense_radius,
            pos.z - options.sense_radius,
            pos.x + options.sense_radius,
            pos.z + options.sense_radius,
            ids,
            kCandidateLimit);

        struct Candidate {
            int id = -1;
            float dist = 0.0f;
        };
        std::array<Candidate, kObservedOthers> nearest{};
        for (Candidate& candidate : nearest) {
            candidate.dist = std::numeric_limits<float>::max();
        }

        for (int c = 0; c < nids; ++c) {
            const int other_id = static_cast<int>(ids[c]);
            if (other_id == i || other_id < 0 || other_id >= agents) {
                continue;
            }
            const float dx = positions[other_id].x - pos.x;
            const float dz = positions[other_id].z - pos.z;
            const float dist = std::sqrt(dx * dx + dz * dz);
            if (dist > options.sense_radius) {
                continue;
            }

            for (int slot = 0; slot < kObservedOthers; ++slot) {
                if (dist < nearest[slot].dist) {
                    for (int move = kObservedOthers - 1; move > slot; --move) {
                        nearest[move] = nearest[move - 1];
                    }
                    nearest[slot] = Candidate{other_id, dist};
                    break;
                }
            }
        }

        int sensed = 0;
        for (int slot = 0; slot < kObservedOthers; ++slot) {
            const Candidate& candidate = nearest[slot];
            if (candidate.id < 0) {
                continue;
            }
            const Vec2& other = positions[candidate.id];
            const int base = 11 + slot * 4;
            obs[base + 0] = (other.x - pos.x) / scale;
            obs[base + 1] = (other.z - pos.z) / scale;
            obs[base + 2] = candidate.dist / scale;
            obs[base + 3] = 1.0f;  // fixed heuristic id placeholder.
            ++sensed;
        }

        obs[obs_dim - 1] = sensed == 0 ? 1.0f : 0.0f;
    }
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

Metrics benchmark_onnx_pipeline(const OnnxPipelineBenchmarkOptions& options) {
#ifndef HAS_ONNXRUNTIME
    (void)options;
    throw std::runtime_error("ONNX Runtime is disabled. Define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR.");
#else
    if (options.agents <= 0 || options.obs_dim != 24 || options.iterations <= 0 || options.warmup < 0) {
        throw std::runtime_error("Invalid ONNX pipeline options. Expected agents > 0, obs_dim == 24, iterations > 0.");
    }

    sac_pathfind::DetourNavMeshWrapper nav;
    if (!nav.load_navmesh(options.navmesh_path)) {
        throw std::runtime_error("Detour navmesh load failed for ONNX pipeline: " + nav.last_error());
    }
    const Bounds bounds = compute_navmesh_bounds(nav.navmesh());
    const std::vector<Vec2> positions = make_agent_positions(bounds, options.agents);
    std::vector<Vec2> velocities(positions.size(), Vec2{0.0f, 0.0f});
    const Vec2 goal{bounds.bmax[0], bounds.bmax[2]};

    dtProximityGrid grid;
    if (!grid.init(options.agents * 4, options.agent_radius * 3.0f)) {
        throw std::runtime_error("dtProximityGrid::init failed for ONNX pipeline.");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "base_move_pipeline");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const auto model_path = to_ort_path(options.onnx_path);
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    std::vector<float> observations(static_cast<std::size_t>(options.agents) * static_cast<std::size_t>(options.obs_dim), 0.0f);
    std::array<int64_t, 2> input_shape{
        static_cast<int64_t>(options.agents),
        static_cast<int64_t>(options.obs_dim),
    };
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto run_once = [&]() {
        build_pipeline_observations(options, positions, velocities, goal, bounds, grid, observations);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info,
            observations.data(),
            observations.size(),
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

    for (int i = 0; i < options.warmup; ++i) {
        run_once();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    const auto total_start = Clock::now();
    for (int i = 0; i < options.iterations; ++i) {
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
