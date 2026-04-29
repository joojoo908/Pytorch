#include "ArrivalBenchmark.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "DetourNavMesh.h"
#include "DetourNavMeshQuery.h"
#include "DetourProximityGrid.h"
#include "detour_navmesh_wrapper.h"

#ifdef HAS_DETOURCROWD
#include <DetourCrowd.h>
#endif

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {

constexpr float kHalfExtents[3] = {200.0f, 400.0f, 200.0f};
constexpr int kObservedOthers = 3;
constexpr int kCandidateLimit = 64;
constexpr float kEngineToDetourScale = 0.01f;
constexpr float kDetourToEngineScale = 100.0f;

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

struct CrowdPoint {
    dtPolyRef ref = 0;
    float pos[3] = {};
};

struct SimAgent {
    CrowdPoint start{};
};

float dist_sq_xz(const float* a, const float* b);

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

CrowdPoint find_nearest_valid_point(
    dtNavMeshQuery* query,
    const dtQueryFilter& filter,
    float x,
    float y,
    float z) {
    const float probe[3] = {x, y, z};
    CrowdPoint out;
    dtStatus status = query->findNearestPoly(probe, kHalfExtents, &filter, &out.ref, out.pos);
    if (dtStatusFailed(status)) {
        out.ref = 0;
    }
    return out;
}

std::vector<CrowdPoint> sample_crowd_points(
    dtNavMeshQuery* query,
    const dtQueryFilter& filter,
    const Bounds& bounds,
    int wanted) {
    std::vector<CrowdPoint> points;
    points.reserve(static_cast<std::size_t>(wanted));

    const float min_x = bounds.bmin[0];
    const float max_x = bounds.bmax[0];
    const float min_y = bounds.bmin[1];
    const float max_y = bounds.bmax[1];
    const float min_z = bounds.bmin[2];
    const float max_z = bounds.bmax[2];
    const float y = 0.5f * (min_y + max_y);

    const int grid = std::max(8, static_cast<int>(std::ceil(std::sqrt(static_cast<float>(wanted)))) * 4);
    const float step_x = (max_x - min_x) / static_cast<float>(std::max(1, grid - 1));
    const float step_z = (max_z - min_z) / static_cast<float>(std::max(1, grid - 1));

    for (int gz = 0; gz < grid && static_cast<int>(points.size()) < wanted; ++gz) {
        for (int gx = 0; gx < grid && static_cast<int>(points.size()) < wanted; ++gx) {
            const float x = min_x + step_x * static_cast<float>(gx);
            const float z = min_z + step_z * static_cast<float>(gz);
            CrowdPoint point = find_nearest_valid_point(query, filter, x, y, z);
            if (point.ref == 0) {
                continue;
            }

            bool duplicate = false;
            for (const CrowdPoint& existing : points) {
                const float dx = existing.pos[0] - point.pos[0];
                const float dz = existing.pos[2] - point.pos[2];
                if ((dx * dx + dz * dz) < 25.0f) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                points.push_back(point);
            }
        }
    }
    return points;
}

std::vector<CrowdPoint> collect_walkable_spawn_points(
    dtNavMeshQuery* query,
    const dtQueryFilter& filter,
    const Bounds& bounds) {
    std::vector<CrowdPoint> points;

    const float min_x = bounds.bmin[0];
    const float max_x = bounds.bmax[0];
    const float min_y = bounds.bmin[1];
    const float max_y = bounds.bmax[1];
    const float min_z = bounds.bmin[2];
    const float max_z = bounds.bmax[2];
    const float y = 0.5f * (min_y + max_y);
    const float span_x = std::max(max_x - min_x, 1.0f);
    const float span_z = std::max(max_z - min_z, 1.0f);
    const float step = 25.0f;
    const int grid_x = std::max(2, static_cast<int>(std::ceil(span_x / step)) + 1);
    const int grid_z = std::max(2, static_cast<int>(std::ceil(span_z / step)) + 1);
    const float step_x = span_x / static_cast<float>(std::max(1, grid_x - 1));
    const float step_z = span_z / static_cast<float>(std::max(1, grid_z - 1));

    points.reserve(static_cast<std::size_t>(grid_x * grid_z / 4));
    for (int gz = 0; gz < grid_z; ++gz) {
        for (int gx = 0; gx < grid_x; ++gx) {
            const float x = min_x + step_x * static_cast<float>(gx);
            const float z = min_z + step_z * static_cast<float>(gz);
            CrowdPoint point = find_nearest_valid_point(query, filter, x, y, z);
            if (point.ref == 0) {
                continue;
            }
            points.push_back(point);
        }
    }
    return points;
}

CrowdPoint sample_spawn_point(
    const std::vector<CrowdPoint>& free_points,
    std::mt19937& rng,
    const std::vector<std::array<float, 3>>& avoid_points,
    float min_dist,
    const std::array<float, 3>* anchor,
    float max_dist) {
    if (free_points.empty()) {
        throw std::runtime_error("No free walkable points available for spawn sampling.");
    }

    const float min_dist_sq = min_dist * min_dist;
    const bool use_anchor = (anchor != nullptr && max_dist > 0.0f);
    const float max_dist_sq = max_dist * max_dist;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(free_points.size()) - 1);

    for (int tries = 0; tries < 128; ++tries) {
        const CrowdPoint& point = free_points[static_cast<std::size_t>(pick(rng))];
        if (use_anchor) {
            const float adx = point.pos[0] - (*anchor)[0];
            const float adz = point.pos[2] - (*anchor)[2];
            if (adx * adx + adz * adz > max_dist_sq) {
                continue;
            }
        }
        bool ok = true;
        for (const auto& avoid : avoid_points) {
            if (dist_sq_xz(point.pos, avoid.data()) < min_dist_sq) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return point;
        }
    }

    if (use_anchor) {
        const CrowdPoint* best = nullptr;
        float best_d2 = std::numeric_limits<float>::max();
        for (const CrowdPoint& point : free_points) {
            const float adx = point.pos[0] - (*anchor)[0];
            const float adz = point.pos[2] - (*anchor)[2];
            const float anchor_d2 = adx * adx + adz * adz;
            if (anchor_d2 > max_dist_sq) {
                continue;
            }
            bool ok = true;
            for (const auto& avoid : avoid_points) {
                if (dist_sq_xz(point.pos, avoid.data()) < min_dist_sq) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
            if (anchor_d2 < best_d2) {
                best = &point;
                best_d2 = anchor_d2;
            }
        }
        if (best != nullptr) {
            return *best;
        }
    }

    return free_points[static_cast<std::size_t>(pick(rng))];
}

std::vector<SimAgent> build_sim_agents(
    sac_pathfind::DetourNavMeshWrapper& nav,
    const CrowdPoint& shared_goal,
    int agents,
    int run_index,
    float min_goal_distance,
    float agent_radius) {
    dtQueryFilter filter;
    dtNavMeshQuery* query = const_cast<dtNavMeshQuery*>(nav.query());
    const Bounds bounds = compute_navmesh_bounds(nav.navmesh());
    std::vector<CrowdPoint> free_points = collect_walkable_spawn_points(query, filter, bounds);
    if (free_points.empty()) {
        const int wanted = std::max(agents * 128, agents + 16);
        free_points = sample_crowd_points(query, filter, bounds, wanted);
    }
    if (free_points.empty()) {
        throw std::runtime_error("No valid navmesh spawn points available.");
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(run_index + 1337));
    const float min_goal_distance_sq = min_goal_distance * min_goal_distance;
    const float success_radius = min_goal_distance;
    const float min_dist = std::max(agent_radius * 1.0f, success_radius * 0.00001f);
    const float max_dist = std::max(min_dist, success_radius * 0.0001f);

    std::vector<CrowdPoint> start_candidates;
    start_candidates.reserve(free_points.size());
    for (const CrowdPoint& point : free_points) {
        if (dist_sq_xz(point.pos, shared_goal.pos) > min_goal_distance_sq) {
            start_candidates.push_back(point);
        }
    }
    if (start_candidates.empty()) {
        start_candidates = free_points;
    }

    std::uniform_int_distribution<int> start_pick(0, static_cast<int>(start_candidates.size()) - 1);
    const CrowdPoint primary = start_candidates[static_cast<std::size_t>(start_pick(rng))];

    std::vector<SimAgent> sim_agents;
    sim_agents.reserve(static_cast<std::size_t>(agents));
    sim_agents.push_back(SimAgent{primary});

    std::vector<std::array<float, 3>> avoid_points;
    avoid_points.reserve(static_cast<std::size_t>(agents + 1));
    avoid_points.push_back({primary.pos[0], primary.pos[1], primary.pos[2]});
    avoid_points.push_back({shared_goal.pos[0], shared_goal.pos[1], shared_goal.pos[2]});

    const std::array<float, 3> anchor = {primary.pos[0], primary.pos[1], primary.pos[2]};
    for (int i = 1; i < agents; ++i) {
        const CrowdPoint point = sample_spawn_point(free_points, rng, avoid_points, min_dist, &anchor, max_dist);
        sim_agents.push_back(SimAgent{point});
        avoid_points.push_back({point.pos[0], point.pos[1], point.pos[2]});
    }
    return sim_agents;
}

CrowdPoint build_shared_goal(sac_pathfind::DetourNavMeshWrapper& nav, const ArrivalBenchmarkOptions& options) {
    dtQueryFilter filter;
    dtNavMeshQuery* query = const_cast<dtNavMeshQuery*>(nav.query());
    CrowdPoint goal = find_nearest_valid_point(
        query,
        filter,
        options.shared_goal_x,
        options.shared_goal_y,
        options.shared_goal_z);
    if (goal.ref == 0) {
        throw std::runtime_error("Shared arrival goal could not be snapped onto the navmesh.");
    }
    return goal;
}

float dist_sq_xz(const float* a, const float* b) {
    const float dx = a[0] - b[0];
    const float dz = a[2] - b[2];
    return dx * dx + dz * dz;
}

bool collides_with_other_agents(
    const std::vector<std::array<float, 3>>& positions,
    float radius,
    const std::array<float, 3>& pos,
    int ignore_index) {
    const float min_dist = radius * 2.0f;
    const float min_dist_sq = min_dist * min_dist;
    for (int idx = 0; idx < static_cast<int>(positions.size()); ++idx) {
        if (idx == ignore_index) {
            continue;
        }
        const float dx = pos[0] - positions[static_cast<std::size_t>(idx)][0];
        const float dz = pos[2] - positions[static_cast<std::size_t>(idx)][2];
        if (dx * dx + dz * dz < min_dist_sq) {
            return true;
        }
    }
    return false;
}

int count_collision_events(
    const std::vector<std::array<float, 3>>& positions,
    float radius,
    std::vector<bool>* collided_agents) {
    int events = 0;
    const float min_dist = radius * 2.0f;
    const float min_dist_sq = min_dist * min_dist;
    for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(positions.size()); ++j) {
            const float dx = positions[static_cast<std::size_t>(i)][0] - positions[static_cast<std::size_t>(j)][0];
            const float dz = positions[static_cast<std::size_t>(i)][2] - positions[static_cast<std::size_t>(j)][2];
            if (dx * dx + dz * dz < min_dist_sq) {
                ++events;
                if (collided_agents != nullptr) {
                    (*collided_agents)[static_cast<std::size_t>(i)] = true;
                    (*collided_agents)[static_cast<std::size_t>(j)] = true;
                }
            }
        }
    }
    return events;
}

ArrivalStats make_arrival_stats(
    const std::vector<std::array<float, 3>>& positions,
    const std::vector<std::array<float, 3>>& goals,
    const std::vector<bool>& collided_agents,
    int steps_run,
    int collision_events,
    float success_radius) {
    ArrivalStats stats;
    stats.agents = static_cast<int>(positions.size());
    stats.steps_run = steps_run;
    stats.collision_events = collision_events;
    const float success_radius_sq = success_radius * success_radius;
    float total_final_dist = 0.0f;
    for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
        const float d_sq = dist_sq_xz(positions[static_cast<std::size_t>(i)].data(), goals[static_cast<std::size_t>(i)].data());
        const float d = std::sqrt(std::max(0.0f, d_sq)) * kDetourToEngineScale;
        total_final_dist += d;
        stats.max_final_goal_dist = std::max(stats.max_final_goal_dist, d);
        if (d_sq <= success_radius_sq) {
            ++stats.arrived;
            if (!collided_agents[static_cast<std::size_t>(i)]) {
                ++stats.collision_free_arrived;
            }
        }
        if (collided_agents[static_cast<std::size_t>(i)]) {
            ++stats.collided_agents;
        }
    }
    stats.avg_final_goal_dist = positions.empty() ? 0.0f : total_final_dist / static_cast<float>(positions.size());
    return stats;
}

void build_base_move_observations(
    const ArrivalBenchmarkOptions& options,
    const Bounds& bounds,
    const std::vector<std::array<float, 3>>& positions,
    const std::vector<std::array<float, 3>>& velocities,
    const std::vector<std::array<float, 3>>& goals,
    dtProximityGrid& grid,
    std::vector<float>& observations) {
    const int agents = static_cast<int>(positions.size());
    unsigned short ids[kCandidateLimit];

    std::fill(observations.begin(), observations.end(), 0.0f);
    grid.clear();
    for (int i = 0; i < agents; ++i) {
        const auto& pos = positions[static_cast<std::size_t>(i)];
        grid.addItem(
            static_cast<unsigned short>(i),
            pos[0] - options.agent_radius,
            pos[2] - options.agent_radius,
            pos[0] + options.agent_radius,
            pos[2] + options.agent_radius);
    }

    const float center_x = 0.5f * (bounds.bmin[0] + bounds.bmax[0]);
    const float center_z = 0.5f * (bounds.bmin[2] + bounds.bmax[2]);
    const float scale = std::max({bounds.bmax[0] - bounds.bmin[0], bounds.bmax[2] - bounds.bmin[2], 1.0f});

    for (int i = 0; i < agents; ++i) {
        float* obs = observations.data() + static_cast<std::size_t>(i * options.obs_dim);
        const auto& pos = positions[static_cast<std::size_t>(i)];
        const auto& vel = velocities[static_cast<std::size_t>(i)];
        const auto& goal = goals[static_cast<std::size_t>(i)];

        obs[0] = (pos[0] - center_x) / scale;
        obs[1] = 0.0f;
        obs[2] = (pos[2] - center_z) / scale;
        obs[3] = (goal[0] - center_x) / scale;
        obs[4] = 0.0f;
        obs[5] = (goal[2] - center_z) / scale;
        obs[6] = (goal[0] - pos[0]) / scale;
        obs[7] = 0.0f;
        obs[8] = (goal[2] - pos[2]) / scale;
        obs[9] = vel[0] / std::max(options.step_size, 1.0f);
        obs[10] = vel[2] / std::max(options.step_size, 1.0f);

        const int nids = grid.queryItems(
            pos[0] - options.sense_radius,
            pos[2] - options.sense_radius,
            pos[0] + options.sense_radius,
            pos[2] + options.sense_radius,
            ids,
            kCandidateLimit);

        struct Candidate {
            int id = -1;
            float dist = std::numeric_limits<float>::max();
        };
        std::array<Candidate, kObservedOthers> nearest{};

        for (int c = 0; c < nids; ++c) {
            const int other_id = static_cast<int>(ids[c]);
            if (other_id == i || other_id < 0 || other_id >= agents) {
                continue;
            }
            const auto& other = positions[static_cast<std::size_t>(other_id)];
            const float dx = other[0] - pos[0];
            const float dz = other[2] - pos[2];
            const float dist = std::sqrt(dx * dx + dz * dz);
            if (dist > options.sense_radius) {
                continue;
            }

            for (int slot = 0; slot < kObservedOthers; ++slot) {
                if (dist < nearest[static_cast<std::size_t>(slot)].dist) {
                    for (int move = kObservedOthers - 1; move > slot; --move) {
                        nearest[static_cast<std::size_t>(move)] = nearest[static_cast<std::size_t>(move - 1)];
                    }
                    nearest[static_cast<std::size_t>(slot)] = Candidate{other_id, dist};
                    break;
                }
            }
        }

        int sensed = 0;
        for (int slot = 0; slot < kObservedOthers; ++slot) {
            const Candidate& candidate = nearest[static_cast<std::size_t>(slot)];
            if (candidate.id < 0) {
                continue;
            }
            const auto& other = positions[static_cast<std::size_t>(candidate.id)];
            const int base = 11 + slot * 4;
            obs[base + 0] = (other[0] - pos[0]) / scale;
            obs[base + 1] = (other[2] - pos[2]) / scale;
            obs[base + 2] = candidate.dist / scale;
            obs[base + 3] = 1.0f;
            ++sensed;
        }

        obs[options.obs_dim - 1] = sensed == 0 ? 1.0f : 0.0f;
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

bool is_arrival_base_move_enabled() {
#ifdef HAS_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

ArrivalStats benchmark_base_move_arrival(const ArrivalBenchmarkOptions& options) {
#ifndef HAS_ONNXRUNTIME
    (void)options;
    throw std::runtime_error("ONNX Runtime is disabled. Define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR.");
#else
    if (options.agents <= 0 || options.obs_dim != 24 || options.max_steps <= 0) {
        throw std::runtime_error("Invalid base_move arrival options.");
    }

    sac_pathfind::DetourNavMeshWrapper nav;
    if (!nav.load_navmesh(options.navmesh_path)) {
        throw std::runtime_error("Detour navmesh load failed for base_move arrival: " + nav.last_error());
    }

    const CrowdPoint shared_goal = build_shared_goal(nav, options);
    const Bounds bounds = compute_navmesh_bounds(nav.navmesh());
    const std::vector<SimAgent> sim_agents =
        build_sim_agents(nav, shared_goal, options.agents, options.run_index, options.sense_radius, options.agent_radius);
    dtQueryFilter filter;
    dtNavMeshQuery* query = const_cast<dtNavMeshQuery*>(nav.query());

    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 3>> goals;
    std::vector<std::array<float, 3>> velocities(static_cast<std::size_t>(options.agents), {0.0f, 0.0f, 0.0f});
    std::vector<bool> collided_agents(static_cast<std::size_t>(options.agents), false);
    positions.reserve(static_cast<std::size_t>(options.agents));
    goals.reserve(static_cast<std::size_t>(options.agents));

    for (const SimAgent& agent : sim_agents) {
        positions.push_back({agent.start.pos[0], agent.start.pos[1], agent.start.pos[2]});
        goals.push_back({shared_goal.pos[0], shared_goal.pos[1], shared_goal.pos[2]});
    }

    dtProximityGrid grid;
    if (!grid.init(options.agents * 4, options.agent_radius * 3.0f)) {
        throw std::runtime_error("dtProximityGrid::init failed for base_move arrival.");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "base_move_arrival");
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

    std::vector<float> observations(static_cast<std::size_t>(options.agents * options.obs_dim), 0.0f);
    std::array<int64_t, 2> input_shape{
        static_cast<int64_t>(options.agents),
        static_cast<int64_t>(options.obs_dim),
    };
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int total_collision_events = 0;
    int steps_run = 0;

    for (int step = 0; step < options.max_steps; ++step) {
        steps_run = step + 1;
        build_base_move_observations(options, bounds, positions, velocities, goals, grid, observations);

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
        float* action_data = outputs.front().GetTensorMutableData<float>();

        bool all_arrived = true;
        const float success_radius_sq = options.sense_radius * options.sense_radius;
        for (int i = 0; i < options.agents; ++i) {
            const auto old_pos = positions[static_cast<std::size_t>(i)];
            const auto goal = goals[static_cast<std::size_t>(i)];
            if (dist_sq_xz(old_pos.data(), goal.data()) > success_radius_sq) {
                all_arrived = false;
            }

            std::array<float, 3> desired_target = goal;
            const float sensor_fail_code = observations[static_cast<std::size_t>(i * options.obs_dim + (options.obs_dim - 1))];
            if (sensor_fail_code <= 0.5f) {
                const float ax = std::max(-1.0f, std::min(1.0f, action_data[i * 2 + 0]));
                const float az = std::max(-1.0f, std::min(1.0f, action_data[i * 2 + 1]));
                desired_target = {
                    old_pos[0] + ax * options.tactical_target_radius,
                    old_pos[1],
                    old_pos[2] + az * options.tactical_target_radius,
                };
            }

            CrowdPoint snapped = find_nearest_valid_point(query, filter, desired_target[0], old_pos[1], desired_target[2]);
            std::array<float, 3> target = old_pos;
            if (snapped.ref != 0) {
                target = {snapped.pos[0], snapped.pos[1], snapped.pos[2]};
            }

            const auto waypoint = nav.find_next_waypoint(
                sac_pathfind::Vec3{old_pos[0], old_pos[1], old_pos[2]},
                sac_pathfind::Vec3{target[0], target[1], target[2]});
            std::array<float, 3> tactical_target = target;
            if (waypoint.has_value()) {
                tactical_target = {waypoint->x, waypoint->y, waypoint->z};
            }

            std::array<float, 3> next = tactical_target;
            const float dx = tactical_target[0] - old_pos[0];
            const float dz = tactical_target[2] - old_pos[2];
            const float move_dist = std::sqrt(dx * dx + dz * dz);
            if (move_dist > options.step_size && move_dist > 1e-6f) {
                const float scale = options.step_size / move_dist;
                next = {
                    old_pos[0] + dx * scale,
                    old_pos[1] + (tactical_target[1] - old_pos[1]) * scale,
                    old_pos[2] + dz * scale,
                };
            }

            CrowdPoint snapped_next = find_nearest_valid_point(query, filter, next[0], next[1], next[2]);
            if (snapped_next.ref != 0) {
                next = {snapped_next.pos[0], snapped_next.pos[1], snapped_next.pos[2]};
            }

            if (collides_with_other_agents(positions, options.agent_radius, next, i)) {
                collided_agents[static_cast<std::size_t>(i)] = true;
                if (options.base_move_collision_resolve) {
                    std::array<float, 3> lo = old_pos;
                    std::array<float, 3> hi = next;
                    std::array<float, 3> best = old_pos;
                    for (int iter = 0; iter < 7; ++iter) {
                        std::array<float, 3> mid = {
                            0.5f * (lo[0] + hi[0]),
                            0.5f * (lo[1] + hi[1]),
                            0.5f * (lo[2] + hi[2]),
                        };
                        if (collides_with_other_agents(positions, options.agent_radius, mid, i)) {
                            hi = mid;
                        } else {
                            best = mid;
                            lo = mid;
                        }
                    }
                    next = best;
                }
            }

            velocities[static_cast<std::size_t>(i)] = {
                next[0] - old_pos[0],
                next[1] - old_pos[1],
                next[2] - old_pos[2],
            };
            positions[static_cast<std::size_t>(i)] = next;
        }

        total_collision_events += count_collision_events(positions, options.agent_radius, &collided_agents);
        if (all_arrived) {
            break;
        }
    }

    return make_arrival_stats(positions, goals, collided_agents, steps_run, total_collision_events, options.sense_radius);
#endif
}

ArrivalStats benchmark_detour_crowd_arrival(const ArrivalBenchmarkOptions& options) {
#ifndef HAS_DETOURCROWD
    (void)options;
    throw std::runtime_error(
        "DetourCrowd is disabled. Add DetourCrowd headers/libs and define HAS_DETOURCROWD.");
#else
    if (options.agents <= 0 || options.max_steps <= 0 || options.dt <= 0.0f) {
        throw std::runtime_error("Invalid DetourCrowd arrival options.");
    }

    sac_pathfind::DetourNavMeshWrapper nav;
    if (!nav.load_navmesh(options.navmesh_path)) {
        throw std::runtime_error("Detour navmesh load failed for crowd arrival: " + nav.last_error());
    }

    const CrowdPoint shared_goal = build_shared_goal(nav, options);
    const std::vector<SimAgent> sim_agents =
        build_sim_agents(nav, shared_goal, options.agents, options.run_index, options.sense_radius, options.agent_radius);

    dtCrowd* crowd = dtAllocCrowd();
    if (crowd == nullptr) {
        throw std::runtime_error("dtAllocCrowd failed.");
    }
    if (!crowd->init(options.agents, 35.0f, const_cast<dtNavMesh*>(nav.navmesh()))) {
        dtFreeCrowd(crowd);
        throw std::runtime_error("dtCrowd::init failed.");
    }

    dtCrowdAgentParams params{};
    params.radius = options.agent_radius;
    params.height = 180.0f;
    params.maxAcceleration = 800.0f;
    params.maxSpeed = options.step_size / std::max(options.dt, 1e-6f);
    params.collisionQueryRange = params.radius * 12.0f;
    params.pathOptimizationRange = params.radius * 30.0f;
    params.updateFlags =
        DT_CROWD_ANTICIPATE_TURNS |
        DT_CROWD_OBSTACLE_AVOIDANCE |
        DT_CROWD_SEPARATION;
    params.obstacleAvoidanceType = 3;
    params.separationWeight = 2.0f;

    std::vector<int> agent_ids;
    std::vector<std::array<float, 3>> goals;
    agent_ids.reserve(static_cast<std::size_t>(options.agents));
    goals.reserve(static_cast<std::size_t>(options.agents));

    for (int i = 0; i < options.agents; ++i) {
        const SimAgent& agent = sim_agents[static_cast<std::size_t>(i)];
        const int agent_id = crowd->addAgent(agent.start.pos, &params);
        if (agent_id >= 0) {
            agent_ids.push_back(agent_id);
            goals.push_back({shared_goal.pos[0], shared_goal.pos[1], shared_goal.pos[2]});
            crowd->requestMoveTarget(agent_id, shared_goal.ref, shared_goal.pos);
        }
    }

    if (agent_ids.empty()) {
        dtFreeCrowd(crowd);
        throw std::runtime_error("No DetourCrowd agents could be placed on the navmesh.");
    }

    std::vector<bool> collided_agents(agent_ids.size(), false);
    std::vector<std::array<float, 3>> positions(agent_ids.size(), {0.0f, 0.0f, 0.0f});
    int total_collision_events = 0;
    int steps_run = 0;

    for (int step = 0; step < options.max_steps; ++step) {
        steps_run = step + 1;
        crowd->update(options.dt, nullptr);

        bool all_arrived = true;
        const float success_radius_sq = options.sense_radius * options.sense_radius;
        for (int i = 0; i < static_cast<int>(agent_ids.size()); ++i) {
            const dtCrowdAgent* agent = crowd->getAgent(agent_ids[static_cast<std::size_t>(i)]);
            if (agent == nullptr || !agent->active) {
                continue;
            }
            positions[static_cast<std::size_t>(i)] = {agent->npos[0], agent->npos[1], agent->npos[2]};
            if (dist_sq_xz(agent->npos, goals[static_cast<std::size_t>(i)].data()) > success_radius_sq) {
                all_arrived = false;
            }
        }

        total_collision_events += count_collision_events(positions, options.agent_radius, &collided_agents);
        if (all_arrived) {
            break;
        }
    }

    dtFreeCrowd(crowd);
    return make_arrival_stats(positions, goals, collided_agents, steps_run, total_collision_events, options.sense_radius);
#endif
}
