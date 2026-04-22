#include "DetourCrowdBenchmark.h"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef HAS_DETOURCROWD
#include <DetourCrowd.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshQuery.h>
#include <DetourStatus.h>
#endif

#include "detour_navmesh_wrapper.h"

namespace {

using Clock = std::chrono::steady_clock;

#ifdef HAS_DETOURCROWD
constexpr float kHalfExtents[3] = {200.0f, 400.0f, 200.0f};

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

void require_dt_success(dtStatus status, const char* what) {
    if (dtStatusFailed(status)) {
        throw std::runtime_error(std::string(what) + " failed.");
    }
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
#endif

}  // namespace

bool is_detour_crowd_enabled() {
#ifdef HAS_DETOURCROWD
    return true;
#else
    return false;
#endif
}

Metrics benchmark_detour_crowd_avoidance(const DetourCrowdBenchmarkOptions& options) {
#ifndef HAS_DETOURCROWD
    (void)options;
    throw std::runtime_error(
        "DetourCrowd is disabled. Add DetourCrowd headers/libs and define HAS_DETOURCROWD.");
#else
    if (options.agents <= 0 || options.iterations <= 0 || options.warmup < 0 || options.dt <= 0.0f) {
        throw std::runtime_error("Invalid DetourCrowd benchmark options.");
    }

    sac_pathfind::DetourNavMeshWrapper nav;
    if (!nav.load_navmesh(options.navmesh_path)) {
        throw std::runtime_error("Detour navmesh load failed: " + nav.last_error());
    }

    dtCrowd* crowd = dtAllocCrowd();
    if (crowd == nullptr) {
        throw std::runtime_error("dtAllocCrowd failed.");
    }

    const int max_agents = options.agents;
    constexpr float max_agent_radius = 35.0f;
    if (!crowd->init(max_agents, max_agent_radius, const_cast<dtNavMesh*>(nav.navmesh()))) {
        dtFreeCrowd(crowd);
        throw std::runtime_error("dtCrowd::init failed.");
    }

    dtQueryFilter filter;
    dtNavMeshQuery* query = const_cast<dtNavMeshQuery*>(nav.query());

    const Bounds bounds = compute_navmesh_bounds(nav.navmesh());
    const std::vector<CrowdPoint> spawn_points =
        sample_crowd_points(query, filter, bounds, options.agents * 2);

    std::vector<int> agent_ids;
    agent_ids.reserve(static_cast<std::size_t>(options.agents));

    dtCrowdAgentParams params{};
    params.radius = 30.0f;
    params.height = 180.0f;
    params.maxAcceleration = 800.0f;
    params.maxSpeed = 120.0f;
    params.collisionQueryRange = params.radius * 12.0f;
    params.pathOptimizationRange = params.radius * 30.0f;
    params.updateFlags =
        DT_CROWD_ANTICIPATE_TURNS |
        DT_CROWD_OBSTACLE_AVOIDANCE |
        DT_CROWD_SEPARATION;
    params.obstacleAvoidanceType = 3;
    params.separationWeight = 2.0f;

    for (std::size_t i = 0; i < spawn_points.size() && static_cast<int>(agent_ids.size()) < options.agents; ++i) {
        const int agent_id = crowd->addAgent(spawn_points[i].pos, &params);
        if (agent_id >= 0) {
            agent_ids.push_back(agent_id);
        }
    }

    if (agent_ids.empty()) {
        dtFreeCrowd(crowd);
        throw std::runtime_error(
            "No DetourCrowd agents could be placed on the navmesh. "
            "Try reducing --crowd-agents or increasing kHalfExtents.");
    }

    for (std::size_t i = 0; i < agent_ids.size(); ++i) {
        const CrowdPoint& target = spawn_points[spawn_points.size() - 1 - (i % spawn_points.size())];
        if (target.ref != 0) {
            crowd->requestMoveTarget(agent_ids[i], target.ref, target.pos);
        }
    }

    for (int i = 0; i < options.warmup; ++i) {
        crowd->update(options.dt, nullptr);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.iterations));
    const auto total_start = Clock::now();
    for (int i = 0; i < options.iterations; ++i) {
        const auto t0 = Clock::now();
        crowd->update(options.dt, nullptr);
        const auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const auto total_end = Clock::now();

    dtFreeCrowd(crowd);

    const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return compute_metrics(std::move(samples), total_ms);
#endif
}
