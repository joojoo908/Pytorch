#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct dtNavMesh;
struct dtNavMeshQuery;

namespace sac_pathfind {

struct Vec3 {
    float x;
    float y;
    float z;
};

class DetourNavMeshWrapper {
public:
    DetourNavMeshWrapper();
    ~DetourNavMeshWrapper();

    DetourNavMeshWrapper(const DetourNavMeshWrapper&) = delete;
    DetourNavMeshWrapper& operator=(const DetourNavMeshWrapper&) = delete;

    bool load_navmesh(const std::string& navmesh_path);
    std::optional<Vec3> find_next_waypoint(const Vec3& start, const Vec3& goal) const;
    void reset();

    bool is_loaded() const noexcept;
    const std::string& last_error() const noexcept;

    const dtNavMesh* navmesh() const noexcept;
    const dtNavMeshQuery* query() const noexcept;

private:
    bool fail(const std::string& message);

    dtNavMesh* navmesh_;
    dtNavMeshQuery* query_;
    std::string last_error_;
};

}  // namespace sac_pathfind
