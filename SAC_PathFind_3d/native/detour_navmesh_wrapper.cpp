#include "detour_navmesh_wrapper.h"

#include <DetourAlloc.h>
#include <DetourNavMesh.h>
#include <DetourNavMeshQuery.h>
#include <DetourStatus.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace {

constexpr std::int32_t kNavMeshSetMagic =
    ('M' << 24) | ('S' << 16) | ('E' << 8) | ('T');
constexpr std::int32_t kNavMeshSetVersion = 1;
constexpr std::int32_t kTileDataMagic =
    ('D' << 24) | ('N' << 16) | ('A' << 8) | ('V');
constexpr std::int32_t kTileDataVersion = 7;

struct NavMeshSetHeader {
    std::int32_t magic;
    std::int32_t version;
    std::int32_t num_tiles;
    dtNavMeshParams params;
};

struct NavMeshTileHeader {
    std::uint32_t tile_ref;
    std::int32_t data_size;
};

struct DtMeshHeaderPrefix {
    std::int32_t magic;
    std::int32_t version;
};

template <typename T>
bool read_struct(const std::vector<unsigned char>& bytes, std::size_t offset, T* out) {
    if (offset + sizeof(T) > bytes.size()) {
        return false;
    }
    std::memcpy(out, bytes.data() + offset, sizeof(T));
    return true;
}

std::string status_to_string(dtStatus status) {
    std::ostringstream oss;
    oss << "0x" << std::hex << static_cast<unsigned int>(status);
    return oss.str();
}

constexpr float kHalfExtents[3] = {200.0f, 400.0f, 200.0f};

struct QueryFilterHolder {
    dtQueryFilter filter;
};

}  // namespace

namespace sac_pathfind {

DetourNavMeshWrapper::DetourNavMeshWrapper()
    : navmesh_(nullptr), query_(nullptr) {}

DetourNavMeshWrapper::~DetourNavMeshWrapper() {
    reset();
}

bool DetourNavMeshWrapper::load_navmesh(const std::string& navmesh_path) {
    reset();

    std::ifstream ifs(navmesh_path, std::ios::binary);
    if (!ifs) {
        return fail("failed to open navmesh file: " + navmesh_path);
    }

    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    if (size <= 0) {
        return fail("navmesh file is empty: " + navmesh_path);
    }
    ifs.seekg(0, std::ios::beg);

    std::vector<unsigned char> raw(static_cast<std::size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(raw.data()), size)) {
        return fail("failed to read navmesh file: " + navmesh_path);
    }

    NavMeshSetHeader header{};
    if (!read_struct(raw, 0, &header)) {
        return fail("navmesh file too small for set header");
    }
    if (header.magic != kNavMeshSetMagic) {
        std::ostringstream oss;
        oss << "unexpected navmesh set magic: 0x" << std::hex << header.magic;
        return fail(oss.str());
    }
    if (header.version != kNavMeshSetVersion) {
        std::ostringstream oss;
        oss << "unexpected navmesh set version: " << std::dec << header.version;
        return fail(oss.str());
    }

    navmesh_ = dtAllocNavMesh();
    if (!navmesh_) {
        return fail("dtAllocNavMesh failed");
    }

    dtStatus status = navmesh_->init(&header.params);
    if (dtStatusFailed(status)) {
        reset();
        return fail("dtNavMesh::init failed: " + status_to_string(status));
    }

    std::size_t offset = sizeof(NavMeshSetHeader);
    for (std::int32_t i = 0; i < header.num_tiles; ++i) {
        NavMeshTileHeader tile_header{};
        if (!read_struct(raw, offset, &tile_header)) {
            reset();
            return fail("truncated navmesh tile header at tile index " + std::to_string(i));
        }
        offset += sizeof(NavMeshTileHeader);

        if (tile_header.tile_ref == 0 || tile_header.data_size <= 0) {
            break;
        }
        if (offset + static_cast<std::size_t>(tile_header.data_size) > raw.size()) {
            reset();
            return fail("truncated navmesh tile data at tile index " + std::to_string(i));
        }

        DtMeshHeaderPrefix tile_prefix{};
        if (static_cast<std::size_t>(tile_header.data_size) < sizeof(DtMeshHeaderPrefix) ||
            !read_struct(raw, offset, &tile_prefix)) {
            reset();
            return fail("tile data too small at tile index " + std::to_string(i));
        }
        if (tile_prefix.magic != kTileDataMagic) {
            reset();
            std::ostringstream oss;
            oss << "unexpected tile magic at index " << i << ": 0x" << std::hex << tile_prefix.magic;
            return fail(oss.str());
        }
        if (tile_prefix.version != kTileDataVersion) {
            reset();
            std::ostringstream oss;
            oss << "unexpected tile version at index " << i << ": " << std::dec << tile_prefix.version;
            return fail(oss.str());
        }

        auto* tile_data = static_cast<unsigned char*>(
            dtAlloc(static_cast<std::size_t>(tile_header.data_size), DT_ALLOC_PERM));
        if (!tile_data) {
            reset();
            return fail("dtAlloc failed for tile data at tile index " + std::to_string(i));
        }

        std::memcpy(
            tile_data,
            raw.data() + offset,
            static_cast<std::size_t>(tile_header.data_size));

        dtStatus add_status = navmesh_->addTile(
            tile_data,
            tile_header.data_size,
            DT_TILE_FREE_DATA,
            static_cast<dtTileRef>(tile_header.tile_ref),
            nullptr);
        if (dtStatusFailed(add_status)) {
            dtFree(tile_data);
            reset();
            return fail("dtNavMesh::addTile failed at tile index " + std::to_string(i) +
                        ": " + status_to_string(add_status));
        }

        offset += static_cast<std::size_t>(tile_header.data_size);
    }

    query_ = dtAllocNavMeshQuery();
    if (!query_) {
        reset();
        return fail("dtAllocNavMeshQuery failed");
    }

    status = query_->init(navmesh_, 2048);
    if (dtStatusFailed(status)) {
        reset();
        return fail("dtNavMeshQuery::init failed: " + status_to_string(status));
    }

    last_error_.clear();
    return true;
}

std::optional<Vec3> DetourNavMeshWrapper::find_next_waypoint(const Vec3& start, const Vec3& goal) const {
    if (!is_loaded()) {
        return std::nullopt;
    }

    QueryFilterHolder filter_holder;

    const float start_pos[3] = {start.x, start.y, start.z};
    const float goal_pos[3] = {goal.x, goal.y, goal.z};

    dtPolyRef start_ref = 0;
    dtPolyRef goal_ref = 0;
    float nearest_start[3] = {0.0f, 0.0f, 0.0f};
    float nearest_goal[3] = {0.0f, 0.0f, 0.0f};

    dtStatus status = query_->findNearestPoly(
        start_pos, kHalfExtents, &filter_holder.filter, &start_ref, nearest_start);
    if (dtStatusFailed(status) || start_ref == 0) {
        return std::nullopt;
    }

    status = query_->findNearestPoly(
        goal_pos, kHalfExtents, &filter_holder.filter, &goal_ref, nearest_goal);
    if (dtStatusFailed(status) || goal_ref == 0) {
        return std::nullopt;
    }

    constexpr int kMaxPolys = 256;
    dtPolyRef polys[kMaxPolys];
    int poly_count = 0;
    status = query_->findPath(
        start_ref,
        goal_ref,
        nearest_start,
        nearest_goal,
        &filter_holder.filter,
        polys,
        &poly_count,
        kMaxPolys);
    if (dtStatusFailed(status) || poly_count <= 0) {
        return std::nullopt;
    }

    constexpr int kMaxStraight = 256;
    float straight_path[kMaxStraight * 3];
    unsigned char straight_flags[kMaxStraight];
    dtPolyRef straight_polys[kMaxStraight];
    int straight_count = 0;
    status = query_->findStraightPath(
        nearest_start,
        nearest_goal,
        polys,
        poly_count,
        straight_path,
        straight_flags,
        straight_polys,
        &straight_count,
        kMaxStraight);
    if (dtStatusFailed(status) || straight_count <= 0) {
        return std::nullopt;
    }

    const int waypoint_index = (straight_count >= 2) ? 1 : 0;
    const float* wp = &straight_path[waypoint_index * 3];
    return Vec3{wp[0], wp[1], wp[2]};
}

void DetourNavMeshWrapper::reset() {
    if (query_ != nullptr) {
        dtFreeNavMeshQuery(query_);
        query_ = nullptr;
    }
    if (navmesh_ != nullptr) {
        dtFreeNavMesh(navmesh_);
        navmesh_ = nullptr;
    }
}

bool DetourNavMeshWrapper::is_loaded() const noexcept {
    return navmesh_ != nullptr && query_ != nullptr;
}

const std::string& DetourNavMeshWrapper::last_error() const noexcept {
    return last_error_;
}

const dtNavMesh* DetourNavMeshWrapper::navmesh() const noexcept {
    return navmesh_;
}

const dtNavMeshQuery* DetourNavMeshWrapper::query() const noexcept {
    return query_;
}

bool DetourNavMeshWrapper::fail(const std::string& message) {
    last_error_ = message;
    return false;
}

}  // namespace sac_pathfind
