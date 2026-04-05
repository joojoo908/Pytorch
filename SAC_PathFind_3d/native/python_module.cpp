#include "detour_navmesh_wrapper.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(detour_navmesh_py, m) {
    m.doc() = "Detour navmesh wrapper for SAC_PathFind_3d";

    py::class_<sac_pathfind::Vec3>(m, "Vec3")
        .def(py::init<>())
        .def_readwrite("x", &sac_pathfind::Vec3::x)
        .def_readwrite("y", &sac_pathfind::Vec3::y)
        .def_readwrite("z", &sac_pathfind::Vec3::z);

    py::class_<sac_pathfind::DetourNavMeshWrapper>(m, "DetourNavMeshWrapper")
        .def(py::init<>())
        .def("load_navmesh", &sac_pathfind::DetourNavMeshWrapper::load_navmesh)
        .def("reset", &sac_pathfind::DetourNavMeshWrapper::reset)
        .def("is_loaded", &sac_pathfind::DetourNavMeshWrapper::is_loaded)
        .def("last_error", &sac_pathfind::DetourNavMeshWrapper::last_error)
        .def(
            "find_next_waypoint",
            [](const sac_pathfind::DetourNavMeshWrapper& self,
               float sx, float sy, float sz,
               float gx, float gy, float gz) -> py::object {
                auto result = self.find_next_waypoint(
                    sac_pathfind::Vec3{sx, sy, sz},
                    sac_pathfind::Vec3{gx, gy, gz});
                if (!result.has_value()) {
                    return py::none();
                }
                return py::make_tuple(result->x, result->y, result->z);
            },
            py::arg("sx"), py::arg("sy"), py::arg("sz"),
            py::arg("gx"), py::arg("gy"), py::arg("gz"));
}
