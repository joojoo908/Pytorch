#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "BenchmarkTypes.h"
#include "DetourCrowdBenchmark.h"
#include "OnnxBenchmark.h"
#include "detour_navmesh_wrapper.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Vec3Arg {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Options {
    std::string onnx_path =
        R"(onnx_roles\base_move.onnx)";
    std::string navmesh_path =
        R"(Resources\NavMesh\all_tiles_navmesh.bin)";
    int obs_dim = 24;
    int iterations = 10000;
    int warmup = 1000;
    int crowd_agents = 20;
    float crowd_dt = 1.0f / 60.0f;
    Vec3Arg detour_start{0.0f, 0.0f, 0.0f};
    Vec3Arg detour_goal{10.0f, 0.0f, 10.0f};
};

[[noreturn]] void fail_usage(const std::string& message) {
    std::cerr << message << "\n\n"
              << "Usage:\n"
              << "  BaseMovePerformanceTest.exe [options]\n\n"
              << "Options:\n"
              << "  --onnx <path>             base_move.onnx path\n"
              << "  --navmesh <path>          Detour navmesh binary path\n"
              << "  --obs-dim <n>             ONNX observation size, default 24\n"
              << "  --iterations <n>          measured iterations, default 10000\n"
              << "  --warmup <n>              warmup iterations, default 1000\n"
              << "  --crowd-agents <n>        DetourCrowd agents, default 32\n"
              << "  --crowd-dt <seconds>      DetourCrowd update dt, default 0.0166667\n"
              << "  --detour-start <x> <y> <z>  Detour-space start, default 0 0 0\n"
              << "  --detour-goal <x> <y> <z>   Detour-space goal, default 10 0 10\n";
    std::exit(2);
}

int parse_int(const char* text, const char* name) {
    try {
        std::size_t used = 0;
        const int value = std::stoi(text, &used);
        if (used != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        fail_usage(std::string("Invalid integer for ") + name + ": " + text);
    }
}

float parse_float(const char* text, const char* name) {
    try {
        std::size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        fail_usage(std::string("Invalid float for ") + name + ": " + text);
    }
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                fail_usage(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--onnx") {
            opt.onnx_path = require_value("--onnx");
        } else if (arg == "--navmesh") {
            opt.navmesh_path = require_value("--navmesh");
        } else if (arg == "--obs-dim") {
            opt.obs_dim = parse_int(require_value("--obs-dim"), "--obs-dim");
        } else if (arg == "--iterations") {
            opt.iterations = parse_int(require_value("--iterations"), "--iterations");
        } else if (arg == "--warmup") {
            opt.warmup = parse_int(require_value("--warmup"), "--warmup");
        } else if (arg == "--crowd-agents") {
            opt.crowd_agents = parse_int(require_value("--crowd-agents"), "--crowd-agents");
        } else if (arg == "--crowd-dt") {
            opt.crowd_dt = parse_float(require_value("--crowd-dt"), "--crowd-dt");
        } else if (arg == "--detour-start" || arg == "--detour-goal") {
            if (i + 3 >= argc) {
                fail_usage("Missing x y z values for " + arg);
            }
            Vec3Arg v{
                parse_float(argv[++i], arg.c_str()),
                parse_float(argv[++i], arg.c_str()),
                parse_float(argv[++i], arg.c_str()),
            };
            if (arg == "--detour-start") {
                opt.detour_start = v;
            } else {
                opt.detour_goal = v;
            }
        } else if (arg == "--help" || arg == "-h") {
            fail_usage("");
        } else {
            fail_usage("Unknown option: " + arg);
        }
    }

    if (opt.obs_dim <= 0 || opt.iterations <= 0 || opt.warmup < 0 || opt.crowd_agents <= 0 || opt.crowd_dt <= 0.0f) {
        fail_usage("--obs-dim, --iterations, --crowd-agents, --crowd-dt must be positive; --warmup must be >= 0");
    }
    return opt;
}

void print_metrics(const std::string& name, const Metrics& m) {
    std::cout << std::left << std::setw(18) << name
              << " total_ms=" << std::fixed << std::setprecision(3) << m.total_ms
              << " avg_us=" << std::setprecision(3) << m.avg_us
              << " p50_us=" << std::setprecision(3) << m.p50_us
              << " p95_us=" << std::setprecision(3) << m.p95_us
              << " p99_us=" << std::setprecision(3) << m.p99_us << "\n";
}

Metrics benchmark_detour_query(const Options& opt) {
    sac_pathfind::DetourNavMeshWrapper nav;
    if (!nav.load_navmesh(opt.navmesh_path)) {
        throw std::runtime_error("Detour navmesh load failed: " + nav.last_error());
    }

    const sac_pathfind::Vec3 start{opt.detour_start.x, opt.detour_start.y, opt.detour_start.z};
    const sac_pathfind::Vec3 goal{opt.detour_goal.x, opt.detour_goal.y, opt.detour_goal.z};

    auto run_once = [&]() {
        const auto waypoint = nav.find_next_waypoint(start, goal);
        volatile float sink = waypoint ? waypoint->x : -1.0f;
        (void)sink;
    };

    for (int i = 0; i < opt.warmup; ++i) {
        run_once();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(opt.iterations));
    const auto total_start = Clock::now();
    for (int i = 0; i < opt.iterations; ++i) {
        const auto t0 = Clock::now();
        run_once();
        const auto t1 = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const auto total_end = Clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return compute_metrics(std::move(samples), total_ms);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        std::cout << "BaseMove ONNX vs Detour benchmark\n"
                  << "iterations=" << opt.iterations
                  << " warmup=" << opt.warmup
                  << " obs_dim=" << opt.obs_dim
                  << " crowd_agents=" << opt.crowd_agents
                  << " crowd_dt=" << opt.crowd_dt << "\n"
                  << "onnx=" << opt.onnx_path << "\n"
                  << "navmesh=" << opt.navmesh_path << "\n";

        if (is_onnx_runtime_enabled()) {
            print_metrics(
                "onnx_base_move",
                benchmark_onnx_base_move(opt.onnx_path, opt.obs_dim, opt.iterations, opt.warmup));
            OnnxPipelineBenchmarkOptions pipeline_options;
            pipeline_options.onnx_path = opt.onnx_path;
            pipeline_options.navmesh_path = opt.navmesh_path;
            pipeline_options.agents = opt.crowd_agents;
            pipeline_options.obs_dim = opt.obs_dim;
            pipeline_options.iterations = opt.iterations;
            pipeline_options.warmup = opt.warmup;
            print_metrics("onnx_pipeline", benchmark_onnx_pipeline(pipeline_options));
        } else {
            std::cout << "onnx_base_move   skipped: define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR\n";
            std::cout << "onnx_pipeline    skipped: define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR\n";
        }

        if (is_detour_crowd_enabled()) {
            DetourCrowdBenchmarkOptions crowd_options;
            crowd_options.navmesh_path = opt.navmesh_path;
            crowd_options.agents = opt.crowd_agents;
            crowd_options.iterations = opt.iterations;
            crowd_options.warmup = opt.warmup;
            crowd_options.dt = opt.crowd_dt;
            print_metrics("detour_crowd", benchmark_detour_crowd_avoidance(crowd_options));
        } else {
            std::cout << "detour_crowd    skipped: DetourCrowd.h/DetourCrowd.lib are not available\n";
        }

        print_metrics("detour_query", benchmark_detour_query(opt));
        std::cout << "\nNote: detour_crowd is the relevant comparison for local collision avoidance. "
                     "detour_query is path/waypoint query cost.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
