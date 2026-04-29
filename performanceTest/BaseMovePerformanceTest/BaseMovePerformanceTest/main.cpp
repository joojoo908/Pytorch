#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ArrivalBenchmark.h"
#include "BenchmarkTypes.h"
#include "DetourCrowdBenchmark.h"
#include "OnnxBenchmark.h"
#include "detour_navmesh_wrapper.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr float kEngineToDetourScale = 0.01f;

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
    int arrival_steps = 128;
    int arrival_runs = 10;
    float goal_radius = 120.0f;
    float step_size = 120.0f;
    float tactical_radius = 600.0f;
    float sense_radius = 1000.0f;
    float agent_radius = 30.0f;
    bool base_move_collision_resolve = false;
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
              << "  --arrival-steps <n>       arrival comparison max steps, default 128\n"
              << "  --arrival-runs <n>        arrival scenarios to average, default 1\n"
              << "  --goal-radius <value>     legacy arrival radius option, default 120\n"
              << "  --step-size <value>       base_move step size, default 120\n"
              << "  --tactical-radius <value> base_move tactical target radius, default 600\n"
              << "  --sense-radius <value>    arrival success radius and base_move sense radius, default 600\n"
              << "  --agent-radius <value>    shared agent radius, default 30\n"
              << "  --base-move-resolve <on|off>  base_move collision clamp, default on\n"
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
        } else if (arg == "--arrival-steps") {
            opt.arrival_steps = parse_int(require_value("--arrival-steps"), "--arrival-steps");
        } else if (arg == "--arrival-runs") {
            opt.arrival_runs = parse_int(require_value("--arrival-runs"), "--arrival-runs");
        } else if (arg == "--goal-radius") {
            opt.goal_radius = parse_float(require_value("--goal-radius"), "--goal-radius");
        } else if (arg == "--step-size") {
            opt.step_size = parse_float(require_value("--step-size"), "--step-size");
        } else if (arg == "--tactical-radius") {
            opt.tactical_radius = parse_float(require_value("--tactical-radius"), "--tactical-radius");
        } else if (arg == "--sense-radius") {
            opt.sense_radius = parse_float(require_value("--sense-radius"), "--sense-radius");
        } else if (arg == "--agent-radius") {
            opt.agent_radius = parse_float(require_value("--agent-radius"), "--agent-radius");
        } else if (arg == "--base-move-resolve") {
            const std::string mode = require_value("--base-move-resolve");
            if (mode == "on") {
                opt.base_move_collision_resolve = true;
            } else if (mode == "off") {
                opt.base_move_collision_resolve = false;
            } else {
                fail_usage("Invalid value for --base-move-resolve. Use on or off.");
            }
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

    if (opt.obs_dim <= 0 || opt.iterations <= 0 || opt.warmup < 0 || opt.crowd_agents <= 0 || opt.crowd_dt <= 0.0f ||
        opt.arrival_runs <= 0 ||
        opt.arrival_steps <= 0 || opt.goal_radius <= 0.0f || opt.step_size <= 0.0f ||
        opt.tactical_radius <= 0.0f || opt.sense_radius <= 0.0f || opt.agent_radius <= 0.0f) {
        fail_usage("--obs-dim, --iterations, --crowd-agents, --crowd-dt, --arrival-runs, --arrival-steps and radius/step values must be positive; --warmup must be >= 0");
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

void print_arrival_stats(const std::string& name, const ArrivalStats& stats) {
    std::cout << std::left << std::setw(18) << name
              << " arrived=" << stats.arrived << "/" << stats.agents
              << " collision_free=" << stats.collision_free_arrived << "/" << stats.agents
              << " collided_agents=" << stats.collided_agents
              << " collision_events=" << stats.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << stats.avg_final_goal_dist
              << " max_final_dist=" << std::setprecision(3) << stats.max_final_goal_dist
              << " steps=" << stats.steps_run << "\n";
}

struct ArrivalAggregateStats {
    int runs = 0;
    double avg_arrived = 0.0;
    double avg_collision_free_arrived = 0.0;
    double avg_collided_agents = 0.0;
    double avg_collision_events = 0.0;
    double avg_final_goal_dist = 0.0;
    double avg_max_final_goal_dist = 0.0;
    double avg_steps_run = 0.0;
    ArrivalStats best{};
    ArrivalStats worst{};
};

void print_arrival_best_worst_compare(
    const ArrivalAggregateStats& base_stats,
    const ArrivalAggregateStats& detour_stats,
    int agents) {
    std::cout << "best_compare\n";
    std::cout << "  base_move"
              << " arrived=" << base_stats.best.arrived << "/" << agents
              << " collision_free=" << base_stats.best.collision_free_arrived << "/" << agents
              << " collision_events=" << base_stats.best.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << base_stats.best.avg_final_goal_dist
              << " steps=" << base_stats.best.steps_run << "\n";
    std::cout << "  detour"
              << " arrived=" << detour_stats.best.arrived << "/" << agents
              << " collision_free=" << detour_stats.best.collision_free_arrived << "/" << agents
              << " collision_events=" << detour_stats.best.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << detour_stats.best.avg_final_goal_dist
              << " steps=" << detour_stats.best.steps_run << "\n";

    std::cout << "worst_compare\n";
    std::cout << "  base_move"
              << " arrived=" << base_stats.worst.arrived << "/" << agents
              << " collision_free=" << base_stats.worst.collision_free_arrived << "/" << agents
              << " collision_events=" << base_stats.worst.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << base_stats.worst.avg_final_goal_dist
              << " steps=" << base_stats.worst.steps_run << "\n";
    std::cout << "  detour"
              << " arrived=" << detour_stats.worst.arrived << "/" << agents
              << " collision_free=" << detour_stats.worst.collision_free_arrived << "/" << agents
              << " collision_events=" << detour_stats.worst.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << detour_stats.worst.avg_final_goal_dist
              << " steps=" << detour_stats.worst.steps_run << "\n";
}

ArrivalAggregateStats aggregate_arrival_stats(const std::vector<ArrivalStats>& runs) {
    ArrivalAggregateStats out;
    if (runs.empty()) {
        return out;
    }
    out.runs = static_cast<int>(runs.size());
    out.best = runs.front();
    out.worst = runs.front();
    for (const ArrivalStats& run : runs) {
        out.avg_arrived += run.arrived;
        out.avg_collision_free_arrived += run.collision_free_arrived;
        out.avg_collided_agents += run.collided_agents;
        out.avg_collision_events += run.collision_events;
        out.avg_final_goal_dist += run.avg_final_goal_dist;
        out.avg_max_final_goal_dist += run.max_final_goal_dist;
        out.avg_steps_run += run.steps_run;
        if (run.arrived > out.best.arrived || (run.arrived == out.best.arrived && run.collision_events < out.best.collision_events)) {
            out.best = run;
        }
        if (run.arrived < out.worst.arrived || (run.arrived == out.worst.arrived && run.collision_events > out.worst.collision_events)) {
            out.worst = run;
        }
    }
    const double denom = static_cast<double>(runs.size());
    out.avg_arrived /= denom;
    out.avg_collision_free_arrived /= denom;
    out.avg_collided_agents /= denom;
    out.avg_collision_events /= denom;
    out.avg_final_goal_dist /= denom;
    out.avg_max_final_goal_dist /= denom;
    out.avg_steps_run /= denom;
    return out;
}

void print_arrival_aggregate_stats(const std::string& name, const ArrivalAggregateStats& stats, int agents) {
    std::cout << std::left << std::setw(18) << name
              << " runs=" << stats.runs
              << " avg_arrived=" << std::fixed << std::setprecision(2) << stats.avg_arrived << "/" << agents
              << " avg_collision_free=" << std::setprecision(2) << stats.avg_collision_free_arrived << "/" << agents
              << " avg_collided=" << std::setprecision(2) << stats.avg_collided_agents
              << " avg_events=" << std::setprecision(2) << stats.avg_collision_events
              << " avg_final_dist=" << std::setprecision(3) << stats.avg_final_goal_dist
              << " avg_steps=" << std::setprecision(2) << stats.avg_steps_run << "\n";
    std::cout << "  best: arrived=" << stats.best.arrived << "/" << agents
              << " collision_free=" << stats.best.collision_free_arrived << "/" << agents
              << " collision_events=" << stats.best.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << stats.best.avg_final_goal_dist
              << " steps=" << stats.best.steps_run << "\n";
    std::cout << "  worst: arrived=" << stats.worst.arrived << "/" << agents
              << " collision_free=" << stats.worst.collision_free_arrived << "/" << agents
              << " collision_events=" << stats.worst.collision_events
              << " avg_final_dist=" << std::fixed << std::setprecision(3) << stats.worst.avg_final_goal_dist
              << " steps=" << stats.worst.steps_run << "\n";
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
                  << "arrival_steps=" << opt.arrival_steps
                  << " arrival_runs=" << opt.arrival_runs
                  << " goal_radius=" << opt.goal_radius
                  << " step_size=" << opt.step_size
                  << " tactical_radius=" << opt.tactical_radius
                  << " sense_radius=" << opt.sense_radius
                  << " agent_radius=" << opt.agent_radius << "\n"
                  << "base_move_resolve=" << (opt.base_move_collision_resolve ? "on" : "off") << "\n"
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
        std::cout << "\nArrival comparison\n";
        ArrivalBenchmarkOptions arrival_options;
        arrival_options.onnx_path = opt.onnx_path;
        arrival_options.navmesh_path = opt.navmesh_path;
        arrival_options.agents = opt.crowd_agents;
        arrival_options.max_steps = opt.arrival_steps;
        arrival_options.obs_dim = opt.obs_dim;
        arrival_options.dt = opt.crowd_dt;
        arrival_options.step_size = opt.step_size * kEngineToDetourScale;
        arrival_options.tactical_target_radius = opt.tactical_radius * kEngineToDetourScale;
        arrival_options.sense_radius = opt.sense_radius * kEngineToDetourScale;
        arrival_options.goal_radius = opt.goal_radius * kEngineToDetourScale;
        arrival_options.agent_radius = opt.agent_radius * kEngineToDetourScale;
        arrival_options.base_move_collision_resolve = opt.base_move_collision_resolve;
        arrival_options.shared_goal_x = opt.detour_goal.x;
        arrival_options.shared_goal_y = opt.detour_goal.y;
        arrival_options.shared_goal_z = opt.detour_goal.z;

        const bool can_run_base = is_arrival_base_move_enabled();
        const bool can_run_detour = is_detour_crowd_enabled();

        if (!can_run_base) {
            std::cout << "base_move_arrival skipped: define HAS_ONNXRUNTIME and set ONNXRUNTIME_DIR\n";
        }
        if (!can_run_detour) {
            std::cout << "detour_arrival    skipped: DetourCrowd.h/DetourCrowd.lib are not available\n";
        }
        if (can_run_base && can_run_detour) {
            std::vector<ArrivalStats> base_runs;
            std::vector<ArrivalStats> detour_runs;
            base_runs.reserve(static_cast<std::size_t>(opt.arrival_runs));
            detour_runs.reserve(static_cast<std::size_t>(opt.arrival_runs));
            for (int run = 0; run < opt.arrival_runs; ++run) {
                arrival_options.run_index = run;
                base_runs.push_back(benchmark_base_move_arrival(arrival_options));
                detour_runs.push_back(benchmark_detour_crowd_arrival(arrival_options));
                std::cout << "run_" << run << "\n";
                print_arrival_stats("  base_move", base_runs.back());
                print_arrival_stats("  detour", detour_runs.back());
            }
            const ArrivalAggregateStats base_stats = aggregate_arrival_stats(base_runs);
            const ArrivalAggregateStats detour_stats = aggregate_arrival_stats(detour_runs);
            print_arrival_aggregate_stats("base_move_arrival", base_stats, opt.crowd_agents);
            print_arrival_aggregate_stats("detour_arrival", detour_stats, opt.crowd_agents);
            print_arrival_best_worst_compare(base_stats, detour_stats, opt.crowd_agents);
        } else if (can_run_base) {
            std::vector<ArrivalStats> runs;
            runs.reserve(static_cast<std::size_t>(opt.arrival_runs));
            for (int run = 0; run < opt.arrival_runs; ++run) {
                arrival_options.run_index = run;
                runs.push_back(benchmark_base_move_arrival(arrival_options));
                print_arrival_stats("base_move_run_" + std::to_string(run), runs.back());
            }
            print_arrival_aggregate_stats("base_move_arrival", aggregate_arrival_stats(runs), opt.crowd_agents);
        } else if (can_run_detour) {
            std::vector<ArrivalStats> runs;
            runs.reserve(static_cast<std::size_t>(opt.arrival_runs));
            for (int run = 0; run < opt.arrival_runs; ++run) {
                arrival_options.run_index = run;
                runs.push_back(benchmark_detour_crowd_arrival(arrival_options));
                print_arrival_stats("detour_run_" + std::to_string(run), runs.back());
            }
            print_arrival_aggregate_stats("detour_arrival", aggregate_arrival_stats(runs), opt.crowd_agents);
        }
        std::cout << "\nNote: detour_crowd is the relevant comparison for local collision avoidance. "
                     "detour_query is path/waypoint query cost.\n"
                     "Arrival comparison uses the same sampled spawn set and a shared goal from --detour-goal, then reports sense-radius entry success plus collision counts.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
