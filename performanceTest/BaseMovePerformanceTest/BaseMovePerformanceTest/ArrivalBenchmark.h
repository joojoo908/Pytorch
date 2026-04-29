#pragma once

#include <string>

struct ArrivalBenchmarkOptions {
    std::string onnx_path;
    std::string navmesh_path;
    int agents = 20;
    int run_index = 0;
    int max_steps = 128;
    int obs_dim = 24;
    float dt = 1.0f / 60.0f;
    float step_size = 120.0f;
    float tactical_target_radius = 600.0f;
    float sense_radius = 600.0f;
    float goal_radius = 120.0f;
    float agent_radius = 30.0f;
    bool base_move_collision_resolve = true;
    float shared_goal_x = 0.0f;
    float shared_goal_y = 0.0f;
    float shared_goal_z = 0.0f;
};

struct ArrivalStats {
    int agents = 0;
    int steps_run = 0;
    int arrived = 0;
    int collision_free_arrived = 0;
    int collided_agents = 0;
    int collision_events = 0;
    float avg_final_goal_dist = 0.0f;
    float max_final_goal_dist = 0.0f;
};

bool is_arrival_base_move_enabled();

ArrivalStats benchmark_base_move_arrival(const ArrivalBenchmarkOptions& options);
ArrivalStats benchmark_detour_crowd_arrival(const ArrivalBenchmarkOptions& options);
