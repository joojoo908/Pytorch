#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct OnnxRunResult {
    std::string input_name;
    std::string output_name;
    std::vector<float> input_values;
    std::vector<std::int64_t> input_shape;
    std::vector<float> output_values;
    std::vector<std::int64_t> output_shape;
};

bool is_onnx_runtime_enabled();

OnnxRunResult run_onnx_float_model(
    const std::string& model_path,
    const std::vector<float>& input_values,
    const std::vector<std::int64_t>& input_shape);

OnnxRunResult run_base_move_example(const std::string& model_path);
