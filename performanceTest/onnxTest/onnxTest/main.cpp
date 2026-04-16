#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "OnnxRunner.h"

namespace {

std::string default_model_path() {
    return R"(base_move.onnx)";
}

void print_shape(const std::vector<std::int64_t>& shape) {
    std::cout << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << shape[i];
    }
    std::cout << "]";
}

void print_values(const std::vector<float>& values, std::size_t max_count) {
    std::cout << "[";
    const std::size_t count = std::min(values.size(), max_count);
    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(6) << values[i];
    }
    if (values.size() > max_count) {
        std::cout << ", ...";
    }
    std::cout << "]";
}

std::string parse_model_path(int argc, char** argv) {
    if (argc <= 1) {
        return default_model_path();
    }
    if (argc == 3 && std::string(argv[1]) == "--model") {
        return argv[2];
    }

    throw std::runtime_error("Usage: onnxTest.exe [--model <path-to-onnx>]");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string model_path = parse_model_path(argc, argv);

        if (!is_onnx_runtime_enabled()) {
            std::cout << "ONNX Runtime is disabled.\n"
                      << "Set ONNXRUNTIME_DIR to the ONNX Runtime C++ SDK root and rebuild.\n";
            return 1;
        }

        const OnnxRunResult result = run_base_move_example(model_path);

        std::cout << "model: " << model_path << "\n";
        std::cout << "input name: " << result.input_name << "\n";
        std::cout << "input shape: ";
        print_shape(result.input_shape);
        std::cout << "\ninput values: ";
        print_values(result.input_values, 24);
        std::cout << "\n";

        std::cout << "output name: " << result.output_name << "\n";
        std::cout << "output shape: ";
        print_shape(result.output_shape);
        std::cout << "\noutput values: ";
        print_values(result.output_values, 16);
        std::cout << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
