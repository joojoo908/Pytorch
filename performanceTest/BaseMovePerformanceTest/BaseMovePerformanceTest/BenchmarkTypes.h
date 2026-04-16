#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct Metrics {
    double total_ms = 0.0;
    double avg_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
};

inline Metrics compute_metrics(std::vector<double> samples_us, double total_ms) {
    Metrics out;
    out.total_ms = total_ms;
    out.avg_us = (total_ms * 1000.0) / static_cast<double>(std::max<std::size_t>(1, samples_us.size()));
    if (samples_us.empty()) {
        return out;
    }

    std::sort(samples_us.begin(), samples_us.end());
    auto percentile = [&](double p) {
        const double idx = p * static_cast<double>(samples_us.size() - 1);
        return samples_us[static_cast<std::size_t>(std::llround(idx))];
    };
    out.p50_us = percentile(0.50);
    out.p95_us = percentile(0.95);
    out.p99_us = percentile(0.99);
    return out;
}
