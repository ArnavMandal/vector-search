#pragma once

#include <cstddef>

namespace vs {
using L2Kernel = double (*)(const float*, const float*, std::size_t);
struct DistanceKernel {
    const char* name;
    L2Kernel compute;
};

double scalar_l2(const float*, const float*, std::size_t);
const DistanceKernel& distance_kernel();
}
