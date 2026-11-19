#include "distance.hpp"

#if !defined(VS_FORCE_SCALAR) && defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VS_NEON 1
#elif !defined(VS_FORCE_SCALAR) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define VS_AVX2 1
#endif

namespace vs {
double scalar_l2(const float* a, const float* b, std::size_t n) {
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double diff = static_cast<double>(a[i]) - b[i];
        sum += diff * diff;
    }
    return sum;
}

namespace {
#if VS_NEON
double neon_l2(const float* a, const float* b, std::size_t n) {
    auto lo = vdupq_n_f64(0), hi = lo;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const auto x = vld1q_f32(a + i), y = vld1q_f32(b + i);
        const auto d0 = vsubq_f64(vcvt_f64_f32(vget_low_f32(x)), vcvt_f64_f32(vget_low_f32(y)));
        const auto d1 = vsubq_f64(vcvt_high_f64_f32(x), vcvt_high_f64_f32(y));
        lo = vfmaq_f64(lo, d0, d0);
        hi = vfmaq_f64(hi, d1, d1);
    }
    return vaddvq_f64(vaddq_f64(lo, hi)) + scalar_l2(a + i, b + i, n - i);
}
#elif VS_AVX2
__attribute__((target("avx2")))
double avx2_l2(const float* a, const float* b, std::size_t n) {
    auto lo = _mm256_setzero_pd(), hi = lo;
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto d0 = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i)),
                                      _mm256_cvtps_pd(_mm_loadu_ps(b+i)));
        const auto d1 = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(a+i+4)),
                                      _mm256_cvtps_pd(_mm_loadu_ps(b+i+4)));
        lo = _mm256_add_pd(lo, _mm256_mul_pd(d0, d0));
        hi = _mm256_add_pd(hi, _mm256_mul_pd(d1, d1));
    }
    double lanes[4];
    _mm256_storeu_pd(lanes, _mm256_add_pd(lo, hi));
    return (lanes[0]+lanes[1]) + (lanes[2]+lanes[3]) + scalar_l2(a+i, b+i, n-i);
}
#endif
}

const DistanceKernel& distance_kernel() {
    static const DistanceKernel kernel = []() -> DistanceKernel {
#if VS_NEON
        return {"neon", neon_l2};
#elif VS_AVX2
        if (__builtin_cpu_supports("avx2")) return {"avx2", avx2_l2};
#endif
        return {"scalar", scalar_l2};
    }();
    return kernel;
}
}
