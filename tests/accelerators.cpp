#include "distance.hpp"
#include "vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void test_distances() {
    std::mt19937 rng(1729);
    std::uniform_real_distribution<float> val(-25.0f, 25.0f);
    for (std::size_t d : {1, 2, 3, 4, 5, 7, 8, 9, 17, 32, 127, 128, 384, 385}) {
        std::vector<float> a(d + 1), b(d + 2);
        for (auto& x : a) x = val(rng);
        for (auto& x : b) x = val(rng);
        const auto* x = a.data() + 1;
        const auto* y = b.data() + 2;
        const double ref = vs::scalar_l2(x, y, d);
        const auto kernel = vs::distance_kernel().compute;
        require(std::abs(kernel(x, y, d) - ref) <= 1e-13 * ref, "SIMD/scalar mismatch");
        require(kernel(x, x, d) == 0.0, "self distance");
        std::fill(a.begin(), a.end(), std::numeric_limits<float>::max());
        std::fill(b.begin(), b.end(), -std::numeric_limits<float>::max());
        const double huge = vs::scalar_l2(x, y, d);
        require(std::abs(kernel(x, y, d) - huge) <= 1e-13 * huge, "overflow before widening");
    }
}

void test_graph() {
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> val(-1, 1);
    std::vector<float> rows(200 * 13), q(30 * 13);
    for (auto& x : rows) x = val(rng);
    for (auto& x : q) x = val(rng);
    vs::Index graph(rows, 13, true, vs::HnswConfig{12, 80});
    const auto exact = graph.search(q, 10);
    const auto hits = graph.search(q, 10, 0, 3, 64);
    require(hits == graph.search(q, 10, 0, 1, 64), "parallel graph results differ");
    std::size_t found = 0;
    for (std::size_t i = 0; i < 30; ++i) {
        auto group = std::span(hits).subspan(10*i, 10);
        auto truth = std::span(exact).subspan(10*i, 10);
        require(std::is_sorted(group.begin(), group.end()), "unordered results");
        for (const auto& hit : group) {
            require(hit.id < graph.size(), "invalid row ID");
            require(std::count(group.begin(), group.end(), hit) == 1, "duplicate graph result");
            const double ref = vs::scalar_l2(q.data() + 13*i, rows.data() + 13*hit.id, 13);
            require(std::abs(hit.distance-ref) <= 1e-13 * ref, "invalid graph score");
            found += std::find(truth.begin(), truth.end(), hit) != truth.end();
        }
    }
    require(found >= 285, "graph recall regression");
    vs::Index tiny({2, 7}, 2, false, vs::HnswConfig{});
    auto one = tiny.search(std::vector<float>{2, 7}, 10, 0, 1, 1);
    require(one.size() == 1 && one.front().distance == 0, "single-row graph");
}
}

int main() {
    test_distances();
    test_graph();
    std::cout << "Distance (" << vs::distance_kernel().name << ") and HNSW tests passed\n";
}
