#include "vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

void check(bool ok) {
    if (!ok) throw std::runtime_error("engine test failed");
}

int main() {
    std::mt19937 rng(81);
    std::uniform_int_distribution<> coord(-3, 3);
    for (std::size_t n : {1, 10, 16, 17, 79, 256}) {
        const std::size_t d = 5;
        std::vector<float> rows(n * d), queries(21 * d);
        for (auto& x : rows) x = coord(rng);
        for (auto& x : queries) x = coord(rng);
        vs::Index index(rows, d, true);
        for (std::size_t k : {1, 10, 350}) {
            const auto width = std::min(k, n);
            std::vector<vs::Neighbor> truth;
            for (std::size_t q = 0; q < queries.size(); q += d) {
                std::vector<vs::Neighbor> scores;
                for (std::size_t i = 0; i < n; ++i) {
                    double s = 0;
                    for (std::size_t j = 0; j < d; ++j) {
                        const double v = queries[q + j] - rows[i * d + j];
                        s += v * v;
                    }
                    scores.push_back({s, i});
                }
                std::sort(scores.begin(), scores.end());
                truth.insert(truth.end(), scores.begin(), scores.begin() + width);
            }
            check(index.search(queries, k) == truth);
            check(index.search(queries, k, n, 4) == truth);
            check(index.search(queries, k, width, 3) == index.search(queries, k, width));
        }
    }
    const float big = std::numeric_limits<float>::max();
    const auto hits = vs::Index({big, -big}, 1).search(std::vector<float>{big});
    check(hits[0].distance == 0 && std::isfinite(hits[1].distance));
    std::cout << "C++ oracle and batch tests passed\n";
}
