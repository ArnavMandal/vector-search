#include "vector_search.hpp"
#include "hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace vs {
namespace {
constexpr std::size_t leaf_size = 16;

void validate_finite(std::span<const float> values) {
    if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
        throw std::invalid_argument("coordinates must be finite");
}
}

Index::Index(std::vector<float> vectors, std::size_t dimension, bool build_tree,
             std::optional<HnswConfig> hnsw)
    : dim_(dimension), vectors_(std::move(vectors)) {
    if (!dim_ || vectors_.empty() || vectors_.size() % dim_)
        throw std::invalid_argument("expected a nonempty matrix with fixed positive dimension");
    validate_finite(vectors_);
    if (hnsw) graph_ = std::make_shared<HnswGraph>(vectors_, dim_, *hnsw);
    if (build_tree) {
        order_.resize(size());
        axes_.resize(size());
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        build(0, size());
    }
}

void Index::build(std::size_t begin, std::size_t end) {
    if (end - begin <= leaf_size) {
        std::sort(order_.begin() + begin, order_.begin() + end);
        return;
    }
    std::size_t axis = 0;
    double widest = -1;
    for (std::size_t d = 0; d < dim_; ++d) {
        float lo = vectors_[order_[begin] * dim_ + d], hi = lo;
        for (auto i = begin + 1; i < end; ++i) {
            const float x = vectors_[order_[i] * dim_ + d];
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        const double width = static_cast<double>(hi) - lo;
        if (width > widest) {
            widest = width;
            axis = d;
        }
    }
    const auto mid = begin + (end - begin) / 2;
    auto less = [&](std::size_t a, std::size_t b) {
        return std::tie(vectors_[a * dim_ + axis], a) < std::tie(vectors_[b * dim_ + axis], b);
    };
    std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end, less);
    axes_[mid] = axis;
    build(begin, mid);
    build(mid + 1, end);
}

std::size_t Index::storage_bytes() const {
    return (graph_ ? graph_->storage_bytes() : 0) + vectors_.capacity() * sizeof(float) + (order_.capacity() + axes_.capacity()) * sizeof(std::size_t);
}

void Index::search_one(const float* query, std::span<Neighbor> out, std::size_t budget) const {
    // Keep the worst accepted neighbor at the root of a bounded max-heap.
    std::size_t used = 0;
    const auto l2 = distance_kernel().compute;
    auto visit = [&](std::size_t id) {
        const float* row = vectors_.data() + id * dim_;
        const Neighbor item{l2(row, query, dim_), id};
        if (used < out.size()) {
            out[used++] = item;
        } else {
            if (item >= out.front()) return;
            std::pop_heap(out.begin(), out.end());
            out.back() = item;
        }
        std::push_heap(out.begin(), out.begin() + used);
    };

    if (!budget) {
        for (std::size_t id = 0; id < size(); ++id) visit(id);
    } else {
        // The far child's splitting plane gives a lower bound for that subtree.
        using Branch = std::tuple<double, std::size_t, std::size_t>;
        std::priority_queue<Branch, std::vector<Branch>, std::greater<>> branches;
        branches.emplace(0.0, 0, size());
        while (budget && !branches.empty()) {
            auto [bound, begin, end] = branches.top();
            branches.pop();
            if (used == out.size() && bound > out.front().distance) break;
            if (end - begin <= leaf_size) {
                for (auto i = begin; i < end && budget; ++i, --budget) visit(order_[i]);
                continue;
            }
            const auto mid = begin + (end - begin) / 2;
            const auto pivot = order_[mid], axis = axes_[mid];
            visit(pivot);
            --budget;
            const double delta = static_cast<double>(query[axis]) - vectors_[pivot * dim_ + axis];
            const double far_bound = std::max(bound, delta * delta);
            branches.emplace(delta <= 0 ? bound : far_bound, begin, mid);
            branches.emplace(delta <= 0 ? far_bound : bound, mid + 1, end);
        }
    }
    std::sort_heap(out.begin(), out.end());
}

void Index::search_hnsw_one(const float* query, std::span<Neighbor> out, std::size_t ef,
                            HnswTrace* trace) const {
    const auto best = graph_->search(vectors_.data(), dim_, query, ef, trace);
    if (best.size() >= out.size()) {
        std::copy_n(best.begin(), out.size(), out.begin());
        return;
    }
    // Pruned edges can leave fewer than k reachable rows. Keep the graph trace
    // separate from the exact scan used to meet the result-count contract.
    if (trace) trace->exact_fallback = true;
    search_one(query, out, 0);
}

HnswTrace Index::trace_hnsw(std::span<const float> query, std::size_t k, std::size_t efSearch) const {
    if (query.size() != dim_) throw std::invalid_argument("trace requires one query of index dimension");
    k = std::min(k, size());
    if (!graph_ || !k || efSearch < k)
        throw std::invalid_argument("trace requires a graph and efSearch >= result count > 0");
    validate_finite(query);
    HnswTrace trace;
    trace.neighbors.resize(k);
    search_hnsw_one(query.data(), trace.neighbors, efSearch, &trace);
    return trace;
}

std::vector<Neighbor> Index::search(std::span<const float> queries, std::size_t k,
                                    std::size_t budget, std::size_t threads, std::size_t efSearch) const {
    if (queries.size() % dim_) throw std::invalid_argument("query dimension mismatch");
    if (!threads || !k) throw std::invalid_argument("k and threads must be positive");
    k = std::min(k, size());
    if (budget && (!has_tree() || budget < k))
        throw std::invalid_argument("budgeted search requires a tree and budget >= result count");
    if (efSearch && (!graph_ || budget || efSearch < k))
        throw std::invalid_argument("HNSW search requires a graph, zero tree budget and efSearch >= result count");
    validate_finite(queries);
    const auto count = queries.size() / dim_;
    if (!count) return {};
    if (count > std::vector<Neighbor>().max_size() / k) throw std::length_error("too many results");
    std::vector<Neighbor> result(count * k);
    auto run = [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
            const float* query = queries.data() + i * dim_;
            auto out = std::span(result).subspan(i * k, k);
            if (efSearch) search_hnsw_one(query, out, efSearch, nullptr);
            else search_one(query, out, budget);
        }
    };
    threads = std::min(threads, count);
    std::vector<std::future<void>> workers;
    const auto chunk = count / threads;
    auto boundary = [&](std::size_t t) { return t * chunk + std::min(t, count % threads); };
    for (std::size_t t = 1; t < threads; ++t)
        workers.push_back(std::async(std::launch::async, run, boundary(t), boundary(t + 1)));
    run(0, boundary(1));
    for (auto& worker : workers) worker.get();
    return result;
}
}
