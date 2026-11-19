#pragma once

#include <cstddef>
#include <compare>
#include <filesystem>
#include <span>
#include <optional>
#include <memory>
#include <utility>
#include <vector>

namespace vs {

struct Neighbor {
    double distance;  // Squared Euclidean distance.
    std::size_t id;   // Input row, starting at zero.
    auto operator<=>(const Neighbor&) const = default;
};

struct HnswConfig {
    std::size_t M = 16;
    std::size_t efConstruction = 100;
};
class HnswGraph;

class Index {
public:
    Index(std::vector<float> vectors, std::size_t dimension, bool build_tree = false,
          std::optional<HnswConfig> hnsw = std::nullopt);
    std::size_t size() const { return vectors_.size() / dim_; }
    std::size_t dimension() const { return dim_; }
    std::size_t storage_bytes() const;
    bool has_tree() const { return !order_.empty(); }

    bool has_hnsw() const { return static_cast<bool>(graph_); }
    // A positive efSearch selects HNSW, a positive budget selects the tree.
    // Both zero selects exact search.
    // Returns min(k, size()) neighbors per query, in query-major order.
    std::vector<Neighbor> search(std::span<const float> queries, std::size_t k = 10,
                                 std::size_t budget = 0, std::size_t threads = 1,
                                 std::size_t efSearch = 0) const;
    void save(const std::filesystem::path& path) const;
    static Index load(const std::filesystem::path& path);

private:
    std::shared_ptr<const HnswGraph> graph_;
    std::size_t dim_;
    std::vector<float> vectors_;
    std::vector<std::size_t> order_, axes_;
    void build(std::size_t begin, std::size_t end);
    void search_one(const float* query, std::span<Neighbor> out, std::size_t budget) const;
};

}
