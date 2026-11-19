#pragma once

#include "vector_search.hpp"
#include "distance.hpp"
#include <cstdint>

namespace vs {
class HnswGraph {
public:
    HnswGraph(std::span<const float> rows, std::size_t dim, HnswConfig config);
    const HnswConfig config;
    std::size_t storage_bytes() const;
    std::vector<Neighbor> search(const float* rows, std::size_t dim, const float* query,
                                 std::size_t ef) const;

private:
    using Id = std::uint32_t;
    struct Visits {
        std::vector<Id> marks;
        Id epoch = 0;
        void reset(std::size_t n);
        bool first(Id id);
    };
    struct Space {
        const float* rows;
        std::size_t dim;
        L2Kernel l2;
        const float* row(Id id) const { return rows + id * dim; }
        Neighbor score(const float* q, Id id) const { return {l2(q, row(id), dim), id}; }
    };

    // Each row has 2M slots at the base and M slots per additional level.
    // Offsets delimit rows. An unused slot marks the end of a neighbor list.
    std::vector<std::size_t> offsets_;
    std::vector<Id> links_;
    Id entry_ = 0;
    unsigned top_ = 0;
    std::size_t start(Id id, unsigned level) const;
    std::size_t capacity(unsigned level) const;
    Neighbor descend(const Space&, const float*, Neighbor, unsigned level) const;
    std::vector<Neighbor> explore(const Space&, const float*, Neighbor, unsigned level,
                                  std::size_t ef, Visits&) const;
    std::vector<Id> select(const Space&, std::span<const Neighbor>, std::size_t limit) const;
    void connect(const Space&, Id from, Id to, unsigned level);
};
}
