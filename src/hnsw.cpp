#include "hnsw.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace vs {
namespace {
constexpr auto unused = std::numeric_limits<std::uint32_t>::max();
}

void HnswGraph::Visits::reset(std::size_t n) {
    if (marks.size() < n) marks.resize(n);
    if (++epoch == 0) {
        std::fill(marks.begin(), marks.end(), 0);
        epoch = 1;
    }
}
bool HnswGraph::Visits::first(Id id) {
    if (marks[id] == epoch) return false;
    marks[id] = epoch;
    return true;
}

std::size_t HnswGraph::capacity(unsigned level) const {
    return level == 0 ? 2 * config.M : config.M;
}
std::size_t HnswGraph::start(Id id, unsigned level) const {
    return offsets_[id] + (level == 0 ? 0 : (level + 1) * config.M);
}

template<class Score>
Neighbor HnswGraph::descend(const Score& score, Neighbor best, unsigned level) const {
    for (;;) {
        const auto current = best;
        for (auto id : std::span(links_).subspan(start(current.id, level), capacity(level))) {
            if (id == unused) break;
            best = std::min(best, score(current.id, id));
        }
        if (best == current) return best;
    }
}

template<class Score>
std::vector<Neighbor> HnswGraph::explore(const Score& score, Neighbor entry,
                                         unsigned level, std::size_t ef, Visits& visited) const {
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<>> frontier;
    std::vector<Neighbor> best{entry}; // max-heap, root is the current rejection threshold
    visited.reset(offsets_.size() - 1);
    visited.first(entry.id);
    frontier.push(entry);
    while (!frontier.empty()) {
        const auto next = frontier.top();
        if (best.size() == ef && next > best.front()) break;
        frontier.pop();
        for (auto id : std::span(links_).subspan(start(next.id, level), capacity(level))) {
            if (id == unused) break;
            if (!visited.first(id)) continue;
            const auto candidate = score(next.id, id);
            if (best.size() == ef && candidate >= best.front()) continue;
            frontier.push(candidate);
            if (best.size() == ef) {
                std::pop_heap(best.begin(), best.end());
                best.pop_back();
            }
            best.push_back(candidate);
            std::push_heap(best.begin(), best.end());
        }
    }
    std::sort_heap(best.begin(), best.end());
    return best;
}

// A closer selected neighbor can provide the route to a redundant candidate.
std::vector<HnswGraph::Id> HnswGraph::select(const Space& space, std::span<const Neighbor> sorted,
                                             std::size_t limit) const {
    std::vector<Id> chosen;
    for (const auto& candidate : sorted) {
        const auto* row = space.row(candidate.id);
        auto occludes = [&](Id other) { return space.score(row, other).distance < candidate.distance; };
        if (std::none_of(chosen.begin(), chosen.end(), occludes)) {
            chosen.push_back(candidate.id);
            if (chosen.size() == limit) break;
        }
    }
    return chosen;
}

void HnswGraph::connect(const Space& space, Id from, Id to, unsigned level) {
    auto slots = std::span(links_).subspan(start(from, level), capacity(level));
    const auto free = std::find(slots.begin(), slots.end(), unused);
    if (free != slots.end()) {
        *free = to;
        return;
    }
    std::vector<Neighbor> options;
    for (const auto id : slots) options.push_back(space.score(space.row(from), id));
    options.push_back(space.score(space.row(from), to));
    std::sort(options.begin(), options.end());
    const auto keep = select(space, options, slots.size());
    std::fill(std::copy(keep.begin(), keep.end(), slots.begin()), slots.end(), unused);
}

HnswGraph::HnswGraph(std::span<const float> rows, std::size_t dim, HnswConfig settings)
    : config(settings) {
    if (config.M < 2 || config.M > 256 || config.efConstruction < config.M)
        throw std::invalid_argument("HNSW requires 2 <= M <= 256 and efConstruction >= M");
    const auto n = rows.size() / dim;
    if (n > unused) throw std::length_error("HNSW requires 32-bit row IDs");
    std::mt19937 rng(42);
    std::vector<unsigned> levels(n);
    offsets_.reserve(n + 1);
    offsets_.push_back(0);
    for (auto& level : levels) {
        // Approximately 1/M of a layer's nodes are promoted to the next layer.
        while (rng() < (std::uint64_t{1} << 32) / config.M) ++level;
        const auto slots = (level + 2) * config.M;
        if (slots > links_.max_size() - offsets_.back()) throw std::length_error("HNSW graph too large");
        offsets_.push_back(offsets_.back() + slots);
    }
    links_.assign(offsets_.back(), unused);
    const Space space{rows.data(), dim, distance_kernel().compute};
    top_ = levels[0];
    Visits visited;
    for (std::size_t i = 1; i < n; ++i) {
        const auto* q = space.row(i);
        const auto score = [&](Id, Id id) { return space.score(q, id); };
        auto entry = space.score(q, entry_);
        for (auto l = top_; l > levels[i]; --l) entry = descend(score, entry, l);
        for (int l = static_cast<int>(std::min(top_, levels[i])); l >= 0; --l) {
            const auto candidates = explore(score, entry, l, config.efConstruction, visited);
            const auto keep = select(space, candidates, config.M);
            std::copy(keep.begin(), keep.end(), links_.begin() + start(i, l));
            for (auto id : keep) connect(space, id, i, l);
            entry = candidates.front();
        }
        if (levels[i] > top_) {
            top_ = levels[i];
            entry_ = i;
        }
    }
}

template<class Score>
std::vector<Neighbor> HnswGraph::search_layers(const Score& score, std::size_t ef, Visits& visited) const {
    auto entry = score(entry_, entry_);
    for (auto l = top_; l > 0; --l) entry = descend(score, entry, l);
    return explore(score, entry, 0, ef, visited);
}

std::vector<Neighbor> HnswGraph::search(const float* rows, std::size_t dim, const float* query,
                                        std::size_t ef, HnswTrace* trace) const {
    const Space space{rows, dim, distance_kernel().compute};
    const auto score = [&](Id, Id id) { return space.score(query, id); };
    thread_local Visits visited;
    if (!trace) return search_layers(score, ef, visited);

    // The observer spans every layer. Re-scoring a row does not count it twice.
    std::unordered_set<Id> seen;
    const auto record = [&](Id source, Id id) {
        if (seen.insert(id).second) {
            trace->examined.push_back(id);
            if (source != id) trace->discovery_edges.emplace_back(source, id);
        }
        return score(source, id);
    };
    return search_layers(record, ef, visited);
}

std::size_t HnswGraph::storage_bytes() const {
    return offsets_.capacity() * sizeof(std::size_t) + links_.capacity() * sizeof(Id);
}
}
