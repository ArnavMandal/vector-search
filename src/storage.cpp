#include "hnsw.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unistd.h>

namespace vs {
namespace {
constexpr std::array<char, 8> v1 = {'V','S','I','D','X','0','1','\0'};
constexpr std::array<char, 8> v2 = {'V','S','I','D','X','0','2','\0'};
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
}

// Store the construction recipe and recreate the immutable accelerators on load.
void Index::save(const std::filesystem::path& path) const {
    std::string temp = path.string() + ".XXXXXX";
    const int fd = ::mkstemp(temp.data());
    if (fd < 0) throw std::runtime_error("cannot create temporary index file");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(::fdopen(fd, "wb"), &std::fclose);
    try {
        if (!file) {
            ::close(fd);
            throw std::runtime_error("cannot open temporary index file");
        }
        const auto& magic = graph_ ? v2 : v1;
        const std::uint64_t header[] = {size(), dim_, has_tree() ? 1u : 0u};
        if (std::fwrite(magic.data(), 1, magic.size(), file.get()) != magic.size() ||
            std::fwrite(header, sizeof(header), 1, file.get()) != 1)
            throw std::runtime_error("cannot write index header");
        if (graph_) {
            const std::uint64_t settings[] = {graph_->config.M, graph_->config.efConstruction};
            if (std::fwrite(settings, sizeof(settings), 1, file.get()) != 1)
                throw std::runtime_error("cannot write HNSW settings");
        }
        if (std::fwrite(vectors_.data(), sizeof(float), vectors_.size(), file.get()) != vectors_.size())
            throw std::runtime_error("cannot write index file");
        if (std::fclose(file.release()) != 0) throw std::runtime_error("cannot finish index file");
        std::filesystem::rename(temp, path);
    } catch (...) {
        ::unlink(temp.c_str());
        throw;
    }
}

Index Index::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open index file");
    const auto length = file.tellg();
    file.seekg(0);
    std::array<char, v1.size()> tag{};
    std::uint64_t header[3]{};
    file.read(tag.data(), tag.size());
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || (tag != v1 && tag != v2)) throw std::runtime_error("invalid index header");
    const auto [rows, dims, tree] = header;
    std::optional<HnswConfig> hnsw;
    if (tag == v2) {
        std::uint64_t settings[2]{};
        file.read(reinterpret_cast<char*>(settings), sizeof(settings));
        if (!file || settings[0] < 2 || settings[0] > 256 || settings[1] < settings[0] ||
            settings[1] > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("invalid HNSW settings");
        hnsw = HnswConfig{static_cast<std::size_t>(settings[0]), static_cast<std::size_t>(settings[1])};
    }
    const auto remaining = length - file.tellg();
    if (!rows || !dims || tree > 1 || rows > std::vector<float>().max_size() / dims ||
        remaining < 0 || static_cast<std::uint64_t>(remaining) != rows * dims * sizeof(float))
        throw std::runtime_error("invalid index dimensions or length");
    std::vector<float> data(rows * dims);
    if (!file.read(reinterpret_cast<char*>(data.data()), remaining)) throw std::runtime_error("cannot read index");
    return Index(std::move(data), dims, tree != 0, hnsw);
}
}
