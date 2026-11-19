#include "vector_search.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
std::size_t positive(std::string_view text) {
    std::size_t n = 0;
    const auto [end, err] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (err != std::errc{} || end != text.data() + text.size() || !n)
        throw std::invalid_argument("k must be a positive integer");
    return n;
}

std::vector<float> parse_row(const std::string& line) {
    std::istringstream in(line);
    std::vector<float> row;
    std::string token;
    while (in >> token) {
        std::size_t consumed = 0;
        const auto value = std::stof(token, &consumed);
        if (consumed != token.size()) throw std::invalid_argument("invalid coordinate: " + token);
        row.push_back(value);
    }
    return row;
}
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: vector-search VECTORS.txt 'q1 q2 ...' [k=10]\n";
        return 2;
    }
    try {
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("cannot open vector file");
        std::vector<float> data;
        std::size_t dimension = 0, line_no = 0;
        std::string line;
        while (std::getline(file, line)) {
            ++line_no;
            const auto row = parse_row(line);
            if (row.empty()) continue;
            if (!dimension) dimension = row.size();
            if (row.size() != dimension)
                throw std::invalid_argument("dimension mismatch on line " + std::to_string(line_no));
            data.insert(data.end(), row.begin(), row.end());
        }
        if (file.bad()) throw std::runtime_error("error reading vector file");
        vs::Index index(std::move(data), dimension);
        const auto query = parse_row(argv[2]);
        if (query.size() != dimension) throw std::invalid_argument("query dimension mismatch");
        const auto hits = index.search(query, argc == 4 ? positive(argv[3]) : 10);
        std::cout << "id\tsquared_l2\n" << std::setprecision(17);
        for (const auto& hit : hits) std::cout << hit.id << '\t' << hit.distance << '\n';
    } catch (const std::exception& error) {
        std::cerr << "vector-search: " << error.what() << '\n';
        return 1;
    }
}
