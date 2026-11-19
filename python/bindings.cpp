#include "vector_search.hpp"
#include "distance.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;
using Matrix = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {
vs::Index build(const Matrix& rows, bool tree, bool hnsw, std::size_t M, std::size_t efConstruction) {
    if (rows.ndim() != 2) throw std::invalid_argument("vectors must be a 2-D matrix");
    std::vector<float> data(rows.data(), rows.data() + rows.size());
    const auto dim = rows.shape(1);
    py::gil_scoped_release release;
    return vs::Index(std::move(data), dim, tree, hnsw ? std::optional(vs::HnswConfig{M, efConstruction}) : std::nullopt);
}

py::tuple search(const vs::Index& index, const Matrix& rows, std::size_t k,
                 std::size_t budget, std::size_t threads, std::size_t efSearch) {
    if (rows.ndim() != 2 || static_cast<std::size_t>(rows.shape(1)) != index.dimension())
        throw std::invalid_argument("queries must have shape (query_count, index.dimension)");
    const auto width = static_cast<py::ssize_t>(std::min(k, index.size()));
    std::vector<float> queries(rows.data(), rows.data() + rows.size());
    std::vector<vs::Neighbor> hits;
    {
        py::gil_scoped_release release;
        hits = index.search(queries, k, budget, threads, efSearch);
    }
    py::array_t<std::uint64_t> ids({rows.shape(0), width});
    py::array_t<double> distances({rows.shape(0), width});
    auto* id = ids.mutable_data();
    auto* distance = distances.mutable_data();
    for (const auto& hit : hits) {
        *id++ = hit.id;
        *distance++ = hit.distance;
    }
    return py::make_tuple(std::move(ids), std::move(distances));
}

py::dict trace_hnsw(const vs::Index& index, const Matrix& query, std::size_t k, std::size_t ef) {
    if (query.ndim() != 1) throw std::invalid_argument("trace query must be a 1-D vector");
    std::vector<float> point(query.data(), query.data() + query.size());
    vs::HnswTrace trace;
    {
        py::gil_scoped_release release;
        trace = index.trace_hnsw(point, k, ef);
    }
    py::list ids, distances;
    for (const auto& neighbor : trace.neighbors) {
        ids.append(neighbor.id);
        distances.append(neighbor.distance);
    }
    py::dict result;
    result["ids"] = ids;
    result["squared_distances"] = distances;
    result["examined"] = trace.examined;
    result["discovery_edges"] = trace.discovery_edges;
    result["exact_fallback"] = trace.exact_fallback;
    return result;
}
}

PYBIND11_MODULE(vector_search, m) {
    m.doc() = "Immutable C++ vector index with batched squared-L2 search";
    m.attr("distance_backend") = vs::distance_kernel().name;
    py::class_<vs::Index>(m, "Index")
        .def(py::init(&build), py::arg("vectors"), py::arg("tree") = false, py::kw_only(),
             py::arg("hnsw") = false, py::arg("M") = 16, py::arg("efConstruction") = 100)
        .def_property_readonly("size", &vs::Index::size)
        .def_property_readonly("dimension", &vs::Index::dimension)
        .def_property_readonly("has_hnsw", &vs::Index::has_hnsw)
        .def_property_readonly("has_tree", &vs::Index::has_tree)
        .def_property_readonly("storage_bytes", &vs::Index::storage_bytes)
        .def("search", &search, py::arg("queries"), py::arg("k") = 10,
             py::arg("budget") = 0, py::arg("threads") = 1, py::kw_only(), py::arg("efSearch") = 0)
        .def("trace_hnsw", &trace_hnsw, py::arg("query"), py::arg("k") = 10,
             py::kw_only(), py::arg("efSearch") = 32)
        .def("save", [](const vs::Index& i, const std::string& p) { i.save(p); },
             py::call_guard<py::gil_scoped_release>())
        .def_static("load", [](const std::string& p) { return vs::Index::load(p); },
                    py::call_guard<py::gil_scoped_release>());
}
