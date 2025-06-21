#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <map>
#include <Eigen/Dense>

#include <iostream>

#include "indexers.hpp"


namespace py = pybind11;

using OrbitalIdx = int;

class SemiSparseSym3DTensor {
public:
    SemiSparseSym3DTensor(
        std::vector<int> keys,
        // We assume (P | mu nu) layout, because Eigen is column-major
        Eigen::MatrixXd unique_dense_data,
        // We assume (P | mu nu) layout, because Eigen is column-major
        std::tuple<int, int, int> shape,
        std::vector<std::vector<OrbitalIdx>> exch_reachable,
        std::vector<std::vector<OrbitalIdx>> exch_reachable_unique,
        std::vector<std::vector<std::pair<int, OrbitalIdx>>> exch_reachable_with_offsets,
        std::vector<std::vector<std::pair<int, OrbitalIdx>>> exch_reachable_unique_with_offsets
    )
    : _keys(std::move(keys)),
      _unique_dense_data(std::move(unique_dense_data)),
      _shape(std::move(shape)),
      _exch_reachable(std::move(exch_reachable)),
      _exch_reachable_unique(std::move(exch_reachable_unique)),
      _exch_reachable_with_offsets(std::move(exch_reachable_with_offsets)),
      _exch_reachable_unique_with_offsets(std::move(exch_reachable_unique_with_offsets)) {

        std::size_t counter = 0;

        for (std::size_t mu = 0; mu < _exch_reachable_unique.size(); ++mu) {
            for (auto nu : _exch_reachable_unique[mu]) {
                _data[ravel_symmetric(mu, nu)] = counter++;
            }
        }
      }

    // Public const accessors
    const auto& keys() const { return _keys; }
    const auto& exch_reachable() const { return _exch_reachable; }
    const auto& exch_reachable_unique() const { return _exch_reachable_unique; }
    const auto& dense_data() const { return _unique_dense_data; }
    const auto& get_data() const { return _data; }
    constexpr auto get_shape() const { return _shape; }
    auto get_aux_vector(const OrbitalIdx mu, const OrbitalIdx nu) const {
        return _unique_dense_data.col(_data.at(ravel_symmetric(mu, nu)));
    }

private:
    std::vector<int> _keys;
    Eigen::MatrixXd _unique_dense_data;
    std::tuple<int, int, int> _shape;

    std::vector<std::vector<OrbitalIdx>> _exch_reachable;
    std::vector<std::vector<OrbitalIdx>> _exch_reachable_unique;
    std::vector<std::vector<std::pair<int, OrbitalIdx>>> _exch_reachable_with_offsets;
    std::vector<std::vector<std::pair<int, OrbitalIdx>>> _exch_reachable_unique_with_offsets;

    std::map<std::size_t, std::size_t> _data;
};




// Binding code
PYBIND11_MODULE(mymodule, m) {
    m.doc() = "Minimal pybind11 + Eigen example";

    py::class_<SemiSparseSym3DTensor>(m, "SemiSparseSym3DTensor")
        .def(py::init<
            std::vector<int>,
            Eigen::MatrixXd,
            std::tuple<int, int, int>,
            std::vector<std::vector<OrbitalIdx>>,
            std::vector<std::vector<OrbitalIdx>>,
            std::vector<std::vector<std::pair<int, OrbitalIdx>>>,
            std::vector<std::vector<std::pair<int, OrbitalIdx>>>
        >())
        .def_property_readonly("keys", &SemiSparseSym3DTensor::keys)
        .def_property_readonly("unique_dense_data", &SemiSparseSym3DTensor::dense_data)
        .def_property_readonly("shape", &SemiSparseSym3DTensor::get_shape)
        .def_property_readonly("exch_reachable", &SemiSparseSym3DTensor::exch_reachable)
        .def_property_readonly("exch_reachable_unique", &SemiSparseSym3DTensor::exch_reachable_unique)
        .def_property_readonly("data", &SemiSparseSym3DTensor::get_data)
        .def("get_aux_vector", &SemiSparseSym3DTensor::get_aux_vector)
        .def("__getitem__",
                [](const SemiSparseSym3DTensor &self, std::tuple<OrbitalIdx, OrbitalIdx> idx) {
                    OrbitalIdx mu = std::get<0>(idx);
                    OrbitalIdx nu = std::get<1>(idx);
                    return self.get_aux_vector(mu, nu);
                },
                py::return_value_policy::reference_internal  // important to keep reference valid
        )
        .doc() =
            "Immutable, semi-sparse, partially symmetric 3-index tensor\n\n"
            "Assumes:\n"
            "  - T_{ijk} = T_{jik} symmetry\n"
            "  - Sparsity over (i, j), dense over k\n"
            "  - Example use: 3-center integrals (μν|P)";
}