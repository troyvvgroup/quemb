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

class SemiSparseSym3DTensor {
public:
    SemiSparseSym3DTensor(
        // We assume (P | mu nu) layout, because Eigen is column-major
        Eigen::MatrixXd unique_dense_data,
        // We assume (P | mu nu) layout, because Eigen is column-major
        std::tuple<int, int, int> shape,
        std::vector<std::vector<OrbitalIdx>> exch_reachable,
        std::vector<std::vector<OrbitalIdx>> exch_reachable_unique,
        std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> exch_reachable_with_offsets,
        std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> exch_reachable_unique_with_offsets,
        std::unordered_map<std::size_t, std::size_t> offsets
    )
    : _unique_dense_data(std::move(unique_dense_data)),
      _shape(std::move(shape)),
      _exch_reachable(std::move(exch_reachable)),
      _exch_reachable_unique(std::move(exch_reachable_unique)),
      _exch_reachable_with_offsets(std::move(exch_reachable_with_offsets)),
      _exch_reachable_unique_with_offsets(std::move(exch_reachable_unique_with_offsets)),
      _offsets(std::move(offsets)) {}

    SemiSparseSym3DTensor(
        // We assume (P | mu nu) layout, because Eigen is column-major
        Eigen::MatrixXd unique_dense_data,
        // We assume (P | mu nu) layout, because Eigen is column-major
        std::tuple<int, int, int> shape,
        std::vector<std::vector<OrbitalIdx>> exch_reachable
    )
    : _unique_dense_data(std::move(unique_dense_data)),
      _shape(std::move(shape)),
      _exch_reachable(std::move(exch_reachable)) {

        // Initialize exch_reachable_unique
        _exch_reachable_unique.resize(_exch_reachable.size());
        for (OrbitalIdx mu = 0; mu < _exch_reachable.size(); ++mu) {
            for (auto nu : _exch_reachable[mu]) {
                if (mu > nu) break; // Ensure mu <= nu for symmetry
                _exch_reachable_unique[mu].push_back(nu);
            }
        }

        std::size_t counter = 0;
        for (OrbitalIdx mu = 0; mu < _exch_reachable_unique.size(); ++mu) {
            for (auto nu : _exch_reachable_unique[mu]) {
                _offsets[ravel_symmetric(mu, nu)] = counter++;
            }
        }

        // Initialize exch_reachable_with_offsets
        _exch_reachable_with_offsets.resize(_exch_reachable.size());
        _exch_reachable_unique_with_offsets.resize(_exch_reachable_unique.size());
        for (std::size_t mu = 0; mu < _exch_reachable.size(); ++mu) {
            std::vector<std::pair<std::size_t, OrbitalIdx>> pairs;
            std::vector<std::pair<std::size_t, OrbitalIdx>> pairs_unique;
            for (auto nu : _exch_reachable[mu]) {
                pairs.emplace_back(_offsets[ravel_symmetric(mu, nu)], nu);
                if (mu <= nu) { // Ensure mu <= nu for symmetry
                    pairs_unique.emplace_back(_offsets[ravel_symmetric(mu, nu)], nu);
                }
            }
            _exch_reachable_with_offsets[mu] = std::move(pairs);
            _exch_reachable_unique_with_offsets[mu] = std::move(pairs_unique);
        }
      }

    // Public const accessors
    const auto& exch_reachable() const { return _exch_reachable; }
    const auto& exch_reachable_unique() const { return _exch_reachable_unique; }
    const auto& dense_data() const { return _unique_dense_data; }
    const auto& get_data() const { return _offsets; }
    constexpr auto get_shape() const { return _shape; }
    auto get_aux_vector(const OrbitalIdx mu, const OrbitalIdx nu) const {
        return _unique_dense_data.col(_offsets.at(ravel_symmetric(mu, nu)));
    }

private:
    std::vector<int> _keys;
    Eigen::MatrixXd _unique_dense_data;
    std::tuple<int, int, int> _shape;

    std::vector<std::vector<OrbitalIdx>> _exch_reachable;
    std::vector<std::vector<OrbitalIdx>> _exch_reachable_unique;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_with_offsets;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_unique_with_offsets;

    // Map from raveled symmetric indices to offsets in the dense data
    std::unordered_map<std::size_t, std::size_t> _offsets;
};


// Input types:
// TA and S_abs are Eigen matrices of double
// epsilon is a double threshold
// Returns vector of vectors of int_t indices
std::vector<std::vector<OrbitalIdx>> get_AO_per_MO(
    const Eigen::MatrixXd& TA,
    const Eigen::MatrixXd& S_abs,
    double epsilon
) {
    const std::size_t n_MO = TA.cols();

    // Compute X = |S_abs * TA|
    const Eigen::MatrixXd X = (S_abs * TA).cwiseAbs();

    std::vector<std::vector<OrbitalIdx>> result(n_MO);

    // For each molecular orbital i_MO
    for (std::size_t i_MO = 0; i_MO < n_MO; ++i_MO) {
        // Check which AO indices satisfy X(row, i_MO) >= epsilon
        for (OrbitalIdx i_AO = 0; to_eigen(i_AO) < X.rows(); ++i_AO) {
            if (X(i_AO, i_MO) >= epsilon) {
                result[i_MO].push_back(i_AO);
            }
        }
    }

    return result;
}


std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> get_AO_reachable_by_MO_with_offset(
    const std::vector<std::vector<OrbitalIdx>>& AO_reachable_by_MO
) {
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> result;
    std::size_t counter = 0;

    for (const auto& AOs : AO_reachable_by_MO) {
        std::vector<std::pair<std::size_t, OrbitalIdx>> pairs;
        for (const auto& AO : AOs) {
            pairs.emplace_back(counter++, AO);
        }
        result.push_back(std::move(pairs));
    }

    return result;
}




// Binding code
PYBIND11_MODULE(mymodule, m) {
    m.doc() = "Minimal pybind11 + Eigen example";

    py::class_<SemiSparseSym3DTensor>(m, "SemiSparseSym3DTensor")
        .def(py::init<
            Eigen::MatrixXd,
            std::tuple<int, int, int>,
            std::vector<std::vector<OrbitalIdx>>
        >())
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