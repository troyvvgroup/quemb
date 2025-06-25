#define EIGEN_USE_OPENMP

#include <iostream>
#include <map>
#include <omp.h>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/TensorSymmetry>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "indexers.hpp"

namespace py = pybind11;

class SemiSparseSym3DTensor
{
  public:
    SemiSparseSym3DTensor(
        Matrix unique_dense_data, std::tuple<int, int, int> shape, std::vector<std::vector<OrbitalIdx>> exch_reachable,
        std::vector<std::vector<OrbitalIdx>> exch_reachable_unique,
        std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> exch_reachable_with_offsets,
        std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> exch_reachable_unique_with_offsets,
        std::unordered_map<std::size_t, std::size_t> offsets)
        : _unique_dense_data(std::move(unique_dense_data)), _shape(std::move(shape)),
          _exch_reachable(std::move(exch_reachable)), _exch_reachable_unique(std::move(exch_reachable_unique)),
          _exch_reachable_with_offsets(std::move(exch_reachable_with_offsets)),
          _exch_reachable_unique_with_offsets(std::move(exch_reachable_unique_with_offsets)),
          _offsets(std::move(offsets))
    {
    }

    SemiSparseSym3DTensor(Matrix unique_dense_data, std::tuple<int, int, int> shape,
                          std::vector<std::vector<OrbitalIdx>> exch_reachable)
        : _unique_dense_data(std::move(unique_dense_data)), _shape(std::move(shape)),
          _exch_reachable(std::move(exch_reachable))
    {

        _exch_reachable_unique = extract_unique(_exch_reachable);

        std::size_t counter = 0;
        for (OrbitalIdx mu = 0; mu < to_eigen(_exch_reachable_unique.size()); ++mu) {
            for (auto nu : _exch_reachable_unique[mu]) {
                _offsets[ravel_symmetric(mu, nu)] = counter++;
            }
        }

        // Initialize exch_reachable_with_offsets
        _exch_reachable_with_offsets.resize(_exch_reachable.size());
        _exch_reachable_unique_with_offsets.resize(_exch_reachable_unique.size());
        for (OrbitalIdx mu = 0; mu < to_eigen(_exch_reachable.size()); ++mu) {
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
    const auto &exch_reachable() const
    {
        return _exch_reachable;
    }
    const auto &exch_reachable_with_offsets() const
    {
        return _exch_reachable_with_offsets;
    }
    const auto &exch_reachable_unique() const
    {
        return _exch_reachable_unique;
    }
    const auto &exch_reachable_unique_with_offsets() const
    {
        return _exch_reachable_unique_with_offsets;
    }
    const auto &dense_data() const
    {
        return _unique_dense_data;
    }
    const auto &get_offsets() const
    {
        return _offsets;
    }
    std::size_t get_size() const
    {
        return std::get<0>(_shape) * std::get<1>(_shape) * std::get<2>(_shape);
    }
    std::size_t get_nonzero_size() const
    {
        return _unique_dense_data.size();
    }
    constexpr auto get_shape() const
    {
        return _shape;
    }
    Matrix::ConstColXpr get_aux_vector(const OrbitalIdx mu, const OrbitalIdx nu) const
    {
        return _unique_dense_data.col(_offsets.at(ravel_symmetric(mu, nu)));
    }

  private:
    // We assume (P | mu nu) layout, because Eigen is column-major
    Matrix _unique_dense_data;
    // We assume (P | mu nu) layout, because Eigen is column-major,
    // i.e. the shape is (naux, nao, nao) where naux is the number of auxiliary basis functions
    std::tuple<int, int, int> _shape;

    std::vector<std::vector<OrbitalIdx>> _exch_reachable;
    std::vector<std::vector<OrbitalIdx>> _exch_reachable_unique;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_with_offsets;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_unique_with_offsets;

    // Map from raveled symmetric indices to offsets in the dense data
    std::unordered_map<std::size_t, std::size_t> _offsets;
};

class SemiSparse3DTensor
{
  public:
    SemiSparse3DTensor(Matrix dense_data, std::tuple<int, int, int> shape,
                       std::vector<std::vector<OrbitalIdx>> AO_reachable_by_MO,
                       std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> AO_reachable_by_MO_with_offsets,
                       std::unordered_map<std::size_t, std::size_t> offsets)
        : _dense_data(std::move(dense_data)), _shape(std::move(shape)),
          _AO_reachable_by_MO(std::move(AO_reachable_by_MO)),
          _AO_reachable_by_MO_with_offsets(std::move(AO_reachable_by_MO_with_offsets)), _offsets(std::move(offsets))
    {
    }

    SemiSparse3DTensor(Matrix dense_data, std::tuple<int, int, int> shape,
                       std::vector<std::vector<OrbitalIdx>> AO_reachable_by_MO)
        : _dense_data(std::move(dense_data)), _shape(std::move(shape)),
          _AO_reachable_by_MO(std::move(AO_reachable_by_MO))
    {

        const auto [naux, nao, nmo] = _shape;
        _AO_reachable_by_MO_with_offsets.resize(_AO_reachable_by_MO.size());

        std::size_t counter = 0;
        for (OrbitalIdx i_MO = 0; i_MO < to_eigen(_AO_reachable_by_MO.size()); ++i_MO) {
            std::vector<std::pair<std::size_t, OrbitalIdx>> pairs;
            for (OrbitalIdx nu : _AO_reachable_by_MO[i_MO]) {
                const std::size_t flat = ravel_Fortran(i_MO, nu, nao);
                _offsets[flat] = counter++;
                pairs.emplace_back(_offsets[flat], nu);
            }
            _AO_reachable_by_MO_with_offsets[i_MO] = std::move(pairs);
        }
    }

    const auto &exch_reachable() const
    {
        return _AO_reachable_by_MO;
    }
    const auto &exch_reachable_with_offsets() const
    {
        return _AO_reachable_by_MO_with_offsets;
    }
    const auto &dense_data() const
    {
        return _dense_data;
    }
    const auto &get_offsets() const
    {
        return _offsets;
    }
    constexpr auto get_shape() const
    {
        return _shape;
    }
    std::size_t get_size() const
    {
        return std::get<0>(_shape) * std::get<1>(_shape) * std::get<2>(_shape);
    }
    std::size_t get_nonzero_size() const
    {
        return _dense_data.size();
    }

    Matrix::ConstColXpr get_aux_vector(OrbitalIdx mu, OrbitalIdx i) const
    {
        const auto [naux, nao, _] = _shape;
        return _dense_data.col(_offsets.at(ravel_Fortran(mu, i, nao)));
    }

  private:
    // We assume (P | mu i) layout, because Eigen is column-major,
    // i.e. the shape is (naux, nao, nmo) where naux is the number of auxiliary basis functions,
    // nao is the number of atomic orbitals, and nmo is the number of molecular orbitals.
    Matrix _dense_data;
    std::tuple<OrbitalIdx, OrbitalIdx, OrbitalIdx> _shape;

    std::vector<std::vector<OrbitalIdx>> _AO_reachable_by_MO;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _AO_reachable_by_MO_with_offsets;

    std::unordered_map<std::size_t, std::size_t> _offsets;
};

// Input types:
// TA and S_abs are Eigen matrices of double
// epsilon is a double threshold
// Returns vector of vectors of int_t indices
std::vector<std::vector<OrbitalIdx>> get_AO_per_MO(const Matrix &TA, const Matrix &S_abs, double epsilon) noexcept
{
    const std::size_t n_MO = TA.cols();

    // Compute X = |S_abs * TA|
    const Matrix X = (S_abs * TA).cwiseAbs();

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
    const std::vector<std::vector<OrbitalIdx>> &AO_reachable_by_MO) noexcept
{
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> result;
    std::size_t counter = 0;

    for (const auto &AOs : AO_reachable_by_MO) {
        std::vector<std::pair<std::size_t, OrbitalIdx>> pairs;
        for (const auto &AO : AOs) {
            pairs.emplace_back(counter++, AO);
        }
        result.push_back(std::move(pairs));
    }

    return result;
}

SemiSparse3DTensor contract_with_TA_1st(const Matrix &TA, const SemiSparseSym3DTensor &int_P_mu_nu,
                                        const std::vector<std::vector<OrbitalIdx>> &AO_by_MO) noexcept
{
    const OrbitalIdx nao = TA.rows();
    const OrbitalIdx nmo = TA.cols();
    const OrbitalIdx naux = std::get<0>(int_P_mu_nu.get_shape());

    const auto AO_by_MO_with_offsets = get_AO_reachable_by_MO_with_offset(AO_by_MO);

    std::size_t n_unique = 0;
    for (const auto &offsets : AO_by_MO_with_offsets) {
        n_unique += offsets.size();
    }

    Matrix g_unique = Matrix::Zero(naux, n_unique);
    std::unordered_map<std::size_t, std::size_t> offsets;
    offsets.reserve(n_unique);

    // Modifying the offsets map to store the offsets for each (mu, i) pair
    // cannot be parallelized.
    for (OrbitalIdx i = 0; i < nmo; ++i) {
        for (const auto &[offset, mu] : AO_by_MO_with_offsets[i]) {
            offsets[ravel_Fortran(mu, i, nao)] = offset;
        }
    }
#pragma omp parallel for
    for (OrbitalIdx i = 0; i < nmo; ++i) {
        for (const auto &[offset, mu] : AO_by_MO_with_offsets[i]) {
            for (const auto &[inner_offset, nu] : int_P_mu_nu.exch_reachable_with_offsets()[mu]) {
                g_unique.col(offset) += TA(nu, i) * int_P_mu_nu.dense_data().col(inner_offset);
            }
        }
    }
    return SemiSparse3DTensor(std::move(g_unique), std::make_tuple(naux, nao, nmo), AO_by_MO,
                              std::move(AO_by_MO_with_offsets), std::move(offsets));
}

py::array_t<double> copy_to_numpy(const Tensor3D &g) noexcept
{
    py::gil_scoped_acquire gil;
    auto shape = g.dimensions();
    py::array_t<double, py::array::f_style> arr({shape[0], shape[1], shape[2]});
    // Copy the data into the numpy array
    std::memcpy(arr.mutable_data(), g.data(), sizeof(double) * g.size());
    return arr;
}

Tensor3D copy_from_numpy(py::array_t<double, py::array::f_style> arr)
{
    // Acquire GIL if needed (not strictly necessary here, but good practice)
    py::gil_scoped_acquire gil;

    // Check input dimensions
    if (arr.ndim() != 3) {
        throw std::runtime_error("Input numpy array must have 3 dimensions");
    }

    auto shape = arr.shape();
    Tensor3D tensor(shape[0], shape[1], shape[2]);

    // Copy the data
    std::memcpy(tensor.data(), arr.data(), sizeof(double) * tensor.size());

    return tensor;
}

Matrix contract_with_TA_2nd_to_sym_dense(const Matrix &TA, const SemiSparse3DTensor &int_mu_i_P) noexcept
{
    const auto [naux, nao, nmo] = int_mu_i_P.get_shape();

    assert(TA.rows() == nao && "TA.shape[0] must match int_mu_i_P.shape[1]");
    assert(TA.cols() == nmo && "TA.shape[1] must match int_mu_i_P.shape[2]");

    const auto n_sym_pairs = to_eigen(ravel_symmetric(nmo - 1, nmo - 1) + 1);

    Matrix sym_P_pq(naux, n_sym_pairs);

#pragma omp parallel for
    for (OrbitalIdx i = 0; i < nmo; ++i) {
        for (OrbitalIdx j = 0; j <= i; ++j) {
            Eigen::VectorXd tmp = Eigen::VectorXd::Zero(naux);
            for (const auto &[offset, mu] : int_mu_i_P.exch_reachable_with_offsets()[i]) {
                tmp += TA(mu, j) * int_mu_i_P.dense_data().col(offset);
            }
            sym_P_pq.col(ravel_symmetric(i, j)) = tmp;
        }
    }

    return sym_P_pq;
}

// Computes the integral (p q | r s) from (P | pq) using the Cholesky factorization of (P | Q).
// sym_P_pq is the (P | pq) matrix and uses a fused indexing scheme for pq, accounting for symmetry of p and q.
// L_PQ is the Cholesky factor of the (P | Q) matrix, which is lower triangular.
Matrix eval_via_cholesky(const Matrix &sym_P_pq, const Matrix &L_PQ) noexcept
{
    // Step 1: Solve L * X = sym_P_pq  →  X = L⁻¹ sym_P_pq
    Matrix X = L_PQ.triangularView<Eigen::Lower>().solve(sym_P_pq);
    // Step 2: Return Xᵀ X
    return X.transpose() * X;
}

Matrix transform_integral(const SemiSparseSym3DTensor &int_P_mu_nu, const Matrix &TA, const Matrix &S_abs,
                          const Matrix &L_PQ, const double MO_coeff_epsilon) noexcept

{
    const auto AO_by_MO = get_AO_per_MO(TA, S_abs, MO_coeff_epsilon);
    const SemiSparse3DTensor int_P_mu_i = contract_with_TA_1st(TA, int_P_mu_nu, AO_by_MO);
    const Matrix P_pq = contract_with_TA_2nd_to_sym_dense(TA, int_P_mu_i);

    return eval_via_cholesky(P_pq, L_PQ);
}

// Binding code
PYBIND11_MODULE(eri_sparse_DF, m)
{
    m.doc() = "Minimal pybind11 + Eigen example";

    py::class_<SemiSparseSym3DTensor>(m, "SemiSparseSym3DTensor")
        // Minimal constructor
        .def(py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>>())
        // Full constructor
        .def(
            py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>,
                     std::vector<std::vector<OrbitalIdx>>, std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>>,
                     std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>>,
                     std::unordered_map<std::size_t, std::size_t>>())
        .def_property_readonly("unique_dense_data", &SemiSparseSym3DTensor::dense_data)
        .def_property_readonly("shape", &SemiSparseSym3DTensor::get_shape)
        .def_property_readonly("exch_reachable", &SemiSparseSym3DTensor::exch_reachable)
        .def_property_readonly("exch_reachable_unique", &SemiSparseSym3DTensor::exch_reachable_unique)
        .def_property_readonly("offsets", &SemiSparseSym3DTensor::get_offsets)
        .def_property_readonly("size", &SemiSparseSym3DTensor::get_size)
        .def_property_readonly("nonzero_size", &SemiSparseSym3DTensor::get_nonzero_size)
        .def("get_aux_vector", &SemiSparseSym3DTensor::get_aux_vector)
        .def(
            "__getitem__",
            [](const SemiSparseSym3DTensor &self, std::tuple<OrbitalIdx, OrbitalIdx> idx) {
                OrbitalIdx mu = std::get<0>(idx);
                OrbitalIdx nu = std::get<1>(idx);
                return self.get_aux_vector(mu, nu);
            },
            py::return_value_policy::reference_internal // important to keep reference valid
            )
        .doc() = "Immutable, semi-sparse, partially symmetric 3-index tensor\n\n"
                 "Assumes:\n"
                 "  - T_{ijk} = T_{jik} symmetry\n"
                 "  - Sparsity over (i, j), dense over k\n"
                 "  - Example use: 3-center integrals (μν|P)";

    py::class_<SemiSparse3DTensor>(m, "SemiSparse3DTensor")
        // Minimal constructor
        .def(py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>>(), py::arg("dense_data"),
             py::arg("shape"), py::arg("AO_reachable_by_MO"))

        // Full constructor
        .def(py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>,
                      std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>>,
                      std::unordered_map<std::size_t, std::size_t>>(),
             py::arg("dense_data"), py::arg("shape"), py::arg("AO_reachable_by_MO"),
             py::arg("AO_reachable_by_MO_with_offsets"), py::arg("offsets"))

        // Read-only accessors
        .def_property_readonly("shape", &SemiSparse3DTensor::get_shape)
        .def_property_readonly("dense_data", &SemiSparse3DTensor::dense_data)
        .def_property_readonly("AO_reachable_by_MO", &SemiSparse3DTensor::exch_reachable)
        .def_property_readonly("AO_reachable_by_MO_with_offsets", &SemiSparse3DTensor::exch_reachable_with_offsets)
        .def_property_readonly("offsets", &SemiSparse3DTensor::get_offsets)
        .def_property_readonly("size", &SemiSparse3DTensor::get_size)
        .def_property_readonly("nonzero_size", &SemiSparse3DTensor::get_nonzero_size)

        // Method
        .def("get_aux_vector", &SemiSparse3DTensor::get_aux_vector, py::arg("mu"), py::arg("i"),
             "Return auxiliary vector for given AO and MO index");

    m.def("contract_with_TA_1st", &contract_with_TA_1st, py::arg("TA"), py::arg("int_P_mu_nu"), py::arg("AO_by_MO"),
          py::call_guard<py::gil_scoped_release>());

    m.def("contract_with_TA_2nd_to_sym_dense", &contract_with_TA_2nd_to_sym_dense, py::arg("TA"), py::arg("int_mu_i_P"),
          py::call_guard<py::gil_scoped_release>(), "Contract with TA to get a symmetric dense tensor (P | i, j)");

    m.def("get_AO_per_MO", &get_AO_per_MO, py::arg("TA"), py::arg("S_abs"), py::arg("epsilon"),
          py::call_guard<py::gil_scoped_release>(),
          "Get AOs per MO based on TA and S_abs matrices with a threshold epsilon");

    m.def("get_AO_reachable_by_MO_with_offset", &get_AO_reachable_by_MO_with_offset, py::arg("AO_reachable_by_MO"),
          py::call_guard<py::gil_scoped_release>(),
          "Get AO reachable by MO with offsets based on the provided AO_reachable_by_MO structure");

    m.def("extract_unique", &extract_unique, py::arg("exch_reachable"), py::call_guard<py::gil_scoped_release>(),
          "Extract unique reachable AOs from the provided exch_reachable structure");

    m.def("transform_integral", &transform_integral, py::arg("int_P_mu_nu"), py::arg("TA"), py::arg("S_abs"),
          py::arg("L_PQ"), py::arg("MO_coeff_epsilon"), py::call_guard<py::gil_scoped_release>(),
          "Transform the integral using TA, int_P_mu_nu, AO_by_MO, and L_PQ, returning the transformed matrix");
}