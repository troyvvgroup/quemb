#define EIGEN_USE_OPENMP

#include <iostream>
#include <map>
#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#ifdef GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wstringop-overread"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef GCC
#pragma GCC diagnostic pop
#endif

#include "indexers.hpp"

namespace py = pybind11;

#ifdef USE_CUDA
#define CUDA_CHECK_THROW(err)                                                                                          \
    if ((err) != cudaSuccess)                                                                                          \
        throw std::runtime_error(cudaGetErrorString(err));

#define CUBLAS_CHECK_THROW(err)                                                                                        \
    if ((err) != CUBLAS_STATUS_SUCCESS)                                                                                \
        throw std::runtime_error("cuBLAS error");

class GPU_MatrixHandle
{
  public:
    explicit GPU_MatrixHandle(const Eigen::MatrixXd &L_host)
        : _n_rows(L_host.rows()), _n_cols(L_host.cols()), _size(static_cast<size_t>(_n_rows * _n_cols))
    {
        const size_t bytes = _size * sizeof(double);
        CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&d_L), bytes));
        CUDA_CHECK_THROW(cudaMemcpy(d_L, L_host.data(), bytes, cudaMemcpyHostToDevice));
    }

    ~GPU_MatrixHandle()
    {
        if (d_L)
            cudaFree(d_L);
    }

    const double *cdata() const noexcept
    {
        return d_L;
    }
    double *data() const noexcept
    {
        return d_L;
    }
    size_t size() const noexcept
    {
        return _size;
    }
    int_t rows() const noexcept
    {
        return _n_rows;
    }
    int_t cols() const noexcept
    {
        return _n_cols;
    }

  private:
    double *d_L = nullptr;
    int_t _n_rows;
    int_t _n_cols;
    size_t _size;
};
#endif

class SemiSparseSym3DTensor
{
  public:
    explicit SemiSparseSym3DTensor(
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

    explicit SemiSparseSym3DTensor(Matrix unique_dense_data, std::tuple<int, int, int> shape,
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
        _offsets = rebuild_unordered_map(_offsets);

        // Initialize
        // exch_reachable_with_offsets
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
    const Matrix &dense_data() const
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
    // We assume (P | mu nu) layout, because Eigen is
    // column-major
    Matrix _unique_dense_data;
    // We assume (P | mu nu) layout, because Eigen is
    // column-major, i.e. the shape is (naux, nao, nao)
    // where naux is the number of auxiliary basis functions
    std::tuple<int, int, int> _shape;

    std::vector<std::vector<OrbitalIdx>> _exch_reachable;
    std::vector<std::vector<OrbitalIdx>> _exch_reachable_unique;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_with_offsets;
    std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> _exch_reachable_unique_with_offsets;

    // Map from raveled symmetric indices to offsets in the
    // dense data
    std::unordered_map<std::size_t, std::size_t> _offsets;
};

class SemiSparse3DTensor
{
  public:
    explicit SemiSparse3DTensor(
        Matrix dense_data, std::tuple<int, int, int> shape, std::vector<std::vector<OrbitalIdx>> AO_reachable_by_MO,
        std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>> AO_reachable_by_MO_with_offsets,
        std::unordered_map<std::size_t, std::size_t> offsets)
        : _dense_data(std::move(dense_data)), _shape(std::move(shape)),
          _AO_reachable_by_MO(std::move(AO_reachable_by_MO)),
          _AO_reachable_by_MO_with_offsets(std::move(AO_reachable_by_MO_with_offsets)), _offsets(std::move(offsets))
    {
    }

    explicit SemiSparse3DTensor(Matrix dense_data, std::tuple<int, int, int> shape,
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
        _offsets = rebuild_unordered_map(_offsets);
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
    // We assume (P | mu i) layout, because Eigen is
    // column-major, i.e. the shape is (naux, nao, nmo)
    // where naux is the number of auxiliary basis
    // functions, nao is the number of atomic orbitals, and
    // nmo is the number of molecular orbitals.
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
        // Check which AO indices
        // satisfy X(row, i_MO) >=
        // epsilon
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
    PROFILE_FUNCTION();
    const OrbitalIdx nao = TA.rows();
    const OrbitalIdx nmo = TA.cols();
    const OrbitalIdx naux = std::get<0>(int_P_mu_nu.get_shape());

    const auto AO_by_MO_with_offsets = get_AO_reachable_by_MO_with_offset(AO_by_MO);

    std::size_t n_unique = 0;
    for (const auto &offsets : AO_by_MO_with_offsets) {
        n_unique += offsets.size();
    }

    if (LOG_LEVEL <= LogLevel::Info) {
        std::cout << "(P | mu i) [MEMORY] "
                     "sparse "
                  << bytes_to_gib(naux * n_unique * sizeof(double)) << " GiB" << "\n";
        std::cout << "(P | mu i) [MEMORY] "
                     "dense "
                  << bytes_to_gib(naux * nao * nmo * sizeof(double)) << " GiB" << "\n";
        std::cout << "(P | mu i) [MEMORY] "
                     "sparsity "
                  << (1. - static_cast<double>(n_unique) / static_cast<double>(nao * nmo)) * 100. << " %"
                  << "\n";
    };

    Matrix g_unique = Matrix::Zero(naux, n_unique);
    std::unordered_map<std::size_t, std::size_t> offsets;
    offsets.reserve(n_unique);

    // Modifying the offsets map to store the offsets for
    // each (mu, i) pair cannot be parallelized.
    for (OrbitalIdx i = 0; i < nmo; ++i) {
        for (const auto &[offset, mu] : AO_by_MO_with_offsets[i]) {
            offsets[ravel_Fortran(mu, i, nao)] = offset;
        }
    }
    offsets = rebuild_unordered_map(offsets);
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
    std::memcpy(arr.mutable_data(), g.data(), sizeof(double) * g.size());
    return arr;
}

Tensor3D copy_from_numpy(py::array_t<double, py::array::f_style> arr)
{
    py::gil_scoped_acquire gil;

    if (arr.ndim() != 3) {
        throw std::runtime_error("Input numpy array must "
                                 "have 3 dimensions");
    }

    auto shape = arr.shape();
    Tensor3D tensor(shape[0], shape[1], shape[2]);

    std::memcpy(tensor.data(), arr.data(), sizeof(double) * tensor.size());

    return tensor;
}

Matrix contract_with_TA_2nd_to_sym_dense(const SemiSparse3DTensor &int_mu_i_P, const Matrix &TA) noexcept
{
    PROFILE_FUNCTION();
#ifndef CLANG
    const auto [naux, nao, nmo] = int_mu_i_P.get_shape();
#else
    // Clang does not yet support capturing structured
    // bindings in OpenMP. Use the structured binding, if it
    // works in the future.
    // https://github.com/llvm/llvm-project/issues/33025
    const auto shape = int_mu_i_P.get_shape();
    const auto naux = std::get<0>(shape);
    const auto nao = std::get<1>(shape);
    UNUSED(nao); // Unused in release
    const auto nmo = std::get<2>(shape);
#endif
    assert(TA.rows() == nao && "TA.shape[0] must match int_mu_i_P.shape[1]");
    assert(TA.cols() == nmo && "TA.shape[1] must match int_mu_i_P.shape[2]");

    const auto n_sym_pairs = to_eigen(ravel_symmetric(nmo - 1, nmo - 1) + 1);

    if (LOG_LEVEL <= LogLevel::Debug) {
        std::cout << "[MEMORY] about to "
                     "allocate "
                     "sym_P_pq(naux, "
                     "n_sym_pairs) with "
                  << bytes_to_gib(naux * n_sym_pairs * sizeof(double)) << " GiB" << std::endl;
    }
    Matrix sym_P_pq(naux, n_sym_pairs);

#pragma omp parallel
    {
        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(naux); // one per thread

#pragma omp for
        for (OrbitalIdx ij_sym = 0; ij_sym < n_sym_pairs; ++ij_sym) {
            const auto &[i, j] = unravel_symmetric(ij_sym);
            tmp.setZero(); // reuse same memory
            for (const auto &[offset, mu] : int_mu_i_P.exch_reachable_with_offsets()[i]) {
                tmp.noalias() += TA(mu, j) * int_mu_i_P.dense_data().col(offset);
            }
            sym_P_pq.col(ij_sym) = tmp;
        }
    }

    return sym_P_pq;
}

// Computes the integral (p q | r s) from (P | pq) using the Cholesky
// factorization of (P | Q). sym_P_pq is the (P | pq) matrix and uses a fused
// indexing scheme for pq, accounting for symmetry of p and q. L_PQ is the
// Cholesky factor of the (P | Q) matrix, which is lower triangular.
Matrix eval_via_cholesky(const Matrix &sym_P_pq, const Matrix &L_PQ) noexcept
{
    Timer cholesky_timer{"eval_via_cholesky"};
    // Step 1: Solve L * X = sym_P_pq  →  X = L⁻¹ sym_P_pq
    const Matrix X = L_PQ.triangularView<Eigen::Lower>().solve(sym_P_pq);
    if (LOG_LEVEL <= LogLevel::Info) {
        cholesky_timer.print("triangular solve "
                             "completed");
    };
    // Step 2: Return Xᵀ X
    return X.transpose() * X;
}

#ifdef USE_CUDA
Matrix eval_via_cholesky_cuda(const Matrix &sym_P_pq, const GPU_MatrixHandle &L_PQ)
{
    const int n_aux = static_cast<int>(L_PQ.rows());
    const int n_sym_pairs = static_cast<int>(sym_P_pq.cols());

    const size_t bytes_sym_P_pq = sizeof(double) * sym_P_pq.size();
    const size_t bytes_X = sizeof(double) * n_aux * n_sym_pairs;
    const size_t bytes_res = sizeof(double) * n_sym_pairs * n_sym_pairs;

    double *d_X = nullptr, *d_result = nullptr;

    if (LOG_LEVEL <= LogLevel::Debug) {
        std::cout << __func__
                  << "[GPU MEMORY] about to "
                     "allocate bytes_X "
                  << bytes_to_gib(bytes_X) << " GiB\n";
        std::cout << __func__
                  << "[GPU MEMORY] about to "
                     "allocate bytes_res "
                  << bytes_to_gib(bytes_res) << " GiB" << std::endl;
    }

    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&d_X), bytes_X));
    CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void **>(&d_result), bytes_res));
    if (LOG_LEVEL <= LogLevel::Debug) {
        std::cout << __func__
                  << "[MEMORY] bytes_X "
                     "succesfully "
                     "allocated\n";
        std::cout << __func__
                  << "[MEMORY] bytes_res "
                     "successfully allocated"
                  << std::endl;
    }

    // Copy data to device
    // We will solve: L * X = sym_P_pq  → X = L⁻¹ * sym_P_pq
    // But X will be initialized with sym_P_pq and then
    // overwritten by the solution.
    CUDA_CHECK_THROW(cudaMemcpy(d_X, sym_P_pq.data(), bytes_sym_P_pq, cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK_THROW(cublasCreate(&handle));

    const double alpha = 1.0;

    // Solve: L * X = sym_P_pq  → X = L⁻¹ * sym_P_pq
    // X is initialized with sym_P_pq and overwrite it with
    // the solution.
    CUBLAS_CHECK_THROW(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                   n_aux, n_sym_pairs, &alpha, L_PQ.cdata(), n_aux, d_X, n_aux));

    // Compute: result = Xᵀ * X
    CUBLAS_CHECK_THROW(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n_sym_pairs, n_aux, &alpha, d_X, n_aux,
                                   &alpha, d_result, n_sym_pairs));

    // Transfer result to host
    Matrix result(n_sym_pairs, n_sym_pairs);
    CUDA_CHECK_THROW(cudaMemcpy(result.data(), d_result, bytes_res, cudaMemcpyDeviceToHost));

    // Fill lower triangle
    for (int i = 0; i < n_sym_pairs; ++i)
        for (int j = i + 1; j < n_sym_pairs; ++j)
            result(j, i) = result(i, j);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_X);
    cudaFree(d_result);

    return result;
}
#endif

#ifdef USE_CUDA
Matrix transform_integral_cuda(const SemiSparseSym3DTensor &int_P_mu_nu, const Matrix &TA, const Matrix &S_abs,
                               const GPU_MatrixHandle &L_PQ, const double MO_coeff_epsilon) noexcept

{
    const auto AO_by_MO = get_AO_per_MO(TA, S_abs, MO_coeff_epsilon);
    const SemiSparse3DTensor int_P_mu_i = contract_with_TA_1st(TA, int_P_mu_nu, AO_by_MO);
    const Matrix P_pq = contract_with_TA_2nd_to_sym_dense(int_P_mu_i, TA);

    return eval_via_cholesky_cuda(P_pq, L_PQ);
}
#endif

Matrix transform_integral(const SemiSparseSym3DTensor &int_P_mu_nu, const Matrix &TA, const Matrix &S_abs,
                          const Matrix &L_PQ, const double MO_coeff_epsilon) noexcept

{
    const auto AO_by_MO = get_AO_per_MO(TA, S_abs, MO_coeff_epsilon);
    const SemiSparse3DTensor int_P_mu_i = contract_with_TA_1st(TA, int_P_mu_nu, AO_by_MO);
    const Matrix P_pq = contract_with_TA_2nd_to_sym_dense(int_P_mu_i, TA);

    return eval_via_cholesky(P_pq, L_PQ);
}

// Automatically generate python type stub pages via
// pip install --no-deps -vvv . && pybind11-stubgen
// quemb.molbe._cpp.eri_sparse_DF -o src/
// --numpy-array-remove-parameters && ruff format && ruff check --fix

// Binding code
PYBIND11_MODULE(eri_sparse_DF, m)
{
    m.doc() = "Perform the sparse DF ERI transformation "
              "using semi-sparse tensors.\n\n"
              "This module provides functionality to "
              "transform ERIs using semi-sparse tensors\n"
              "and optionally CUDA for GPU acceleration.";

    m.def("get_log_level", &get_log_level);
    m.def("set_log_level", &set_log_level, py::arg("log_level"));

#ifdef USE_CUDA
    py::class_<GPU_MatrixHandle>(m, "GPU_MatrixHandle")
        .def(py::init<const Eigen::MatrixXd &>(), py::arg("L_host"),
             "Create a GPU_MatrixHandle from a host "
             "matrix.\n\n"
             "This allocates memory on the GPU and copies "
             "the data from the host to the GPU.")
        .def("__repr__", [](const GPU_MatrixHandle &self) {
            return "<GPU_MatrixHandle "
                   "of size " +
                   std::to_string(self.size()) + ">";
        });
#endif

    py::class_<SemiSparseSym3DTensor>(m, "SemiSparseSym3DTensor")
        // Minimal constructor
        .def(py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>>(),
             py::arg("unique_dense_data"), py::arg("shape"), py::arg("exch_reachable"))
        // Full constructor
        .def(
            py::init<Matrix, std::tuple<int, int, int>, std::vector<std::vector<OrbitalIdx>>,
                     std::vector<std::vector<OrbitalIdx>>, std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>>,
                     std::vector<std::vector<std::pair<std::size_t, OrbitalIdx>>>,
                     std::unordered_map<std::size_t, std::size_t>>(),
            py::arg("unique_dense_data"), py::arg("shape"), py::arg("exch_reachable"), py::arg("exch_reachable_unique"),
            py::arg("exch_reachable_with_offsets"), py::arg("exch_reachable_unique_with_offsets"), py::arg("offsets"))
        .def_property_readonly("unique_dense_data", &SemiSparseSym3DTensor::dense_data)
        .def_property_readonly("shape", &SemiSparseSym3DTensor::get_shape)
        .def_property_readonly("exch_reachable", &SemiSparseSym3DTensor::exch_reachable)
        .def_property_readonly("exch_reachable_unique", &SemiSparseSym3DTensor::exch_reachable_unique)
        .def_property_readonly("offsets", &SemiSparseSym3DTensor::get_offsets)
        .def_property_readonly("size", &SemiSparseSym3DTensor::get_size)
        .def_property_readonly("nonzero_size", &SemiSparseSym3DTensor::get_nonzero_size)
        .def(
            "__getitem__",
            [](const SemiSparseSym3DTensor &self, std::tuple<OrbitalIdx, OrbitalIdx> idx) {
                OrbitalIdx mu = std::get<0>(idx);
                OrbitalIdx nu = std::get<1>(idx);
                return self.get_aux_vector(mu, nu);
            },
            py::return_value_policy::reference_internal // important to keep
                                                        // reference valid
            )
        .doc() = "Immutable, semi-sparse, partially symmetric "
                 "3-index tensor\n\n"
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

        .def(
            "__getitem__",
            [](const SemiSparse3DTensor &self, std::tuple<OrbitalIdx, OrbitalIdx> idx) {
                OrbitalIdx mu = std::get<0>(idx);
                OrbitalIdx i = std::get<1>(idx);
                return self.get_aux_vector(mu, i);
            },
            py::return_value_policy::reference_internal // important to keep
                                                        // reference valid
        );

    m.def("contract_with_TA_1st", &contract_with_TA_1st, py::arg("TA"), py::arg("int_P_mu_nu"), py::arg("AO_by_MO"),
          py::call_guard<py::gil_scoped_release>());

    m.def("contract_with_TA_2nd_to_sym_dense", &contract_with_TA_2nd_to_sym_dense, py::arg("int_mu_i_P"), py::arg("TA"),
          py::call_guard<py::gil_scoped_release>(),
          "Contract with TA to get a symmetric dense "
          "tensor (P | i, j)");

    m.def("get_AO_per_MO", &get_AO_per_MO, py::arg("TA"), py::arg("S_abs"), py::arg("epsilon"),
          py::call_guard<py::gil_scoped_release>(),
          "Get AOs per MO based on TA and S_abs matrices "
          "with a threshold epsilon");

    m.def("get_AO_reachable_by_MO_with_offset", &get_AO_reachable_by_MO_with_offset, py::arg("AO_reachable_by_MO"),
          py::call_guard<py::gil_scoped_release>(),
          "Get AO reachable by MO with offsets based on the\n"
          "provided AO_reachable_by_MO structure");

    m.def("extract_unique", &extract_unique, py::arg("exch_reachable"), py::call_guard<py::gil_scoped_release>(),
          "Extract unique reachable AOs from the provided "
          "exch_reachable structure");

    m.def("transform_integral", &transform_integral, py::arg("int_P_mu_nu"), py::arg("TA"), py::arg("S_abs"),
          py::arg("L_PQ"), py::arg("MO_coeff_epsilon"), py::call_guard<py::gil_scoped_release>(),
          "Transform the integral using TA, int_P_mu_nu, "
          "AO_by_MO, and L_PQ,\n"
          "returning the transformed matrix");

#ifdef USE_CUDA
    m.def("transform_integral_cuda", &transform_integral_cuda, py::arg("int_P_mu_nu"), py::arg("TA"), py::arg("S_abs"),
          py::arg("L_PQ"), py::arg("MO_coeff_epsilon"), py::call_guard<py::gil_scoped_release>(),
          "Transform the integral using TA, int_P_mu_nu, "
          "AO_by_MO, and L_PQ,\n"
          "returning the transformed matrix.\n"
          "This uses CUDA for performance.");
#endif
}