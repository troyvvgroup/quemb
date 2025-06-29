#pragma once
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using int_t = int64_t;
using OrbitalIdx = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Tensor3D = Eigen::Tensor<double, 3, Eigen::ColMajor>;

// Sum of integers from 1 to n
constexpr int_t gauss_sum(int_t n)
{
    return (n * (n + 1)) / 2;
}

// Ravel symmetric index (i,j) -> unique index
constexpr int_t ravel_symmetric(int_t a, int_t b)
{
    return (a > b) ? gauss_sum(a) + b : gauss_sum(b) + a;
}

// Total number of unique (i, j) pairs with i <= j < n
constexpr int_t n_symmetric(int_t n)
{
    return ravel_symmetric(n - 1, n - 1) + 1;
}

// Invert symmetric raveled index (not constexpr due to sqrt)
inline std::tuple<int_t, int_t> unravel_symmetric(int_t i)
{
    int_t a = static_cast<int_t>((std::sqrt(8.0 * i + 1.0) - 1.0) / 2.0);
    int_t offset = gauss_sum(a);
    int_t b = i - offset;
    if (b > a)
        std::swap(a, b);
    return {a, b};
}

// Ravel four indices using symmetric ravel
constexpr int_t ravel_eri_idx(int_t a, int_t b, int_t c, int_t d)
{
    return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d));
}

// Invert raveled ERI index (not constexpr due to sqrt)
inline std::tuple<int_t, int_t, int_t, int_t> unravel_eri_idx(int_t i)
{
    auto [ab, cd] = unravel_symmetric(i);
    auto [a, b] = unravel_symmetric(ab);
    auto [c, d] = unravel_symmetric(cd);
    return {a, b, c, d};
}

// Total number of unique ERIs with same orbital count
constexpr int_t n_eri(int_t n)
{
    return ravel_eri_idx(n - 1, n - 1, n - 1, n - 1) + 1;
}

// Ravel (a,b) to 1D C-style index
constexpr int_t ravel_C(int_t a, int_t b, int_t n_cols)
{
    return a * n_cols + b;
}

// Ravel (a,b) to 1D Fortran-style index
constexpr int_t ravel_Fortran(int_t a, int_t b, int_t n_rows)
{
    return a + b * n_rows;
}

// Unique entries in a symmetric matrix with m, n dimensions
constexpr int_t symmetric_different_size(int_t m, int_t n)
{
    return (m > n) ? gauss_sum(n) + n * (m - n) : gauss_sum(m) + m * (n - m);
}

// Unique ERI count with non-equal orbital sizes
constexpr int_t get_flexible_n_eri(int_t p_max, int_t q_max, int_t r_max, int_t s_max)
{
    return symmetric_different_size(symmetric_different_size(p_max, q_max), symmetric_different_size(r_max, s_max));
}

constexpr inline Eigen::Index to_eigen(std::size_t idx) noexcept
{
    return static_cast<Eigen::Index>(idx);
}

std::vector<std::vector<OrbitalIdx>> extract_unique(const std::vector<std::vector<OrbitalIdx>> &exch_reachable)
{
    std::vector<std::vector<OrbitalIdx>> result;
    result.resize(exch_reachable.size());

    for (OrbitalIdx mu = 0; mu < to_eigen(exch_reachable.size()); ++mu) {
        for (OrbitalIdx nu : exch_reachable[mu]) {
            if (nu > mu)
                break; // assumes mu, nu are sorted
            result[mu].push_back(nu);
        }
    }
    return result;
}

class Timer
{
  public:
    explicit Timer(const char *name) : _name(name), _start(std::chrono::high_resolution_clock::now())
    {
    }

    ~Timer()
    {
        auto duration = elapsed_ms();
        std::cout << "[TIMER] " << _name << " finished in " << duration << " ms\n";
    }

    void print(const std::string &message = "Checkpoint") const
    {
        auto duration = elapsed_ms();
        std::cout << "[TIMER] " << _name << " - " << message << ": " << duration << " ms\n";
    }

  private:
    const char *_name;
    std::chrono::high_resolution_clock::time_point _start;

    double elapsed_ms() const
    {
        using namespace std::chrono;
        auto now = high_resolution_clock::now();
        return duration_cast<microseconds>(now - _start).count() * 1e-3;
    }
};

#define PROFILE_FUNCTION() Timer timer(__func__)
#define PROFILE_SCOPE() Timer timer(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " in " + __func__)