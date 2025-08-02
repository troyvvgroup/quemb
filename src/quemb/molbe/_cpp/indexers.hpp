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

#define UNUSED(x) (void)(x)

using int_t = int64_t;
using OrbitalIdx = Eigen::Index;
using Matrix = Eigen::MatrixXd;
using Tensor3D = Eigen::Tensor<double, 3, Eigen::ColMajor>;

// Matches python logging levels
// https://docs.python.org/3/library/logging.html#logging-levels
enum class LogLevel : int
{
    NotSet = 0,
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
    Critical = 50
};

inline LogLevel LOG_LEVEL = LogLevel::NotSet;

// Expose int getter of LogLevel
inline int get_log_level()
{
    return static_cast<int>(LOG_LEVEL);
}

// Expose int setter of LogLevel
inline void set_log_level(int lvl)
{
    LOG_LEVEL = static_cast<LogLevel>(lvl);
}

// Utility: constrain to integral types and cast to int_t
template <typename T> constexpr int_t to_int_t(const T value) noexcept
{
    static_assert(std::is_integral_v<T>, "Only integral types are supported.");
    return static_cast<int_t>(value);
}

constexpr inline Eigen::Index to_eigen(std::size_t idx) noexcept
{
    return static_cast<Eigen::Index>(idx);
}

template <typename T> constexpr inline std::size_t to_index(T i) noexcept
{
    static_assert(std::is_integral_v<T>, "to_index requires an integral type");
    assert(i >= 0);
    return static_cast<std::size_t>(i);
}

// Sum of integers from 1 to n
template <typename T> constexpr int_t gauss_sum(const T n) noexcept
{
    const int_t N = to_int_t(n);
    return (N * (N + 1)) / 2;
}

// Ravel symmetric index (i,j) -> unique index
template <typename T1, typename T2>
constexpr int_t ravel_symmetric(const T1 a, const T2 b) noexcept
{
    const int_t A = to_int_t(a), B = to_int_t(b);
    return (A > B) ? gauss_sum(A) + B : gauss_sum(B) + A;
}

// Total number of unique (i, j) pairs with i <= j < n
template <typename T> constexpr int_t n_symmetric(const T n) noexcept
{
    const int_t N = to_int_t(n);
    return ravel_symmetric(N - 1, N - 1) + 1;
}

// Invert symmetric raveled index (not constexpr due to std::sqrt in C++20)
template <typename T> inline std::pair<int_t, int_t> unravel_symmetric(const T i)
{
    const int_t I = to_int_t(i);
    const int_t a =
        static_cast<int_t>((std::sqrt(8.0 * static_cast<double>(I) + 1.0) - 1.0) / 2.0);
    const int_t offset = gauss_sum(a);
    const int_t b = I - offset;
    return (a <= b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

// Ravel four indices using symmetric ravel
template <typename T1, typename T2, typename T3, typename T4>
constexpr int_t ravel_eri_idx(const T1 a, const T2 b, const T3 c, const T4 d) noexcept
{
    return ravel_symmetric(ravel_symmetric(a, b), ravel_symmetric(c, d));
}

// Invert raveled ERI index (not constexpr due to sqrt)
template <typename T>
inline std::tuple<int_t, int_t, int_t, int_t> unravel_eri_idx(const T i)
{
    const auto [ab, cd] = unravel_symmetric(i);
    const auto [a, b] = unravel_symmetric(ab);
    const auto [c, d] = unravel_symmetric(cd);
    return {a, b, c, d};
}

// Total number of unique ERIs with same orbital count
template <typename T> constexpr int_t n_eri(const T n) noexcept
{
    const int_t N = to_int_t(n);
    return ravel_eri_idx(N - 1, N - 1, N - 1, N - 1) + 1;
}

// Ravel (a,b) to 1D C-style index
template <typename T1, typename T2, typename T3>
constexpr int_t ravel_C(const T1 a, const T2 b, const T3 n_cols) noexcept
{
    return to_int_t(a) * to_int_t(n_cols) + to_int_t(b);
}

// Ravel (a,b) to 1D Fortran-style index
template <typename T1, typename T2, typename T3>
constexpr int_t ravel_Fortran(const T1 a, const T2 b, const T3 n_rows) noexcept
{
    return to_int_t(a) + to_int_t(b) * to_int_t(n_rows);
}

// Unique entries in a symmetric matrix with m, n dimensions
template <typename T1, typename T2>
constexpr int_t symmetric_different_size(const T1 m, const T2 n) noexcept
{
    const int_t M = to_int_t(m), N = to_int_t(n);
    return (M > N) ? gauss_sum(N) + N * (M - N) : gauss_sum(M) + M * (N - M);
}

// Unique ERI count with non-equal orbital sizes
template <typename T1, typename T2, typename T3, typename T4>
constexpr int_t get_flexible_n_eri(const T1 p_max, const T2 q_max, const T3 r_max,
                                   const T4 s_max) noexcept
{
    return symmetric_different_size(symmetric_different_size(p_max, q_max),
                                    symmetric_different_size(r_max, s_max));
}

std::vector<std::vector<OrbitalIdx>> extract_unique(
    const std::vector<std::vector<OrbitalIdx>> &exch_reachable)
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

template <typename Int>
constexpr typename std::enable_if<std::is_integral<Int>::value, double>::type
bytes_to_gib(Int bytes) noexcept
{
    return static_cast<double>(bytes) / (1024.0 * 1024 * 1024);
}

class Timer
{
  public:
    explicit Timer(const char *name)
        : _name(name), _start(std::chrono::high_resolution_clock::now())
    {
    }

    ~Timer()
    {
        if (LOG_LEVEL <= LogLevel::Info) {
            const auto duration = elapsed_s();
            std::cout << "[TIMER] " << _name << " finished in " << duration << " s\n";
        };
    }

    void print(const std::string &message = "Checkpoint") const
    {
        const auto duration = elapsed_s();
        std::cout << "[TIMER] " << _name << " - " << message << ": " << duration
                  << " s\n";
    }

  private:
    const char *_name;
    std::chrono::high_resolution_clock::time_point _start;

    double elapsed_s() const
    {
        using namespace std::chrono;
        const auto now = high_resolution_clock::now();
        return std::chrono::duration<double>(now - _start).count();
    }
};

#define PROFILE_FUNCTION() Timer timer(__func__)
#define PROFILE_SCOPE()                                                                \
    Timer timer(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " in " +      \
                __func__)

// Rebuilds an unordered_map to improve lookup performance by reducing collisions
// and improving memory layout. This is useful if the original map grew inefficiently.
template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          typename Allocator = std::allocator<std::pair<const Key, Value>>>
std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> rebuild_unordered_map(
    const std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> &original)
{
    std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> rebuilt;
    rebuilt.reserve(original.size()); // avoids rehashing during insertions
    for (const auto &pair : original)
        rebuilt.emplace(pair);
    return rebuilt;
}