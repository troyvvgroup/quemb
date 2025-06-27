cmake_minimum_required(VERSION 3.18)
project(quemb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
# find_package(Kokkos REQUIRED)  # Optional: Only if you're using Kokkos

# Define the extension module
pybind11_add_module(eri_sparse_DF src/quemb/molbe/_cpp/eri_sparse_DF.cpp)

# Link against the Eigen library
target_include_directories(eri_sparse_DF PRIVATE ${CMAKE_SOURCE_DIR}/external/Eigen)


# Optional: Link Kokkos or other libraries
# target_link_libraries(eri_sparse_DF PRIVATE Kokkos::kokkos)

# Optional: Add compiler flags
target_compile_options(eri_sparse_DF PRIVATE -O3 -march=native -ffast-math -fopenmp)
