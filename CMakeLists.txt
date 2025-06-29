cmake_minimum_required(VERSION 3.18)
project(quemb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Kokkos REQUIRED)  # Optional: Only if you're using Kokkos
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)


message(STATUS "Kokkos found: ${Kokkos_DIR}")


# Define the extension module
pybind11_add_module(eri_sparse_DF src/quemb/molbe/_cpp/eri_sparse_DF.cpp)

# Link against the Eigen library
target_include_directories(eri_sparse_DF PRIVATE ${CMAKE_SOURCE_DIR}/external/eigen)
target_include_directories(eri_sparse_DF PRIVATE /usr/include)

target_link_libraries(eri_sparse_DF PRIVATE CUDA::cublas)

target_link_libraries(eri_sparse_DF PRIVATE OpenMP::OpenMP_CXX)

# Optional: Link Kokkos or other libraries
target_link_libraries(eri_sparse_DF PRIVATE Kokkos::kokkos)

# Optional: Add compiler flags
target_compile_options(eri_sparse_DF PRIVATE -O3 -march=native -ffast-math)

install(TARGETS eri_sparse_DF
  LIBRARY DESTINATION quemb/molbe/_cpp
)
