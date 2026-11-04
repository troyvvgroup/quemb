cmake_minimum_required(VERSION 3.18)
project(quemb LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the module
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
pybind11_add_module(eri_sparse_DF src/quemb/molbe/_cpp/eri_sparse_DF.cpp)

# Include Eigen
target_include_directories(eri_sparse_DF SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/external/eigen)

# Optional CUDA (cuBLAS), for the sparse-DF-gpu variants.
#   -DCUDA=AUTO (default): use CUDA if the toolkit is found.
#   -DCUDA=ON:   require CUDA, fail if the toolkit is not found.
#   -DCUDA=OFF:  build without CUDA.
set(CUDA "AUTO" CACHE STRING "CUDA support: AUTO, ON or OFF")
set_property(CACHE CUDA PROPERTY STRINGS AUTO ON OFF)
if(NOT CUDA MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "CUDA must be AUTO, ON or OFF, got '${CUDA}'")
endif()
if(NOT CUDA STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
endif()
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found: building with cuBLAS")
    target_link_libraries(eri_sparse_DF PRIVATE CUDA::cublas CUDA::cuda_driver CUDA::cudart)
    target_compile_definitions(eri_sparse_DF PRIVATE USE_CUDA)
elseif(CUDA STREQUAL "ON")
    message(FATAL_ERROR "CUDA=ON, but no CUDA toolkit found. "
        "Point CMake to it with -DCUDAToolkit_ROOT=..., or build with -DCUDA=AUTO or OFF.")
else()
    message(STATUS "Building without CUDA (CUDA=${CUDA})")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(eri_sparse_DF PRIVATE OpenMP::OpenMP_CXX)

# Eigen itself must never use BLAS: with EIGEN_USE_BLAS every Eigen GEMM, also inside our
# OpenMP loops, goes to the BLAS library, and a multithreaded BLAS oversubscribes the cores.
if(CMAKE_CXX_FLAGS MATCHES "EIGEN_USE_(BLAS|MKL_ALL)")
    message(FATAL_ERROR "Eigen must not use BLAS (oversubscribes the OpenMP loops), "
        "remove EIGEN_USE_BLAS/EIGEN_USE_MKL_ALL from CMAKE_CXX_FLAGS. "
        "BLAS is only called explicitly, see -DBLAS=AUTO|ON|OFF.")
endif()

# Optional multithreaded BLAS which runs outside OpenMP regions.
# Eigen itself never uses BLAS (see the check above).
# A multithreaded BLAS is required; a single-threaded one (e.g. the reference BLAS) is slower
# than Eigen. A sequential OpenBLAS is rejected, other vendors are not checked for parallelisation.
#   -DBLAS=AUTO (default): use BLAS if found, otherwise Eigen only.
#   -DBLAS=ON:   require BLAS, fail otherwise.
#   -DBLAS=OFF:  no BLAS, Eigen only.
# The vendor can be chosen with -DBLA_VENDOR=..., e.g. OpenBLAS or Intel10_64lp (threaded MKL).
set(BLAS "AUTO" CACHE STRING "Multithreaded BLAS for explicit BLAS calls: AUTO, ON or OFF")
set_property(CACHE BLAS PROPERTY STRINGS AUTO ON OFF)
if(NOT BLAS MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "BLAS must be AUTO, ON or OFF, got '${BLAS}'")
endif()

if(NOT BLAS STREQUAL "OFF")
    find_package(BLAS QUIET)
endif()
if(BLAS_FOUND)
    string(REPLACE ";" " " _blas_libs "${BLAS_LIBRARIES}")
    if(BLAS_LIBRARIES MATCHES "openblas")
        # Ask OpenBLAS itself: 0 = sequential, 1 = pthreads, 2 = OpenMP.
        file(WRITE ${CMAKE_BINARY_DIR}/check_openblas_parallel.cpp
            "#include <cstdio>\n"
            "extern \"C\" int openblas_get_parallel();\n"
            "int main() { std::printf(\"%d\", openblas_get_parallel()); return 0; }\n")
        try_run(_run_result _compile_result ${CMAKE_BINARY_DIR}/check_openblas_parallel
            ${CMAKE_BINARY_DIR}/check_openblas_parallel.cpp
            LINK_LIBRARIES BLAS::BLAS
            RUN_OUTPUT_VARIABLE _openblas_parallel)
        if(_openblas_parallel STREQUAL "0")
            message(FATAL_ERROR "Sequential OpenBLAS found (${_blas_libs}); "
                "a multithreaded BLAS is required. Install the pthreads or OpenMP variant "
                "and choose it with -DBLA_VENDOR=..., or build with -DBLAS=OFF.")
        endif()
    endif()
    message(STATUS "BLAS: ${_blas_libs}")
    target_link_libraries(eri_sparse_DF PRIVATE BLAS::BLAS)
    target_compile_definitions(eri_sparse_DF PRIVATE QUEMB_USE_BLAS)
elseif(BLAS STREQUAL "ON")
    message(FATAL_ERROR "BLAS=ON, but no BLAS found. Install a multithreaded BLAS "
        "(e.g. OpenBLAS or MKL), choose it with -DBLA_VENDOR=..., or build with -DBLAS=AUTO or OFF.")
else()
    message(STATUS "No BLAS (BLAS=${BLAS}): Eigen only")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

target_compile_features(eri_sparse_DF PRIVATE cxx_std_17)

# =====================
# Compile flags
# =====================

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_definitions(eri_sparse_DF PRIVATE CLANG=1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(eri_sparse_DF PRIVATE GCC=1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_definitions(eri_sparse_DF PRIVATE INTEL=1)
else()
    target_compile_definitions(eri_sparse_DF PRIVATE UNKNOWN_COMPILER=1)
endif()



# =====================
# Optimization and Tuning
# =====================

# Enable debug symbols always (fine-grained tuning later)
target_compile_options(eri_sparse_DF PRIVATE -g)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(eri_sparse_DF PRIVATE -O3 -ffast-math)

    if(NOT MSVC)
        target_compile_options(eri_sparse_DF PRIVATE -march=native)
    endif()

elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(eri_sparse_DF PRIVATE -O2)

elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(eri_sparse_DF PRIVATE -O0)
endif()

# =====================
# Warnings
# =====================

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
    target_compile_options(eri_sparse_DF PRIVATE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor
        -Wold-style-cast -Wcast-align
        -Wunused -Woverloaded-virtual
        -Wconversion -Wno-sign-conversion
        -Wnull-dereference -Wdouble-promotion
        -Wformat=2 -Wimplicit-fallthrough
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(eri_sparse_DF PRIVATE -Werror)
    endif()
endif()

# =====================
# Installation
# =====================

install(TARGETS eri_sparse_DF
    LIBRARY DESTINATION quemb/molbe/_cpp
    )
