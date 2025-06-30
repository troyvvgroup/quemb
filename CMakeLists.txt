cmake_minimum_required(VERSION 3.18)
project(quemb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
pybind11_add_module(eri_sparse_DF src/quemb/molbe/_cpp/eri_sparse_DF.cpp)

# Link against the Eigen library
target_include_directories(eri_sparse_DF SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/external/eigen)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(eri_sparse_DF PRIVATE CUDA::cublas CUDA::cuda_driver CUDA::cudart)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(eri_sparse_DF PRIVATE OpenMP::OpenMP_CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Add compile options based on build type
target_compile_options(eri_sparse_DF PRIVATE -g)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(eri_sparse_DF PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(eri_sparse_DF PRIVATE
        -g
        -O0

        # Warnings
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough

        # Runtime checks
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer

        -Werror
    )

    # Link-time flags for runtime sanitizers
    target_link_options(eri_sparse_DF PRIVATE
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
    )
endif()

install(TARGETS eri_sparse_DF
  LIBRARY DESTINATION quemb/molbe/_cpp
)
