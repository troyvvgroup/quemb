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

# Detect CUDA
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(eri_sparse_DF PRIVATE CUDA::cublas CUDA::cuda_driver CUDA::cudart)
    target_compile_definitions(eri_sparse_DF PRIVATE USE_CUDA)
endif()

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(eri_sparse_DF PRIVATE OpenMP::OpenMP_CXX)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# =====================
# Common flags
# =====================
target_compile_features(eri_sparse_DF PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_definitions(eri_sparse_DF PRIVATE CLANG=1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_definitions(eri_sparse_DF PRIVATE GCC=1)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_definitions(eri_sparse_DF PRIVATE INTEL=1)
else()
    target_compile_definitions(eri_sparse_DF PRIVATE UNKNOWN_COMPILER=1)
endif()


# Enable debug symbols always (fine-grained tuning later)
target_compile_options(eri_sparse_DF PRIVATE -g)

# =====================
# Optimization per config
# =====================
target_compile_options(eri_sparse_DF PRIVATE
    $<$<CONFIG:Release>:-O3 -ffast-math>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:Debug>:-O0>
)

# Native tuning (except MSVC)
if (NOT MSVC)
    target_compile_options(eri_sparse_DF PRIVATE
        $<$<CONFIG:Release>:-march=native>
    )
endif()

# =====================
# Warnings (except MSVC)
# =====================
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|Intel")
    target_compile_options(eri_sparse_DF PRIVATE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor
        -Wold-style-cast -Wcast-align
        -Wunused -Woverloaded-virtual
        -Wconversion -Wno-sign-conversion
        -Wnull-dereference -Wdouble-promotion
        -Wformat=2 -Wimplicit-fallthrough
        $<$<CONFIG:Debug>:-Werror>
    )
endif()

# =====================
# Sanitizers (Debug only)
# =====================
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    target_compile_options(eri_sparse_DF PRIVATE
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
    )
    target_link_options(eri_sparse_DF PRIVATE
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer
    )
endif()

# =====================
# Installation
# =====================
install(TARGETS eri_sparse_DF
    LIBRARY DESTINATION quemb/molbe/_cpp
)
