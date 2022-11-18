cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# SET (CMAKE_C_COMPILER "clang")
# SET (CMAKE_CXX_COMPILER "clang++")

project(amx_conv3d)

# PyTorch binary dependency, Must specify
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# TODO: Force IOMP, default
if(DEFINED ENV{CONDA_PREFIX})
    link_directories($ENV{CONDA_PREFIX}/lib)
endif()


set(amx_src src/amx_init.cpp src/i_gemm_tpp.hpp src/test_gemm.cpp)
add_compile_options(-fPIC -fopenmp -O3 -std=c++14 -Wno-unused-function -march=native -mfma -Dusercp)

add_executable(amx_conv main.cpp ${amx_src})
target_link_libraries(amx_conv libomp.so "${TORCH_LIBRARIES}")
set_property(TARGET amx_conv PROPERTY CXX_STANDARD 14)

