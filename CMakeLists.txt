# ########################################################################
# Practical Course: GPU Programming in Computer Vision (SS 2018)
# Technical University of Munich, Computer Vision Group
# Bjoern Haefner, Robert Maier, David Schubert
# ########################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)

PROJECT(GpuLabCourse LANGUAGES C CXX)

# ------------------------------------------------------------------------
# CMake configuration
IF(NOT CMAKE_BUILD_TYPE)
	# CMAKE_BUILD_TYPE: RelWithDebInfo, Release or Debug
    SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

# CMake executable output path
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# ------------------------------------------------------------------------
# dependencies

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_35)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v --use_fast_math -O3 --compiler-options -Wall)

# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS
    core highgui imgproc video
    REQUIRED
)
IF(OPENCV_CORE_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -Wno-deprecated -pthread")
IF(CMAKE_COMPILER_IS_GNUCC)
    # Add -O0 to remove optimizations when using gcc
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
ENDIF()


# ------------------------------------------------------------------------
# preprocessor defines

# source directory
ADD_DEFINITIONS(-DGPULAB_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# ------------------------------------------------------------------------
# include directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

# ------------------------------------------------------------------------
# exercise applications

# exercise 3: gamma correction
CUDA_ADD_EXECUTABLE(ex3_gamma
        src/helper.cuh src/helper.cu
        src/gamma.cuh src/gamma.cu
        src/ex3_gamma.cu
)
TARGET_LINK_LIBRARIES(ex3_gamma ${OpenCV_LIBS})

# exercise 4: linear operators
CUDA_ADD_EXECUTABLE(ex4_linearoperator
        src/helper.cuh src/helper.cu
        src/gradient.cuh src/gradient.cu
        src/divergence.cuh src/divergence.cu
        src/norm.cuh src/norm.cu
        src/ex4_linearoperator.cu
)
TARGET_LINK_LIBRARIES(ex4_linearoperator ${OpenCV_LIBS})

# exercise 5/6: convolution
CUDA_ADD_EXECUTABLE(ex5_6_convolution
        src/helper.cuh src/helper.cu
        src/convolution.cuh src/convolution.cu
        src/ex5_6_convolution.cu
)
TARGET_LINK_LIBRARIES(ex5_6_convolution ${OpenCV_LIBS})

# exercise 7/8: structure tensor
CUDA_ADD_EXECUTABLE(ex7_8_structure_tensor
        src/helper.cuh src/helper.cu
        src/convolution.cuh src/convolution.cu
        src/structure_tensor.cuh src/structure_tensor.cu
        src/ex7_8_structure_tensor.cu
)
TARGET_LINK_LIBRARIES(ex7_8_structure_tensor ${OpenCV_LIBS})

# exercise 9: diffusion
CUDA_ADD_EXECUTABLE(ex9_diffusion
        src/helper.cuh src/helper.cu
        src/gradient.cuh src/gradient.cu
        src/divergence.cuh src/divergence.cu
        src/diffusion.cuh src/diffusion.cu
        src/convolution.cuh src/convolution.cu
        src/ex9_diffusion.cu
)
TARGET_LINK_LIBRARIES(ex9_diffusion ${OpenCV_LIBS})

# exercise 10: anisotropic diffusion
CUDA_ADD_EXECUTABLE(ex10_diffusion_anisotropic
        src/helper.cuh src/helper.cu
        src/gradient.cuh src/gradient.cu
        src/convolution.cuh src/convolution.cu
        src/structure_tensor.cuh src/structure_tensor.cu
        src/divergence.cuh src/divergence.cu
        src/diffusion.cuh src/diffusion.cu
        src/ex10_diffusion_anisotropic.cu
)
TARGET_LINK_LIBRARIES(ex10_diffusion_anisotropic ${OpenCV_LIBS})

# exercise 11/12: denoising + reduction
CUDA_ADD_EXECUTABLE(ex11_12_denoising
        src/helper.cuh src/helper.cu
        src/diffusion.cuh src/diffusion.cu
        src/energy.cuh src/energy.cu
        src/reduction.cuh src/reduction.cu
        src/ex11_12_denoising.cu
)
CUDA_ADD_CUBLAS_TO_TARGET(ex11_12_denoising)
TARGET_LINK_LIBRARIES(ex11_12_denoising ${OpenCV_LIBS})

# exercise 13: histogram
CUDA_ADD_EXECUTABLE(ex13_histogram
        src/helper.cuh src/helper.cu
        src/histogram.cuh src/histogram.cu
        src/ex13_histogram.cu
)
TARGET_LINK_LIBRARIES(ex13_histogram ${OpenCV_LIBS})
