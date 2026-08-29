cmake_minimum_required(VERSION 3.24)

project(pnu_depth_registration_cuda VERSION 0.1.0 LANGUAGES CXX CUDA)

set(PNU_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build")

add_library(pnu_depth_registration_cuda SHARED
  src/pnu_depth_registration_cuda.cu
)

target_include_directories(pnu_depth_registration_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(pnu_depth_registration_cuda PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(pnu_depth_registration_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;-lineinfo>
)

set_target_properties(pnu_depth_registration_cuda PROPERTIES
  CUDA_ARCHITECTURES "${PNU_CUDA_ARCHITECTURES}"
  CUDA_RUNTIME_LIBRARY Static
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)

install(TARGETS pnu_depth_registration_cuda
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(FILES include/pnu_depth_registration_cuda.h DESTINATION include)
