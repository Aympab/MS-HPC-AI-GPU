add_executable(matx_cuda "")
target_sources(matx_cuda PRIVATE main.cu)

target_link_libraries(matx_cuda PUBLIC matx::matx)
# target_link_libraries(saxpy_cuda PUBLIC OpenMP::OpenMP_CXX)
# target_link_libraries(saxpy_cuda PUBLIC cuda::common)