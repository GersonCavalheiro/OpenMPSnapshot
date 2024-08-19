#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::print_all(bool force_cpu) const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
const T *vald = data();
#pragma omp target
for (auto i = decltype(get_row()){0}; i < get_row(); i++) {
for (auto j = decltype(get_col()){0}; j < get_col(); j++) {
printf("%lu %lu %f\n", i + 1, j + 1, vald[i * get_col() + j]);
}
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
for (auto i = decltype(get_row()){0}; i < get_row(); i++) {
for (auto j = decltype(get_col()){0}; j < get_col(); j++) {
std::cout << i + 1 << " " << j + 1 << " " << data()[i * get_col() + j]
<< std::endl;
}
}
}

logger.util_out();
}
template void Dense<double>::print_all(bool force_cpu) const;
template void Dense<float>::print_all(bool force_cpu) const;

} 
} 
