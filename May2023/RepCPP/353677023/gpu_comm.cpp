#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> void vector<T>::send() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
const T *d = data();
const auto N = size();

if (gpu_status == true) {
#pragma omp target update to(d [0:N])
} else {
#pragma omp target enter data map(to : d [0:N])
gpu_status = true;
}
#endif
logger.util_out();
}

template <typename T> void vector<T>::recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *d = data();
const auto N = size();

#pragma omp target exit data map(from : d [0:N])

gpu_status = false;
}
#endif
logger.util_out();
}

template <typename T> void vector<T>::nonfree_recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *d = data();
const auto N = size();
#pragma omp target update from(d [0:N])
}
#endif
logger.util_out();
}

template <typename T> void vector<T>::device_free() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
const T *d = data();
const auto N = size();
#pragma omp target exit data map(release : d [0:N])
gpu_status = false;
}
#endif
logger.util_out();
}

template void vector<float>::send() const;
template void vector<double>::send() const;

template void vector<float>::recv();
template void vector<double>::recv();

template void vector<float>::nonfree_recv();
template void vector<double>::nonfree_recv();

template void vector<float>::device_free() const;
template void vector<double>::device_free() const;

template <typename T> void matrix::CRS<T>::send() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
const T *vald = data();
const auto *cold = col_ind.data();
const auto *rowd = row_ptr.data();
const auto N = get_row();
const auto nnz = get_nnz();

if (gpu_status == true) {
#pragma omp target update to(vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
} else {
#pragma omp target enter data map(to                                           \
:                                            \
vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
gpu_status = true;
}
#endif
logger.util_out();
}

template <typename T> void matrix::CRS<T>::recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto *cold = col_ind.data();
auto *rowd = row_ptr.data();
auto N = get_row();
auto nnz = get_nnz();

#pragma omp target exit data map(from                                          \
: vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
gpu_status = false;
}
#endif
logger.util_out();
}

template <typename T> void matrix::CRS<T>::nonfree_recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto *cold = col_ind.data();
auto *rowd = row_ptr.data();
auto N = get_row();
auto nnz = get_nnz();

#pragma omp target update from(vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
}
#endif
logger.util_out();
}

template <typename T> void matrix::CRS<T>::device_free() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
const T *vald = data();
const auto *cold = col_ind.data();
const auto *rowd = row_ptr.data();
const auto N = get_row();
const auto nnz = get_nnz();

#pragma omp target exit data map(release                                       \
: vald [0:nnz], cold [0:nnz], rowd [0:N + 1])

gpu_status = false;
}
#endif
logger.util_out();
}
template void matrix::CRS<float>::send() const;
template void matrix::CRS<double>::send() const;

template void matrix::CRS<float>::recv();
template void matrix::CRS<double>::recv();

template void matrix::CRS<float>::nonfree_recv();
template void matrix::CRS<double>::nonfree_recv();

template void matrix::CRS<float>::device_free() const;
template void matrix::CRS<double>::device_free() const;
template <typename T> void matrix::Dense<T>::send() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
const T *vald = data();
const auto nnz = get_nnz();

if (gpu_status == true) {
#pragma omp target update to(vald [0:nnz])
} else {
#pragma omp target enter data map(to : vald [0:nnz])
gpu_status = true;
}
#endif
logger.util_out();
}

template <typename T> void matrix::Dense<T>::recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto nnz = get_nnz();

#pragma omp target exit data map(from : vald [0:nnz])
gpu_status = false;
}
#endif
logger.util_out();
}

template <typename T> void matrix::Dense<T>::nonfree_recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto nnz = get_nnz();

#pragma omp target update from(vald [0:nnz])
}
#endif
logger.util_out();
}

template <typename T> void matrix::Dense<T>::device_free() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
const T *vald = data();
const auto nnz = get_nnz();

#pragma omp target exit data map(release : vald [0:nnz])

gpu_status = false;
}
#endif
logger.util_out();
}
template void matrix::Dense<float>::send() const;
template void matrix::Dense<double>::send() const;

template void matrix::Dense<float>::recv();
template void matrix::Dense<double>::recv();

template void matrix::Dense<float>::nonfree_recv();
template void matrix::Dense<double>::nonfree_recv();

template void matrix::Dense<float>::device_free() const;
template void matrix::Dense<double>::device_free() const;

template <typename T> void tensor::tensor_Dense<T>::send() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
const T *vald = data();
const auto nnz = get_nnz();

if (gpu_status == true) {
#pragma omp target update to(vald [0:nnz])
} else {
#pragma omp target enter data map(to : vald [0:nnz])
gpu_status = true;
}
#endif
logger.util_out();
}

template <typename T> void tensor::tensor_Dense<T>::recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto nnz = get_nnz();

#pragma omp target exit data map(from : vald [0:nnz])
gpu_status = false;
}
#endif
logger.util_out();
}

template <typename T> void tensor::tensor_Dense<T>::nonfree_recv() {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
T *vald = data();
auto nnz = get_nnz();

#pragma omp target update from(vald [0:nnz])
}
#endif
logger.util_out();
}

template <typename T> void tensor::tensor_Dense<T>::device_free() const {
Logger &logger = Logger::get_instance();
logger.util_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
if (gpu_status == true) {
const T *vald = data();
const auto nnz = get_nnz();

#pragma omp target exit data map(release : vald [0:nnz])

gpu_status = false;
}
#endif
logger.util_out();
}
template void tensor::tensor_Dense<float>::send() const;
template void tensor::tensor_Dense<double>::send() const;

template void tensor::tensor_Dense<float>::recv();
template void tensor::tensor_Dense<double>::recv();

template void tensor::tensor_Dense<float>::nonfree_recv();
template void tensor::tensor_Dense<double>::nonfree_recv();

template void tensor::tensor_Dense<float>::device_free() const;
template void tensor::tensor_Dense<double>::device_free() const;

} 
