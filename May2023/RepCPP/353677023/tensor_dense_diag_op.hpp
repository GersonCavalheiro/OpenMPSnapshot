#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace {

template <typename T>
void tensor_Dense_diag_add_core(tensor::tensor_Dense<T> &TENS, const T alpha) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] += alpha;
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] += alpha;
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_add_core(tensor::tensor_Dense<T> &TENS,
const size_t size, const T *vecd) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

assert(Len == size);

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] += vecd[i];
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] += vecd[i];
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_sub_core(tensor::tensor_Dense<T> &TENS, const T alpha) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] -= alpha;
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] -= alpha;
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_sub_core(tensor::tensor_Dense<T> &TENS,
const size_t size, const T *vecd) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

assert(Len == size);

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] -= vecd[i];
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] -= vecd[i];
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_mul_core(tensor::tensor_Dense<T> &TENS, const T alpha) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] *= alpha;
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] *= alpha;
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_mul_core(tensor::tensor_Dense<T> &TENS,
const size_t size, const T *vecd) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

assert(Len == size);

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] *= vecd[i];
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] *= vecd[i];
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_div_core(tensor::tensor_Dense<T> &TENS, const T alpha) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] /= alpha;
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] /= alpha;
}
}

logger.func_out();
}

template <typename T>
void tensor_Dense_diag_div_core(tensor::tensor_Dense<T> &TENS,
const size_t size, const T *vecd) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

T *vald = TENS.data();
size_t N = 1;
size_t shift = 0;
size_t Len = TENS.get_nnz();
auto shape = TENS.get_shape();
for (int i = 0; i < shape.size(); ++i) {
shift = shift + N;
N *= shape[i];
Len = std::min(Len, shape[i]);
}

assert(Len == size);

if (TENS.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU 
#pragma omp target teams distribute parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] /= vecd[i];
}
#else
throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(Len){0}; i < Len; i++) {
vald[shift * i] /= vecd[i];
}
}

logger.func_out();
}

} 
} 
