#pragma once

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3>
void Dxpay_core(const F1 alpha, const F2 &x, F3 &y) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(x, y));
assert(util::is_same_device_mem_stat(x, y));

const double *xd = x.begin();
double *yd = y.begin();
auto size = x.size();

if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(size){0}; i < size; i++) {
yd[i] = xd[i] + alpha * yd[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(size){0}; i < size; i++) {
yd[i] = xd[i] + alpha * yd[i];
}
}
logger.func_out();
}

template <typename F1, typename F2, typename F3>
void Sxpay_core(const F1 alpha, const F2 &x, F3 &y) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(x, y));
assert(util::is_same_device_mem_stat(x, y));

const float *xd = x.begin();
float *yd = y.begin();
auto size = x.size();

if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(size){0}; i < size; i++) {
yd[i] = xd[i] + alpha * yd[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(size){0}; i < size; i++) {
yd[i] = xd[i] + alpha * yd[i];
}
}
logger.func_out();
}

} 

} 
