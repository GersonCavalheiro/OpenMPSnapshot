#include "../../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish::internal {
void vadd(const size_t N, const double *a, const double alpha, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] + alpha;
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] + alpha;
}
}
logger.func_out();
}

void vsub(const size_t N, const double *a, const double alpha, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] - alpha;
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] - alpha;
}
}
logger.func_out();
}

void vmul(const size_t N, const double *a, const double alpha, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] * alpha;
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] * alpha;
}
}
logger.func_out();
}

void vdiv(const size_t N, const double *a, const double alpha, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] / alpha;
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] / alpha;
}
}
logger.func_out();
}

void vadd(const size_t N, const double *a, const double *b, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] + b[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#if MONOLISH_USE_MKL
vdAdd(N, a, b, y);
#else
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] + b[i];
}
#endif
}
logger.func_out();
}

void vsub(const size_t N, const double *a, const double *b, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] - b[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#if MONOLISH_USE_MKL
vdSub(N, a, b, y);
#else
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] - b[i];
}
#endif
}
logger.func_out();
}

void vmul(const size_t N, const double *a, const double *b, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] * b[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#if MONOLISH_USE_MKL
vdMul(N, a, b, y);
#else
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] * b[i];
}
#endif
}
logger.func_out();
}

void vdiv(const size_t N, const double *a, const double *b, double *y,
bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] / b[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#if MONOLISH_USE_MKL
vdDiv(N, a, b, y);
#else
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = a[i] / b[i];
}
#endif
}
logger.func_out();
}


void vcopy(const size_t N, const double *a, double *y, bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
cublasHandle_t h;
internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(a, y)
{ internal::check_CUDA(cublasDcopy(h, N, a, 1, y, 1)); }
cublasDestroy(h);
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
cblas_dcopy(N, a, 1, y, 1);
}
logger.func_out();
}

void vbroadcast(const size_t N, double alpha, double *y, bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = alpha;
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = alpha;
}
}
logger.func_out();
}

bool vequal(const size_t N, const double *a, const double *y, bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

bool ans = true;

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
if (y[i] != a[i]) {
ans = false;
}
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
if (y[i] != a[i]) {
ans = false;
}
}
}
logger.func_out();
return ans;
}

void vreciprocal(const size_t N, const double *a, double *y, bool gpu_status) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = 1.0 / a[i];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(N){0}; i < N; i++) {
y[i] = 1.0 / a[i];
}
}
logger.func_out();
}
} 
