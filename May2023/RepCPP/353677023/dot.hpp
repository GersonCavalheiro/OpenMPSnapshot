#pragma once

namespace monolish {

namespace {
template <typename F1, typename F2> double Ddot_core(const F1 &x, const F2 &y) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(x, y));
assert(util::is_same_device_mem_stat(x, y));

double ans = 0;
const double *xd = x.begin();
const double *yd = y.begin();
const auto size = x.size();

if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
cublasHandle_t h;
internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
{ internal::check_CUDA(cublasDdot(h, size, xd, 1, yd, 1, &ans)); }
cublasDestroy(h);
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
ans = cblas_ddot(size, xd, 1, yd, 1);
}

#if MONOLISH_USE_MPI
mpi::comm &comm = mpi::comm::get_instance();
ans = comm.Allreduce(ans);
#endif

logger.func_out();
return ans;
}

template <typename F1, typename F2> float Sdot_core(const F1 &x, const F2 &y) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(x, y));
assert(util::is_same_device_mem_stat(x, y));

float ans = 0;
const float *xd = x.begin();
const float *yd = y.begin();
const auto size = x.size();

if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
cublasHandle_t h;
internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
{ internal::check_CUDA(cublasSdot(h, size, xd, 1, yd, 1, &ans)); }
cublasDestroy(h);
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
ans = cblas_sdot(size, xd, 1, yd, 1);
}

#if MONOLISH_USE_MPI
mpi::comm &comm = mpi::comm::get_instance();
ans = comm.Allreduce(ans);
#endif

logger.func_out();
return ans;
}

} 
} 
