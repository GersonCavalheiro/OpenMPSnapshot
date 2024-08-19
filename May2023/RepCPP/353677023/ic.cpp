#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"
#include "./kernel/ic_kernel.hpp"

namespace monolish {

template <typename MATRIX, typename T>
void equation::IC<MATRIX, T>::create_precond(MATRIX &A) {
Logger &logger = Logger::get_instance();
logger.solver_in(monolish_func);

if (A.get_row() != A.get_col()) {
throw std::runtime_error("error A.row != A.col");
}

if (A.get_device_mem_stat() != true) {
throw std::runtime_error("IC on CPU does not impl.");
}

#if MONOLISH_USE_NVIDIA_GPU
cusparseHandle_t handle;
cusparseCreate(&handle);
cudaDeviceSynchronize();

cusparseMatDescr_t descr_M = 0;
csric02Info_t info_M = 0;
cusparseMatDescr_t descr_L = 0;
csrsv2Info_t info_L = 0;
csrsv2Info_t info_Lt = 0;

const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

cusolver_ic_create_descr(A, descr_M, info_M, descr_L, info_L, info_Lt,
handle);

bufsize = cusolver_ic_get_buffersize(A, descr_M, info_M, descr_L, info_L,
trans_L, info_Lt, trans_Lt, handle);

buf.device_free();
buf.resize(bufsize);
buf.send();

this->precond.M.device_free();
this->precond.M.resize(A.get_nnz());
#pragma omp parallel for
for (size_t i = 0; i < A.get_nnz(); i++) {
this->precond.M.data()[i] = A.data()[i];
}
this->precond.M.send();

cusolver_ic(A, this->precond.M.data(), descr_M, info_M, policy_M, descr_L,
info_L, policy_L, trans_L, info_Lt, policy_Lt, trans_Lt, buf,
handle);

matM = descr_M;
infoM = info_M;

matL = descr_L;
infoL = info_L;
infoLt = info_Lt;

cusparse_handle = handle;

this->precond.A = &A;

zbuf.device_free();
zbuf.resize(A.get_row());
zbuf.send();
#else
throw std::runtime_error("IC on CPU does not impl.");
#endif

logger.solver_out();
}

template void
equation::IC<matrix::CRS<float>, float>::create_precond(matrix::CRS<float> &A);
template void equation::IC<matrix::CRS<double>, double>::create_precond(
matrix::CRS<double> &A);



template <typename MATRIX, typename T>
void equation::IC<MATRIX, T>::apply_precond(const vector<T> &r, vector<T> &z) {
Logger &logger = Logger::get_instance();
logger.solver_in(monolish_func);

#if MONOLISH_USE_NVIDIA_GPU
double start = omp_get_wtime();
T *d_z = z.data();
T *d_r = (T *)r.data();
T *d_tmp = zbuf.data();

cusparseHandle_t handle = (cusparseHandle_t)cusparse_handle;

cusparseMatDescr_t descr_M = (cusparseMatDescr_t)matM;
csric02Info_t info_M = (csric02Info_t)infoM;
cusparseMatDescr_t descr_L = (cusparseMatDescr_t)matL;
csrsv2Info_t info_L = (csrsv2Info_t)infoL;
csrsv2Info_t info_Lt = (csrsv2Info_t)infoLt;

const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

cusolver_ic_solve(*this->precond.A, this->precond.M.data(), descr_M, info_M,
policy_M, descr_L, info_L, policy_L, trans_L, info_Lt,
policy_Lt, trans_Lt, d_z, d_r, d_tmp, buf, handle);

#else
throw std::runtime_error("IC on CPU does not impl.");
#endif

logger.solver_out();
}

template void
equation::IC<matrix::CRS<float>, float>::apply_precond(const vector<float> &r,
vector<float> &z);
template void equation::IC<matrix::CRS<double>, double>::apply_precond(
const vector<double> &r, vector<double> &z);



template <typename MATRIX, typename T>
int equation::IC<MATRIX, T>::cusparse_IC(MATRIX &A, vector<T> &x,
vector<T> &b) {
Logger &logger = Logger::get_instance();
logger.solver_in(monolish_func);

if (A.get_row() != A.get_col()) {
throw std::runtime_error("error, A.row != A.col");
}
if (A.get_device_mem_stat() != x.get_device_mem_stat() &&
A.get_device_mem_stat() != b.get_device_mem_stat()) {
throw std::runtime_error("error, A.get_device_mem_stat != "
"x.get_device_mem_stat != b.get_device_mem_stat");
}

if (A.get_device_mem_stat() != true) {
throw std::runtime_error("IC on CPU does not impl.");
}

#if MONOLISH_USE_NVIDIA_GPU
T *d_x = x.data();
T *d_b = b.data();

monolish::vector<T> tmp(x.size(), 0.0);
tmp.send();
T *d_tmp = tmp.data();

cusparseHandle_t handle;
cusparseCreate(&handle);
cudaDeviceSynchronize();

cusparseMatDescr_t descr_M = (cusparseMatDescr_t)matM;
csric02Info_t info_M = (csric02Info_t)infoM;
cusparseMatDescr_t descr_L = (cusparseMatDescr_t)matL;
csrsv2Info_t info_L = (csrsv2Info_t)infoL;
csrsv2Info_t info_Lt = (csrsv2Info_t)infoLt;

const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

cusolver_ic_create_descr(A, descr_M, info_M, descr_L, info_L, info_Lt,
handle);
bufsize = cusolver_ic_get_buffersize(A, descr_M, info_M, descr_L, info_L,
trans_L, info_Lt, trans_Lt, handle);

monolish::vector<T> tmpval(A.data(), A.data() + A.get_nnz());
tmpval.send();

monolish::vector<double> buf(bufsize);
buf.send();

cusolver_ic(A, tmpval.data(), descr_M, info_M, policy_M, descr_L, info_L,
policy_L, trans_L, info_Lt, policy_Lt, trans_Lt, buf, handle);

cusolver_ic_solve(A, tmpval.data(), descr_M, info_M, policy_M, descr_L,
info_L, policy_L, trans_L, info_Lt, policy_Lt, trans_Lt,
d_x, d_b, d_tmp, buf, handle);

cusparseDestroyMatDescr(descr_M);
cusparseDestroyMatDescr(descr_L);
cusparseDestroyCsric02Info(info_M);
cusparseDestroyCsrsv2Info(info_L);
cusparseDestroyCsrsv2Info(info_Lt);
cusparseDestroy(handle);
#else
throw std::runtime_error("IC on CPU does not impl.");
#endif

logger.solver_out();
return MONOLISH_SOLVER_SUCCESS;
}

template int equation::IC<matrix::CRS<float>, float>::cusparse_IC(
matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::IC<matrix::CRS<double>, double>::cusparse_IC(
matrix::CRS<double> &A, vector<double> &x, vector<double> &b);



template <typename MATRIX, typename T>
int equation::IC<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
Logger &logger = Logger::get_instance();
logger.solver_in(monolish_func);

int ret = 0;
if (this->get_lib() == 0) {
ret = cusparse_IC(A, x, b);
}

logger.solver_out();
return ret; 
}


template int equation::IC<matrix::CRS<float>, float>::solve(
matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::IC<matrix::CRS<double>, double>::solve(
matrix::CRS<double> &A, vector<double> &x, vector<double> &b);


template <typename MATRIX, typename T> equation::IC<MATRIX, T>::~IC() {
#if MONOLISH_USE_NVIDIA_GPU
cusparseDestroyMatDescr((cusparseMatDescr_t)matM);
cusparseDestroyMatDescr((cusparseMatDescr_t)matL);

cusparseDestroyCsric02Info((csric02Info_t)infoM);
cusparseDestroyCsrsv2Info((csrsv2Info_t)infoL);
cusparseDestroyCsrsv2Info((csrsv2Info_t)infoLt);

cusparseDestroy((cusparseHandle_t)cusparse_handle);
#endif
}

template equation::IC<matrix::CRS<float>, float>::~IC();
template equation::IC<matrix::CRS<double>, double>::~IC();

} 
