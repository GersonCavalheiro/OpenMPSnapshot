#pragma once

namespace monolish {

namespace {

#if MONOLISH_USE_NVIDIA_GPU

void cusolver_ilu_create_descr(
matrix::CRS<float> &A, cusparseMatDescr_t &descr_M, csrilu02Info_t &info_M,
cusparseMatDescr_t &descr_L, csrsv2Info_t &info_L,
cusparseMatDescr_t &descr_U, csrsv2Info_t &info_U,
const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();
auto *d_csrVal = A.data();

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
{
cusparseCreateMatDescr(&descr_M);
cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

cusparseCreateMatDescr(&descr_L);
cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

cusparseCreateMatDescr(&descr_U);
cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

cusparseCreateCsrilu02Info(&info_M);
cusparseCreateCsrsv2Info(&info_L);
cusparseCreateCsrsv2Info(&info_U);
}

logger.func_out();
}

void cusolver_ilu_create_descr(
matrix::CRS<double> &A, cusparseMatDescr_t &descr_M, csrilu02Info_t &info_M,
cusparseMatDescr_t &descr_L, csrsv2Info_t &info_L,
cusparseMatDescr_t &descr_U, csrsv2Info_t &info_U,
const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();
auto *d_csrVal = A.data();

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
{
cusparseCreateMatDescr(&descr_M);
cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

cusparseCreateMatDescr(&descr_L);
cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

cusparseCreateMatDescr(&descr_U);
cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

cusparseCreateCsrilu02Info(&info_M);
cusparseCreateCsrsv2Info(&info_L);
cusparseCreateCsrsv2Info(&info_U);
}

logger.func_out();
}


int cusolver_ilu_get_buffersize(
matrix::CRS<float> &A, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseMatDescr_t &descr_L,
const csrsv2Info_t &info_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseOperation_t &trans_U, const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();
auto *d_csrVal = A.data();

int bufsize;
int bufsize_M;
int bufsize_L;
int bufsize_U;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
{
cusparseScsrilu02_bufferSize(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, &bufsize_M);
cusparseScsrsv2_bufferSize(handle, trans_L, M, nnz, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, &bufsize_L);
cusparseScsrsv2_bufferSize(handle, trans_U, M, nnz, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, &bufsize_U);

bufsize = std::max(bufsize_M, std::max(bufsize_L, bufsize_U));
}

logger.func_out();
return bufsize;
}

int cusolver_ilu_get_buffersize(
matrix::CRS<double> &A, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseMatDescr_t &descr_L,
const csrsv2Info_t &info_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseOperation_t &trans_U, const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();
auto *d_csrVal = A.data();

int bufsize;
int bufsize_M;
int bufsize_L;
int bufsize_U;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd)
{
cusparseDcsrilu02_bufferSize(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, &bufsize_M);
cusparseDcsrsv2_bufferSize(handle, trans_L, M, nnz, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, &bufsize_L);
cusparseDcsrsv2_bufferSize(handle, trans_U, M, nnz, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, &bufsize_U);

bufsize = std::max(bufsize_M, std::max(bufsize_L, bufsize_U));
}

logger.func_out();
return bufsize;
}


bool cusolver_ilu(
matrix::CRS<float> &A, float *d_csrVal, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseSolvePolicy_t &policy_M,
const cusparseMatDescr_t &descr_L, const csrsv2Info_t &info_L,
const cusparseSolvePolicy_t &policy_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseSolvePolicy_t &policy_U, const cusparseOperation_t &trans_U,
vector<double> buf, const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();

double *pBuffer = buf.data();
int structural_zero;
int numerical_zero;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd,     \
pBuffer)
{
cusparseScsrilu02_analysis(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, policy_M, pBuffer);
auto status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);

if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
throw std::runtime_error("ILU error.");
}

cusparseScsrsv2_analysis(handle, trans_L, M, nnz, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, policy_L,
pBuffer);

cusparseScsrsv2_analysis(handle, trans_U, M, nnz, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, policy_U,
pBuffer);

cusparseScsrilu02(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, policy_M, pBuffer);
status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
throw std::runtime_error("ILU error.");
}
}

logger.func_out();
return true;
}

bool cusolver_ilu(
matrix::CRS<double> &A, double *d_csrVal, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseSolvePolicy_t &policy_M,
const cusparseMatDescr_t &descr_L, const csrsv2Info_t &info_L,
const cusparseSolvePolicy_t &policy_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseSolvePolicy_t &policy_U, const cusparseOperation_t &trans_U,
vector<double> &buf, const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();

double *pBuffer = buf.data();

int structural_zero;
int numerical_zero;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd,     \
pBuffer)
{
cusparseDcsrilu02_analysis(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, policy_M, pBuffer);
auto status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);

if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
}

cusparseDcsrsv2_analysis(handle, trans_L, M, nnz, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, policy_L,
pBuffer);

cusparseDcsrsv2_analysis(handle, trans_U, M, nnz, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, policy_U,
pBuffer);

cusparseDcsrilu02(handle, M, nnz, descr_M, d_csrVal, d_csrRowPtr,
d_csrColInd, info_M, policy_M, pBuffer);
status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
}
}

logger.func_out();
return true;
}


bool cusolver_ilu_solve(
matrix::CRS<float> &A, float *d_csrVal, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseSolvePolicy_t &policy_M,
const cusparseMatDescr_t &descr_L, const csrsv2Info_t &info_L,
const cusparseSolvePolicy_t &policy_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseSolvePolicy_t &policy_U, const cusparseOperation_t &trans_U,
float *d_x, float *d_b, float *d_tmp, vector<double> &buf,
const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();

double *pBuffer = buf.data();
const float alpha = 1.0;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd,     \
d_x, d_b, d_tmp, pBuffer)
{
cusparseScsrsv2_solve(handle, trans_L, M, nnz, &alpha, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, d_b, d_tmp,
policy_L, pBuffer);

cusparseScsrsv2_solve(handle, trans_U, M, nnz, &alpha, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, d_tmp, d_x,
policy_U, pBuffer);
}

logger.func_out();
return true;
}

bool cusolver_ilu_solve(
matrix::CRS<double> &A, double *d_csrVal, const cusparseMatDescr_t &descr_M,
const csrilu02Info_t &info_M, const cusparseSolvePolicy_t &policy_M,
const cusparseMatDescr_t &descr_L, const csrsv2Info_t &info_L,
const cusparseSolvePolicy_t &policy_L, const cusparseOperation_t &trans_L,
const cusparseMatDescr_t &descr_U, const csrsv2Info_t &info_U,
const cusparseSolvePolicy_t &policy_U, const cusparseOperation_t &trans_U,
double *d_x, double *d_b, double *d_tmp, vector<double> &buf,
const cusparseHandle_t &handle) {

Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

auto M = A.get_row();
auto nnz = A.get_nnz();
int *d_csrRowPtr = A.row_ptr.data();
int *d_csrColInd = A.col_ind.data();

double *pBuffer = buf.data();
const double alpha = 1.0;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd,     \
d_x, d_b, d_tmp, pBuffer)
{
cusparseDcsrsv2_solve(handle, trans_L, M, nnz, &alpha, descr_L, d_csrVal,
d_csrRowPtr, d_csrColInd, info_L, d_b, d_tmp,
policy_L, pBuffer);

cusparseDcsrsv2_solve(handle, trans_U, M, nnz, &alpha, descr_U, d_csrVal,
d_csrRowPtr, d_csrColInd, info_U, d_tmp, d_x,
policy_U, pBuffer);
}

logger.func_out();
return true;
}

#endif

} 
} 
