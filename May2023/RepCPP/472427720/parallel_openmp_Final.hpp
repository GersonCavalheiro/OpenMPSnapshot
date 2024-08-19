#pragma once
#include "shared_struct.hpp"
#include <omp.h>
#include <chrono>

std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

template <typename mat_T>
void openmp_spmv_mutex_var(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
}

template <>
void openmp_spmv_mutex_var<COO>(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
omp_lock_t mutex_var;
omp_init_lock(&mutex_var);
#pragma omp parallel num_threads(num_thread) default(none) shared(alpha, mat_A, vec_x, beta, vec_b, mutex_var)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t nz_idx = 0; nz_idx < mat_A.mat_data.num_nz; ++nz_idx)
{
double temp_val = (alpha * mat_A.mat_elements[nz_idx].val * vec_x[mat_A.mat_elements[nz_idx].col_idx]);
omp_set_lock(&mutex_var);
vec_b[mat_A.mat_elements[nz_idx].row_idx] += temp_val;
omp_unset_lock(&mutex_var);
}
}
omp_destroy_lock(&mutex_var);
}

template <>
void openmp_spmv_mutex_var<CSC>(
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
omp_lock_t mutex_var;
omp_init_lock(&mutex_var);
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; idx_nz++)
{
double temp_Val = (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
omp_set_lock(&mutex_var);
vec_b[mat_A.row_and_val[idx_nz].idx] += temp_Val;
omp_unset_lock(&mutex_var);
}
}
}
omp_destroy_lock(&mutex_var);
}

template <typename mat_T>
void openmp_spmv_mutex_arr(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
}

template <>
void openmp_spmv_mutex_arr<COO>(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<omp_lock_t> mutex_arr(vec_x.size());
for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_init_lock(&(mutex_arr[idx_mutex]));
}

#pragma omp parallel num_threads(num_thread) default(none) shared(alpha, mat_A, vec_x, beta, vec_b, mutex_arr)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t nz_idx = 0; nz_idx < mat_A.mat_data.num_nz; ++nz_idx)
{
double temp_val = (alpha * mat_A.mat_elements[nz_idx].val * vec_x[mat_A.mat_elements[nz_idx].col_idx]);
omp_set_lock(&(mutex_arr[mat_A.mat_elements[nz_idx].row_idx]));
vec_b[mat_A.mat_elements[nz_idx].row_idx] += temp_val;
omp_unset_lock(&(mutex_arr[mat_A.mat_elements[nz_idx].row_idx]));
}
}

for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_destroy_lock(&(mutex_arr[idx_mutex]));
}
}

template <>
void openmp_spmv_mutex_arr<CSC>(
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<omp_lock_t> mutex_arr(vec_x.size());
for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_init_lock(&(mutex_arr[idx_mutex]));
}

#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; idx_nz++)
{
double temp_Val = (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
omp_set_lock(&(mutex_arr[mat_A.row_and_val[idx_nz].idx]));
vec_b[mat_A.row_and_val[idx_nz].idx] += temp_Val;
omp_unset_lock(&(mutex_arr[mat_A.row_and_val[idx_nz].idx]));
}
}
}

for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_destroy_lock(&(mutex_arr[idx_mutex]));
}
}

template <typename mat_T>
void openmp_spmv_atomic(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
}

template <>
void openmp_spmv_atomic<COO>(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t nz_idx = 0; nz_idx < mat_A.mat_data.num_nz; ++nz_idx)
{
double temp_val = alpha * mat_A.mat_elements[nz_idx].val * vec_x[mat_A.mat_elements[nz_idx].col_idx];
#pragma omp atomic
vec_b[mat_A.mat_elements[nz_idx].row_idx] += temp_val;
}
}
}

template <>
void openmp_spmv_atomic<CSC>(
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t vec_b_idx = 0; vec_b_idx < vec_b.size(); ++vec_b_idx)
{
vec_b[vec_b_idx] = beta * vec_b[vec_b_idx];
}

#pragma omp for
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; idx_nz++)
{
double temp_Val = (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
#pragma omp atomic
vec_b[mat_A.row_and_val[idx_nz].idx] += temp_Val;
}
}
}
}

template <typename mat_T>
void openmp_spmv_none(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
}

template <>
void openmp_spmv_none<CSR>(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t idx_r_ptr = 0; idx_r_ptr < mat_A.mat_data.num_row; idx_r_ptr++)
{
double temp_val = 0.0;
vec_b[idx_r_ptr] = beta * vec_b[idx_r_ptr];
for (uint32_t idx_nz = mat_A.row_ptr[idx_r_ptr]; idx_nz < mat_A.row_ptr[idx_r_ptr + 1]; idx_nz++)
{
temp_val += (mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_r_ptr] = alpha * temp_val + beta * vec_b[idx_r_ptr];
}
}
}

template <typename mat_T>
void spmv_parallel(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread,
const MAT_TYPE mat_type,
const SYNC_TYPE sync_type)
{
switch (sync_type)
{
case SYNC_TYPE::mutex_var:
openmp_spmv_mutex_var<mat_T>(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
case SYNC_TYPE::mutex_arr:
openmp_spmv_mutex_arr<mat_T>(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
case SYNC_TYPE::atomic:
openmp_spmv_atomic<mat_T>(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
case SYNC_TYPE::none:
openmp_spmv_none<mat_T>(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
}
}

template <typename mat_T>
void benchmark_spmv_parallel(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread,
const MAT_TYPE mat_type,
const SYNC_TYPE sync_type)
{
uint32_t num_iter = 10;
start = std::chrono::steady_clock::now();
for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
spmv_parallel<mat_T>(alpha, mat_A, vec_x, beta, vec_b, num_thread, mat_type, sync_type);
}
end = std::chrono::steady_clock::now();
elapse = end - start;

std::string oper_type;
switch (mat_type)
{
case MAT_TYPE::coo:
if (sync_type == SYNC_TYPE::mutex_var)
{
oper_type = "COO (P-Mut_Var)  : ";
}
else if (sync_type == SYNC_TYPE::mutex_arr)
{
oper_type = "COO (P-Mut_Arr)  : ";
}
else if (sync_type == SYNC_TYPE::atomic)
{
oper_type = "COO (P-Atomic)   : ";
}
else
{
}
break;
case MAT_TYPE::csc:
if (sync_type == SYNC_TYPE::mutex_var)
{
oper_type = "CSC (P-Mut_Var)  : ";
}
else if (sync_type == SYNC_TYPE::mutex_arr)
{
oper_type = "CSC (P-Mut_Arr)  : ";
}
else if (sync_type == SYNC_TYPE::atomic)
{
oper_type = "CSC (P-Atomic)   : ";
}
else
{
}
break;
case MAT_TYPE::csr:
oper_type = "CSR (P)          : ";
break;
}
std::cout << oper_type << elapse.count() / num_iter << '\n';
}