#pragma once
#include "shared_struct.hpp"
#include <omp.h>
#include <chrono>

void openmp_spmv_mutex_var(
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

void openmp_spmv_mutex_arr(
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

void openmp_spmv_atomic(
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

void spmv_parallel(
const double alpha,
const COO &mat_A,
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
openmp_spmv_mutex_var(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
case SYNC_TYPE::mutex_arr:
openmp_spmv_mutex_arr(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
case SYNC_TYPE::atomic:
openmp_spmv_atomic(alpha, mat_A, vec_x, beta, vec_b, num_thread);
break;
}
}