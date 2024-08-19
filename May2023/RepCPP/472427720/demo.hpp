#pragma once
#include "shared_struct.hpp"
#include <omp.h>
#include <chrono>

void openmp_spmv_CSR_static(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
auto thread_ID = omp_get_thread_num();

#pragma omp for
for (uint32_t idx_r_ptr = 0; idx_r_ptr < mat_A.mat_data.num_row; idx_r_ptr++)
{
#pragma omp critical
{
std::cout << "Thread " << thread_ID << " processing row - " << idx_r_ptr << "\n";
}

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

void openmp_spmv_CSR_dynamic(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
auto thread_ID = omp_get_thread_num();

#pragma omp for schedule(dynamic, 1)
for (uint32_t idx_r_ptr = 0; idx_r_ptr < mat_A.mat_data.num_row; idx_r_ptr++)
{
#pragma omp critical
{
std::cout << "Thread " << thread_ID << " processing row - " << idx_r_ptr << "\n";
}

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