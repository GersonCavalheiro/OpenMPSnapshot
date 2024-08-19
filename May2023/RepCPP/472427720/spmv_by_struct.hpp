#pragma once
#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/matrix_struct.hpp"
#include "shared_struct.hpp"
#include <omp.h>

struct arr_COO
{
MATRIX_INFO mat_data;
std::vector<uint32_t> row_idx;
std::vector<uint32_t> col_idx;
std::vector<double> val;
};

struct arr_CSC
{
MATRIX_INFO mat_data;
std::vector<uint32_t> col_ptr;
std::vector<uint32_t> row_idx;
std::vector<double> val;
};

struct arr_CSR
{
MATRIX_INFO mat_data;
std::vector<uint32_t> row_ptr;
std::vector<uint32_t> col_idx;
std::vector<double> val;
};

void struct_convert(const COO &struct_arr, arr_COO &arr_struct)
{
arr_struct.mat_data = struct_arr.mat_data;

arr_struct.row_idx = std::vector<uint32_t>(struct_arr.mat_data.num_nz);
arr_struct.col_idx = std::vector<uint32_t>(struct_arr.mat_data.num_nz);
arr_struct.val = std::vector<double>(struct_arr.mat_data.num_nz);
for (uint32_t idx_nz = 0; idx_nz < struct_arr.mat_data.num_nz; ++idx_nz)
{
arr_struct.row_idx[idx_nz] = struct_arr.mat_elements[idx_nz].row_idx;
arr_struct.col_idx[idx_nz] = struct_arr.mat_elements[idx_nz].col_idx;
arr_struct.val[idx_nz] = struct_arr.mat_elements[idx_nz].val;
}
}

void struct_convert(const CSC &struct_arr, arr_CSC &arr_struct)
{
arr_struct.mat_data = struct_arr.mat_data;

arr_struct.col_ptr = std::vector<uint32_t>(struct_arr.mat_data.num_col + 1);
for (uint32_t idx_c_ptr = 0; idx_c_ptr < struct_arr.col_ptr.size(); ++idx_c_ptr)
{
arr_struct.col_ptr[idx_c_ptr] = struct_arr.col_ptr[idx_c_ptr];
}

arr_struct.row_idx = std::vector<uint32_t>(struct_arr.mat_data.num_nz);
arr_struct.val = std::vector<double>(struct_arr.mat_data.num_nz);
for (uint32_t idx_nz = 0; idx_nz < struct_arr.mat_data.num_nz; ++idx_nz)
{
arr_struct.row_idx[idx_nz] = struct_arr.row_and_val[idx_nz].idx;
arr_struct.val[idx_nz] = struct_arr.row_and_val[idx_nz].val;
}
}

void struct_convert(const CSR &struct_arr, arr_CSR &arr_struct)
{
arr_struct.mat_data = struct_arr.mat_data;

arr_struct.row_ptr = std::vector<uint32_t>(struct_arr.mat_data.num_row + 1);
for (uint32_t idx_r_ptr = 0; idx_r_ptr < struct_arr.row_ptr.size(); ++idx_r_ptr)
{
arr_struct.row_ptr[idx_r_ptr] = struct_arr.row_ptr[idx_r_ptr];
}

arr_struct.col_idx = std::vector<uint32_t>(struct_arr.mat_data.num_nz);
arr_struct.val = std::vector<double>(struct_arr.mat_data.num_nz);
for (uint32_t idx_nz = 0; idx_nz < struct_arr.mat_data.num_nz; ++idx_nz)
{
arr_struct.col_idx[idx_nz] = struct_arr.col_and_val[idx_nz].idx;
arr_struct.val[idx_nz] = struct_arr.col_and_val[idx_nz].val;
}
}

void serial_spmv_arr(
const double alpha,
const arr_COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_nz = 0; idx_nz < mat_A.mat_data.num_nz; ++idx_nz)
{
vec_b[mat_A.row_idx[idx_nz]] += (alpha * mat_A.val[idx_nz] * vec_x[mat_A.col_idx[idx_nz]]);
}
}

void serial_spmv_arr(
const double alpha,
const arr_CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
vec_b[mat_A.row_idx[idx_nz]] += (alpha * mat_A.val[idx_nz] * vec_x[idx_ptr]);
}
}
}

void serial_spmv_arr(
const double alpha,
const arr_CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_row; ++idx_ptr)
{
double temp_val = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_ptr]; idx_nz < mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
temp_val += (alpha * mat_A.val[idx_nz] * vec_x[mat_A.col_idx[idx_nz]]);
}
vec_b[idx_ptr] = temp_val;
}
}

template <typename mat_T>
void benchmark_serial_spmv_arr(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const MAT_TYPE mat_type)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

uint32_t num_iter = 10;
start = std::chrono::steady_clock::now();
for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); ++idx_vec_b)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}
serial_spmv_arr(alpha, mat_A, vec_x, beta, vec_b);
}
end = std::chrono::steady_clock::now();
elapse = end - start;

std::string oper_type;
switch (mat_type)
{
case MAT_TYPE::coo:
oper_type = "COO (arr)        : ";
break;
case MAT_TYPE::csc:
oper_type = "CSC (arr)        : ";
break;
case MAT_TYPE::csr:
oper_type = "CSR (arr)        : ";
break;
}

std::cout << oper_type << elapse.count() / num_iter << '\n';
}

void parallel_spmv_arr(
const double alpha,
const arr_COO &mat_A,
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
for (uint32_t idx_nz = 0; idx_nz < mat_A.mat_data.num_nz; ++idx_nz)
{
double temp_Val = (alpha * mat_A.val[idx_nz] * vec_x[mat_A.col_idx[idx_nz]]);
omp_set_lock(&(mutex_arr[mat_A.row_idx[idx_nz]]));
vec_b[mat_A.row_idx[idx_nz]] += temp_Val;
omp_unset_lock(&(mutex_arr[mat_A.row_idx[idx_nz]]));
}
}

for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_destroy_lock(&(mutex_arr[idx_mutex]));
}
}

void parallel_spmv_arr(
const double alpha,
const arr_CSC &mat_A,
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
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
double temp_val = (alpha * mat_A.val[idx_nz] * vec_x[idx_ptr]);
omp_set_lock(&(mutex_arr[mat_A.row_idx[idx_nz]]));
vec_b[mat_A.row_idx[idx_nz]] += temp_val;
omp_unset_lock(&(mutex_arr[mat_A.row_idx[idx_nz]]));
}
}
}

for (size_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
omp_destroy_lock(&(mutex_arr[idx_mutex]));
}
}

void parallel_spmv_arr(
const double alpha,
const arr_CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_row; ++idx_ptr)
{
double temp_val = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_ptr]; idx_nz < mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
temp_val += (alpha * mat_A.val[idx_nz] * vec_x[mat_A.col_idx[idx_nz]]);
}
vec_b[idx_ptr] = temp_val;
}
}
}

template <typename mat_T>
void benchmark_spmv_parallel_arr(
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread,
const MAT_TYPE mat_type)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

uint32_t num_iter = 10;
start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); ++idx_vec_b)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}
}

for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
parallel_spmv_arr(alpha, mat_A, vec_x, beta, vec_b, num_thread);
}

end = std::chrono::steady_clock::now();
elapse = end - start;

std::string oper_type;
switch (mat_type)
{
case MAT_TYPE::coo:
oper_type = "COO (P-arr)      : ";
break;
case MAT_TYPE::csc:
oper_type = "CSC (P-arr)      : ";
break;
case MAT_TYPE::csr:
oper_type = "CSR (P-arr)      : ";
break;
}
std::cout << oper_type << elapse.count() / num_iter << '\n';
}