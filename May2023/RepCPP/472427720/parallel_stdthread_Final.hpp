#pragma once
#include "../include_for_cas766/PCH.hpp"
#include "shared_struct.hpp"
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

std::mutex mutex_var;
std::vector<std::mutex> mutex_arr;

template <typename mat_T>
void stdthread_spmv_mut_var(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
}

template <>
void stdthread_spmv_mut_var<COO>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_nz = idx_start; idx_nz < idx_end; ++idx_nz)
{
std::lock_guard<std::mutex> lockGuard(mutex_var);
vec_b[mat_A.mat_elements[idx_nz].row_idx] += (alpha *
mat_A.mat_elements[idx_nz].val *
vec_x[mat_A.mat_elements[idx_nz].col_idx]);
}
}

template <>
void stdthread_spmv_mut_var<CSC>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = idx_start; idx_ptr < idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
std::lock_guard<std::mutex> lockGuard(mutex_var);
vec_b[mat_A.row_and_val[idx_nz].idx] += (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
}
}
}

template <typename mat_T>
void stdthread_spmv_mut_arr(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
}

template <>
void stdthread_spmv_mut_arr<COO>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_nz = idx_start; idx_nz < idx_end; ++idx_nz)
{
std::lock_guard<std::mutex> lockGuard(mutex_arr[mat_A.mat_elements[idx_nz].row_idx]);
vec_b[mat_A.mat_elements[idx_nz].row_idx] += (alpha *
mat_A.mat_elements[idx_nz].val *
vec_x[mat_A.mat_elements[idx_nz].col_idx]);
}
}

template <>
void stdthread_spmv_mut_arr<CSC>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = idx_start; idx_ptr < idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
std::lock_guard<std::mutex> lockGuard(mutex_arr[mat_A.row_and_val[idx_nz].idx]);
vec_b[mat_A.row_and_val[idx_nz].idx] += (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
}
}
}

template <typename mat_T>
void stdthread_spmv_atomic(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<std::atomic<double>> &vec_b)
{
}

template <>
void stdthread_spmv_atomic<COO>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<std::atomic<double>> &vec_b)
{
for (uint32_t idx_nz = idx_start; idx_nz < idx_end; ++idx_nz)
{
vec_b[mat_A.mat_elements[idx_nz].row_idx] += (alpha *
mat_A.mat_elements[idx_nz].val *
vec_x[mat_A.mat_elements[idx_nz].col_idx]);
}
}

template <>
void stdthread_spmv_atomic<CSC>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<std::atomic<double>> &vec_b)
{
for (uint32_t idx_ptr = idx_start; idx_ptr < idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
vec_b[mat_A.row_and_val[idx_nz].idx] += (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
}
}
}

template <typename mat_T>
void stdthread_spmv_none(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const mat_T &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
}

template <>
void stdthread_spmv_none<CSR>(
const uint32_t idx_start,
const uint32_t idx_end,
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = idx_start; idx_ptr < idx_end; ++idx_ptr)
{
double temp_val = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_ptr]; idx_nz < mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
temp_val += (alpha * mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_ptr] = alpha * temp_val + beta * vec_b[idx_ptr];
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
std::vector<std::atomic<double>> atomic_vec_b;
switch (sync_type)
{
case SYNC_TYPE::mutex_var:
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}
break;
case SYNC_TYPE::mutex_arr:
mutex_arr = std::vector<std::mutex>(vec_x.size());
for (uint32_t idx_vec = 0; idx_vec < vec_b.size(); idx_vec++)
{
vec_b[idx_vec] = beta * vec_b[idx_vec];
}
break;
case SYNC_TYPE::atomic:
atomic_vec_b = std::vector<std::atomic<double>>(vec_b.size());
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
atomic_vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}
break;
case SYNC_TYPE::none:
break;
}

uint32_t num_nz = 0;
switch (mat_type)
{
case MAT_TYPE::coo:
num_nz = mat_A.mat_data.num_nz;
break;
case MAT_TYPE::csc:
num_nz = mat_A.mat_data.num_col;
break;
case MAT_TYPE::csr:
num_nz = mat_A.mat_data.num_row;
break;
}
uint32_t work_range = num_nz / num_thread;

std::vector<uint32_t> idx_start_list(num_thread);
std::vector<uint32_t> idx_end_list(num_thread);
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
idx_start_list[idx_thread] = idx_thread * work_range;
idx_end_list[idx_thread] = idx_thread == (num_thread - 1) ? num_nz : idx_thread * work_range + work_range;
}

std::vector<std::thread> thread_list;
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
switch (sync_type)
{
case SYNC_TYPE::mutex_var:
thread_list.push_back(std::move(std::thread(stdthread_spmv_mut_var<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(vec_b))));
break;
case SYNC_TYPE::mutex_arr:
thread_list.push_back(std::move(std::thread(stdthread_spmv_mut_arr<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(vec_b))));
break;
case SYNC_TYPE::atomic:
thread_list.push_back(std::move(std::thread(stdthread_spmv_atomic<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(atomic_vec_b))));
break;
case SYNC_TYPE::none:
thread_list.push_back(std::move(std::thread(stdthread_spmv_none<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(vec_b))));
break;
}
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
thread_list[idx_thread].join();
}

if (sync_type == SYNC_TYPE::atomic)
{
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = atomic_vec_b[idx_vec_b];
}
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
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

uint32_t num_iter = 10;
start = std::chrono::steady_clock::now();
for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
spmv_parallel(alpha, mat_A, vec_x, beta, vec_b, num_thread, mat_type, sync_type);
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