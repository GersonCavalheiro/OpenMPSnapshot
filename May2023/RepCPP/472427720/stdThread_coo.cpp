#pragma once
#include "../include_for_cas766/PCH.hpp"
#include "shared_struct.hpp"
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

std::mutex mutex_var;
std::vector<std::mutex> mutex_arr;

void stdthread_spmv_single_mutex(
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

void spmv_parallel_single_mutex(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread const uint32_t num_thread)
{
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
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
thread_list.push_back(std::move(std::thread(
stdthread_spmv_single_mutex,
idx_start_list[idx_thread],
idx_end_list[idx_thread],
alpha, std::ref(mat_A),
std::ref(vec_x),
beta,
std::ref(vec_b))));
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
thread_list[idx_thread].join();
}
}

void stdthread_spmv_mut_arr(
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

void spmv_parallel_array_mutex(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b const uint32_t num_thread)
{

mutex_arr = std::vector<std::mutex>(vec_x.size());
for (uint32_t idx_vec = 0; idx_vec < vec_b.size(); idx_vec++)
{
vec_b[idx_vec] = beta * vec_b[idx_vec];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
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
thread_list.push_back(std::move(std::thread(stdthread_spmv_mut_arr<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(vec_b))));
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
thread_list[idx_thread].join();
}
}

void stdthread_spmv_atomic(
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

void spmv_parallel_atomic(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<std::atomic<double>> atomic_vec_b(vec_b.size());
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
atomic_vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
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
thread_list.push_back(std::move(std::thread(stdthread_spmv_atomic<mat_T>, idx_start_list[idx_thread], idx_end_list[idx_thread], alpha, std::ref(mat_A), std::ref(vec_x), beta, std::ref(atomic_vec_b))));
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
thread_list[idx_thread].join();
}

for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = atomic_vec_b[idx_vec_b];
}
}