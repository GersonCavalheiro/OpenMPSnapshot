#pragma once
#include "../include_for_cas766/PCH.hpp"
#include "shared_struct.hpp"
#include <pthread.h>
#include <semaphore.h>
#include <atomic>
#include <chrono>

pthread_mutex_t mutex_var;
pthread_mutex_t *mutex_arr;

template <typename mat_T, typename vec_T>
struct pthread_spmv_arg
{
uint32_t idx_start;
uint32_t idx_end;
double alpha;
double beta;
const mat_T &mat_A;
const std::vector<double> &vec_x;
vec_T &vec_b;

pthread_spmv_arg(
const uint32_t _idx_start,
const uint32_t _idx_end,
const double _alpha,
const double _beta,
const mat_T &_mat_A,
const std::vector<double> &_vec_x,
vec_T &_vec_b) : idx_start(_idx_start),
idx_end(_idx_end),
alpha(_alpha),
beta(_beta),
mat_A(_mat_A),
vec_x(_vec_x),
vec_b(_vec_b)
{
}
};

void *pthread_coo_spmv_mutex_var(void *coo_arg)
{
pthread_spmv_arg<COO, std::vector<double>> *t_param = (pthread_spmv_arg<COO, std::vector<double>> *)coo_arg;
for (uint32_t idx_nz = t_param->idx_start; idx_nz < t_param->idx_end; ++idx_nz)
{
double temp_val =
t_param->alpha *
t_param->mat_A.mat_elements[idx_nz].val *
t_param->vec_x[t_param->mat_A.mat_elements[idx_nz].col_idx];

pthread_mutex_lock(&mutex_var);
t_param->vec_b[t_param->mat_A.mat_elements[idx_nz].row_idx] += temp_val;
pthread_mutex_unlock(&mutex_var);
}
return nullptr;
}

void spmv_parallel_single_mutex(
const double alpha,
const cookie_read_function_t &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); ++idx_vec_b)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
uint32_t work_range = num_nz / num_thread;
std::vector<pthread_spmv_arg<mat_T, std::vector<double>>> thread_args_general;
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
uint32_t idx_start = idx_thread * work_range;
uint32_t idx_end = idx_thread == (num_thread - 1) ? num_nz : idx_start + work_range;
thread_args_general.push_back(pthread_spmv_arg<mat_T, std::vector<double>>(
idx_start, idx_end, alpha, beta, mat_A, vec_x, vec_b));
}

pthread_mutex_init(&mutex_var, NULL);

std::vector<pthread_t> thread_list(num_thread);
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_mutex_var, (void *)&thread_args_general[idx_thread]);
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_join(thread_list[idx_thread], NULL);
}

pthread_mutex_destroy(&mutex_var);
}

void *pthread_coo_spmv_mutex_arr(void *coo_arg)
{
pthread_spmv_arg<COO, std::vector<double>> *t_param = (pthread_spmv_arg<COO, std::vector<double>> *)coo_arg;
for (uint32_t idx_nz = t_param->idx_start; idx_nz < t_param->idx_end; ++idx_nz)
{
double temp_val =
t_param->alpha *
t_param->mat_A.mat_elements[idx_nz].val *
t_param->vec_x[t_param->mat_A.mat_elements[idx_nz].col_idx];

pthread_mutex_lock(&mutex_arr[t_param->mat_A.mat_elements[idx_nz].row_idx]);
t_param->vec_b[t_param->mat_A.mat_elements[idx_nz].row_idx] += temp_val;
pthread_mutex_unlock(&mutex_arr[t_param->mat_A.mat_elements[idx_nz].row_idx]);
}
return nullptr;
}

void spmv_parallel_array_mutex(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<pthread_spmv_arg<mat_T, std::vector<double>>> thread_args_general;

mutex_arr = new pthread_mutex_t[vec_x.size()];
for (uint32_t idx_vec = 0; idx_vec < vec_b.size(); idx_vec++)
{
pthread_mutex_init(&mutex_arr[idx_vec], NULL);
vec_b[idx_vec] = beta * vec_b[idx_vec];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
uint32_t work_range = num_nz / num_thread;
std::vector<pthread_spmv_arg<mat_T, std::vector<double>>> thread_args_general;
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
uint32_t idx_start = idx_thread * work_range;
uint32_t idx_end = idx_thread == (num_thread - 1) ? num_nz : idx_start + work_range;
thread_args_general.push_back(pthread_spmv_arg<mat_T, std::vector<double>>(
idx_start, idx_end, alpha, beta, mat_A, vec_x, vec_b));
}

std::vector<pthread_t> thread_list(num_thread);
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_mutex_arr, (void *)&thread_args_general[idx_thread]);
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_join(thread_list[idx_thread], NULL);
}

for (uint32_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
pthread_mutex_destroy(&mutex_arr[idx_mutex]);
}
delete mutex_arr;
}

void *pthread_coo_spmv_atomic(void *coo_arg)
{
pthread_spmv_arg<COO, std::vector<std::atomic<double>>> *t_param = (pthread_spmv_arg<COO, std::vector<std::atomic<double>>> *)coo_arg;
for (uint32_t idx_nz = t_param->idx_start; idx_nz < t_param->idx_end; ++idx_nz)
{
double temp_val =
t_param->alpha *
t_param->mat_A.mat_elements[idx_nz].val *
t_param->vec_x[t_param->mat_A.mat_elements[idx_nz].col_idx];

t_param->vec_b[t_param->mat_A.mat_elements[idx_nz].row_idx] += temp_val;
}
return nullptr;
}

void spmv_parallel_atomic(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<pthread_spmv_arg<mat_T, std::vector<std::atomic<double>>>> thread_args_atomic;
std::vector<std::atomic<double>> atomic_vec_b(vec_b.size());
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
atomic_vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}

uint32_t num_nz = mat_A.mat_data.num_nz;
uint32_t work_range = num_nz / num_thread;
std::vector<pthread_spmv_arg<mat_T, std::vector<double>>> thread_args_general;
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
uint32_t idx_start = idx_thread * work_range;
uint32_t idx_end = idx_thread == (num_thread - 1) ? num_nz : idx_start + work_range;
thread_args_atomic.push_back(
pthread_spmv_arg<mat_T, std::vector<std::atomic<double>>>(
idx_start, idx_end, alpha, beta, mat_A, vec_x, atomic_vec_b));
}

std::vector<pthread_t> thread_list(num_thread);
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_atomic, (void *)&thread_args_atomic[idx_thread]);
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_join(thread_list[idx_thread], NULL);
}

for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = atomic_vec_b[idx_vec_b];
}
}