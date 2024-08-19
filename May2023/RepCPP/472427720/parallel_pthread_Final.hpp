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

void *pthread_csc_spmv_mutex_var(void *csc_arg)
{
pthread_spmv_arg<CSC, std::vector<double>> *t_param = (pthread_spmv_arg<CSC, std::vector<double>> *)csc_arg;
for (uint32_t idx_ptr = t_param->idx_start; idx_ptr < t_param->idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = t_param->mat_A.col_ptr[idx_ptr]; idx_nz < t_param->mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
double temp_Val = (t_param->alpha *
t_param->mat_A.row_and_val[idx_nz].val *
t_param->vec_x[idx_ptr]);

pthread_mutex_lock(&mutex_var);
t_param->vec_b[t_param->mat_A.row_and_val[idx_nz].idx] += temp_Val;
pthread_mutex_unlock(&mutex_var);
}
}
return nullptr;
}

void *pthread_csc_spmv_mutex_arr(void *csc_arg)
{
pthread_spmv_arg<CSC, std::vector<double>> *t_param = (pthread_spmv_arg<CSC, std::vector<double>> *)csc_arg;
for (uint32_t idx_ptr = t_param->idx_start; idx_ptr < t_param->idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = t_param->mat_A.col_ptr[idx_ptr]; idx_nz < t_param->mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
double temp_Val = (t_param->alpha *
t_param->mat_A.row_and_val[idx_nz].val *
t_param->vec_x[idx_ptr]);
pthread_mutex_lock(&mutex_arr[t_param->mat_A.row_and_val[idx_nz].idx]);
t_param->vec_b[t_param->mat_A.row_and_val[idx_nz].idx] += temp_Val;
pthread_mutex_unlock(&mutex_arr[t_param->mat_A.row_and_val[idx_nz].idx]);
}
}
return nullptr;
}

void *pthread_csc_spmv_atomic(void *csc_arg)
{
pthread_spmv_arg<CSC, std::vector<std::atomic<double>>> *t_param = (pthread_spmv_arg<CSC, std::vector<std::atomic<double>>> *)csc_arg;
for (uint32_t idx_ptr = t_param->idx_start; idx_ptr < t_param->idx_end; ++idx_ptr)
{
for (uint32_t idx_nz = t_param->mat_A.col_ptr[idx_ptr]; idx_nz < t_param->mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
double temp_Val = (t_param->alpha *
t_param->mat_A.row_and_val[idx_nz].val *
t_param->vec_x[idx_ptr]);

t_param->vec_b[t_param->mat_A.row_and_val[idx_nz].idx] += temp_Val;
}
}
return nullptr;
}

void *pthread_csr_spmv(void *csr_arg)
{
pthread_spmv_arg<CSR, std::vector<double>> *t_param = (pthread_spmv_arg<CSR, std::vector<double>> *)csr_arg;
for (uint32_t idx_ptr = t_param->idx_start; idx_ptr < t_param->idx_end; ++idx_ptr)
{
double temp_val = 0.0;
for (uint32_t idx_nz = t_param->mat_A.row_ptr[idx_ptr]; idx_nz < t_param->mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
temp_val += (t_param->mat_A.col_and_val[idx_nz].val *
t_param->vec_x[t_param->mat_A.col_and_val[idx_nz].idx]);
}
t_param->vec_b[idx_ptr] = t_param->alpha * temp_val + t_param->beta * t_param->vec_b[idx_ptr];
}
return nullptr;
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
const VEC_TYPE vec_type,
const SYNC_TYPE sync_type)
{
std::vector<pthread_spmv_arg<mat_T, std::vector<double>>> thread_args_general;
std::vector<pthread_spmv_arg<mat_T, std::vector<std::atomic<double>>>> thread_args_atomic;
std::vector<std::atomic<double>> atomic_vec_b;

switch (sync_type)
{
case SYNC_TYPE::mutex_var:
pthread_mutex_init(&mutex_var, NULL);
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = beta * vec_b[idx_vec_b];
}
break;
case SYNC_TYPE::mutex_arr:
mutex_arr = new pthread_mutex_t[vec_x.size()];
for (uint32_t idx_vec = 0; idx_vec < vec_b.size(); idx_vec++)
{
pthread_mutex_init(&mutex_arr[idx_vec], NULL);
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
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
uint32_t idx_start = idx_thread * work_range;
uint32_t idx_end = idx_thread == (num_thread - 1) ? num_nz : idx_start + work_range;
if (sync_type == SYNC_TYPE::atomic)
{
thread_args_atomic.push_back(
pthread_spmv_arg<mat_T, std::vector<std::atomic<double>>>(
idx_start, idx_end, alpha, beta, mat_A, vec_x, atomic_vec_b));
}
else
{
thread_args_general.push_back(pthread_spmv_arg<mat_T, std::vector<double>>(
idx_start, idx_end, alpha, beta, mat_A, vec_x, vec_b));
}
}

std::vector<pthread_t> thread_list(num_thread);
for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
switch (mat_type)
{
case MAT_TYPE::coo:
if (sync_type == SYNC_TYPE::mutex_var)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_mutex_var, (void *)&thread_args_general[idx_thread]);
}
else if (sync_type == SYNC_TYPE::mutex_arr)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_mutex_arr, (void *)&thread_args_general[idx_thread]);
}
else if (sync_type == SYNC_TYPE::atomic)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_coo_spmv_atomic, (void *)&thread_args_atomic[idx_thread]);
}
else
{
}
break;

case MAT_TYPE::csc:
if (sync_type == SYNC_TYPE::mutex_var)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_csc_spmv_mutex_var, (void *)&thread_args_general[idx_thread]);
}
else if (sync_type == SYNC_TYPE::mutex_arr)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_csc_spmv_mutex_arr, (void *)&thread_args_general[idx_thread]);
}
else if (sync_type == SYNC_TYPE::atomic)
{
pthread_create(&thread_list[idx_thread], NULL, &pthread_csc_spmv_atomic, (void *)&thread_args_atomic[idx_thread]);
}
else
{
}
break;

case MAT_TYPE::csr:
pthread_create(&thread_list[idx_thread], NULL, &pthread_csr_spmv, (void *)&thread_args_general[idx_thread]);
break;
}
}

for (uint32_t idx_thread = 0; idx_thread < num_thread; ++idx_thread)
{
pthread_join(thread_list[idx_thread], NULL);
}

switch (sync_type)
{
case SYNC_TYPE::mutex_var:
pthread_mutex_destroy(&mutex_var);
break;
case SYNC_TYPE::mutex_arr:
for (uint32_t idx_mutex = 0; idx_mutex < vec_x.size(); ++idx_mutex)
{
pthread_mutex_destroy(&mutex_arr[idx_mutex]);
}
delete mutex_arr;
break;
case SYNC_TYPE::atomic:
for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); idx_vec_b++)
{
vec_b[idx_vec_b] = atomic_vec_b[idx_vec_b];
}
break;
case SYNC_TYPE::none:
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
const VEC_TYPE vec_type,
const SYNC_TYPE sync_type)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

uint32_t num_iter = 10;
start = std::chrono::steady_clock::now();
for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
spmv_parallel(alpha, mat_A, vec_x, beta, vec_b, num_thread, mat_type, vec_type, sync_type);
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