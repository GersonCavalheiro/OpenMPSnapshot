#pragma once
#include "shared_struct.hpp"
#include "../include_for_cas766/work_balance.hpp"
#include <omp.h>
#include <chrono>

std::chrono::duration<double, std::milli> internal = std::chrono::milliseconds::zero();

void csr_spmv_parallel_static(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
int work_chunk = mat_A.mat_data.num_row / num_thread;
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for schedule(static, work_chunk)
for (uint32_t idx_r_ptr = 0; idx_r_ptr < mat_A.mat_data.num_row; idx_r_ptr++)
{
double temp_val = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_r_ptr]; idx_nz < mat_A.row_ptr[idx_r_ptr + 1]; idx_nz++)
{
temp_val += (mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_r_ptr] = alpha * temp_val + beta * vec_b[idx_r_ptr];
}
}
}

void csr_spmv_parallel_manual_balancing(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;

start = std::chrono::steady_clock::now();
std::vector<std::pair<int, std::vector<work_info>>> work_queue;
gen_work_queue(mat_A, num_thread, work_queue);
end = std::chrono::steady_clock::now();
internal += (end - start);

#pragma omp parallel num_threads(num_thread)
{
#pragma omp for schedule(static, work_queue.size() / num_thread)
for (size_t work_idx = 0; work_idx < work_queue.size(); ++work_idx)
{
for (const auto &assigned_works : work_queue[work_idx].second)
{
double temp_val = 0.0;
for (uint32_t nz_idx = assigned_works.nz_start; nz_idx < assigned_works.nz_end; nz_idx++)
{
temp_val += (mat_A.col_and_val[nz_idx].val * vec_x[mat_A.col_and_val[nz_idx].idx]);
}
vec_b[assigned_works.idx] = alpha * temp_val + beta * vec_b[assigned_works.idx];
}
}
}
}

void csr_spmv_parallel_default_dynamic(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
uint32_t work_chunk = static_cast<uint32_t>(mat_A.mat_data.num_row / num_thread * 0.10);
#pragma omp parallel num_threads(num_thread)
{
#pragma omp for schedule(guided, work_chunk)
for (uint32_t idx_r_ptr = 0; idx_r_ptr < mat_A.mat_data.num_row; idx_r_ptr++)
{
double temp_val = 0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_r_ptr]; idx_nz < mat_A.row_ptr[idx_r_ptr + 1]; idx_nz++)
{
temp_val += (mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_r_ptr] = alpha * temp_val + beta * vec_b[idx_r_ptr];
}
}
}

void csr_spmv_parallel_queue_dynamic(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;

start = std::chrono::steady_clock::now();
std::queue<work_info> work_queue;
gen_work_queue(mat_A, work_queue);
end = std::chrono::steady_clock::now();
internal += (end - start);

#pragma omp parallel num_threads(num_thread)
{
bool work_remained = true;
work_info retrieved_work{0, 0, 0};
while (work_remained == true)
{

#pragma omp critical
{
if (work_queue.empty() == true)
{
work_remained = false;
}
else
{
retrieved_work = work_queue.front();
work_queue.pop();
}
}

if (work_remained == true)
{
double temp_val = 0.0;
for (uint32_t nz_idx = retrieved_work.nz_start; nz_idx < retrieved_work.nz_end; nz_idx++)
{
temp_val += (alpha * mat_A.col_and_val[nz_idx].val * vec_x[mat_A.col_and_val[nz_idx].idx]);
}
vec_b[retrieved_work.idx] = alpha * temp_val + beta * vec_b[retrieved_work.idx];
}
}
}
}

void csr_spmv_parallel_vector_dynamic(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;

start = std::chrono::steady_clock::now();
std::vector<work_info> work_queue;
gen_work_queue(mat_A, work_queue);
end = std::chrono::steady_clock::now();
internal += (end - start);

uint32_t queue_len = static_cast<uint32_t>(work_queue.size());
uint32_t global_start_p = 0;
uint32_t work_chunk = static_cast<uint32_t>(queue_len / num_thread * 0.10);

#pragma omp parallel num_threads(num_thread)
{
uint32_t local_start = 0;
uint32_t local_end = 0;
bool work_remained = true;
while (work_remained == true)
{
#pragma omp critical
{
if (global_start_p == queue_len)
{
work_remained = false;
}
else
{
local_start = global_start_p;
local_end = local_start + work_chunk;
global_start_p += work_chunk;

if (local_end >= queue_len)
{
local_end = queue_len;
global_start_p = queue_len;
}
}
}

if (work_remained == true)
{
for (uint32_t work_idx = local_start; work_idx < local_end; ++work_idx)
{
double temp_val = 0.0;
for (uint32_t nz_idx = work_queue[work_idx].nz_start; nz_idx < work_queue[work_idx].nz_end; nz_idx++)
{
temp_val += mat_A.col_and_val[nz_idx].val * vec_x[mat_A.col_and_val[nz_idx].idx];
}
vec_b[work_queue[work_idx].idx] = alpha * temp_val + beta * vec_b[work_queue[work_idx].idx];
}
}
}
}
}

void benchmark_csr_parallel(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b,
const uint32_t num_thread)
{
std::vector<std::function<void(
const double,
const CSR &,
const std::vector<double> &,
const double,
std::vector<double> &,
const uint32_t)>>
func_ptr_list{
csr_spmv_parallel_static,
csr_spmv_parallel_manual_balancing,
csr_spmv_parallel_default_dynamic,
csr_spmv_parallel_queue_dynamic,
csr_spmv_parallel_vector_dynamic};

std::vector<std::string> oper_list{
"(def_Static) : ",
"(w___Balance): ",
"(def_Dynamic): ",
"(que_Dynamic): ",
"(vec_Dynamic): "};

std::chrono::steady_clock::time_point start;
std::chrono::steady_clock::time_point end;
std::chrono::duration<double, std::milli> elapse;

uint32_t num_iter = 10;
std::vector<double> old_vec;
for (uint32_t work_list = 0; work_list < oper_list.size(); ++work_list)
{
std::vector<double> veb_b_old = vec_b;
start = std::chrono::steady_clock::now();
for (uint32_t repeat = 0; repeat < num_iter; ++repeat)
{
(func_ptr_list[work_list])(alpha, mat_A, vec_x, beta, vec_b, num_thread);
}
end = std::chrono::steady_clock::now();

for (uint32_t idx_vec_b = 0; idx_vec_b < vec_b.size(); ++idx_vec_b)
{
double diff = veb_b_old[idx_vec_b] - vec_b[idx_vec_b];
if (diff > 0.0)
{
std::cout << diff << '\n';
}
}

elapse = end - start;
std::cout << "CSR " << oper_list[work_list] << elapse.count() / num_iter << '\n';

if (work_list == 1 || work_list == 3 || work_list == 4)
{
std::cout << "\tInternal : " << internal.count() / num_iter << '\n';
std::cout << "\tActual   : " << (elapse.count() - internal.count()) / num_iter << '\n';
internal = std::chrono::milliseconds::zero();
}
}
}