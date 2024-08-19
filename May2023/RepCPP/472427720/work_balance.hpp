#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

struct work_info
{
uint32_t idx;
uint32_t nz_start;
uint32_t nz_end;
};

void gen_work_queue(
const CSR &mat,
const int num_threads,
std::vector<std::pair<int, std::vector<work_info>>> &work_queue)
{
uint32_t row_idx = 0;
uint32_t nz_start = 0;
uint32_t nz_end = 0;
double work_average = mat.mat_data.num_col;
work_queue = std::vector<std::pair<int, std::vector<work_info>>>(num_threads);
for (uint32_t idx_ptr = 0; idx_ptr < mat.mat_data.num_row; ++idx_ptr)
{
row_idx = idx_ptr;
nz_start = mat.row_ptr[idx_ptr];
nz_end = mat.row_ptr[idx_ptr + 1];

if (nz_start != nz_end)
{
work_info temp{row_idx, nz_start, nz_end};
for (int idx_thread = 0; idx_thread < num_threads; ++idx_thread)
{
if (work_queue[idx_thread].first <= work_average)
{
work_queue[idx_thread].first += (nz_end - nz_start);
work_queue[idx_thread].second.push_back(temp);
break;
}
}

double work_sum = 0;
for (int idx_thread = 0; idx_thread < num_threads; ++idx_thread)
{
work_sum += work_queue[idx_thread].first;
}
work_average = work_sum / num_threads;
}
}
}

void gen_work_queue(
const CSR &mat,
std::queue<work_info> &work_queue)
{
work_info temp{0, 0, 0};
for (uint32_t idx_ptr = (uint32_t)0; idx_ptr < mat.mat_data.num_row; ++idx_ptr)
{
temp.idx = idx_ptr;
temp.nz_start = mat.row_ptr[idx_ptr];
temp.nz_end = mat.row_ptr[idx_ptr + 1];
if (temp.nz_start != temp.nz_end)
{
work_queue.push(temp);
}
}
}

void gen_work_queue(
const CSR &mat,
std::vector<work_info> &work_queue)
{
work_info temp{0, 0, 0};
work_queue.reserve(mat.mat_data.num_row);
for (uint32_t idx_ptr = (uint32_t)0; idx_ptr < mat.mat_data.num_row; ++idx_ptr)
{
temp.idx = idx_ptr;
temp.nz_start = mat.row_ptr[idx_ptr];
temp.nz_end = mat.row_ptr[idx_ptr + 1];
if (temp.nz_start != temp.nz_end)
{
work_queue.push_back(temp);
}
}
}




