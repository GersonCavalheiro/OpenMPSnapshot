#pragma once
#include "shared_struct.hpp"

void serial_spmv(
const double alpha,
const COO &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_nz = 0; idx_nz < mat_A.mat_data.num_nz; ++idx_nz)
{
vec_b[mat_A.mat_elements[idx_nz].row_idx] += (alpha * mat_A.mat_elements[idx_nz].val * vec_x[mat_A.mat_elements[idx_nz].col_idx]);
}
}

void serial_spmv(
const double alpha,
const CSC &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_col; ++idx_ptr)
{
for (uint32_t idx_nz = mat_A.col_ptr[idx_ptr]; idx_nz < mat_A.col_ptr[idx_ptr + 1]; ++idx_nz)
{
vec_b[mat_A.row_and_val[idx_nz].idx] += (alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
}
}
}

void serial_spmv(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_row; ++idx_ptr)
{
double temp_val = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_ptr]; idx_nz < mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
temp_val += (alpha * mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_ptr] = temp_val + beta * vec_b[idx_ptr];
}
}

template <typename mat_T>
void benchmark_spmv_serial(
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
serial_spmv(alpha, mat_A, vec_x, beta, vec_b);
}
end = std::chrono::steady_clock::now();
elapse = end - start;

std::string oper_type;
switch (mat_type)
{
case MAT_TYPE::coo:
oper_type = "COO (S)          : ";
break;
case MAT_TYPE::csc:
oper_type = "CSC (S)          : ";
break;
case MAT_TYPE::csr:
oper_type = "CSR (S)          : ";
break;
}

std::cout << oper_type << elapse.count() / num_iter << '\n';
}