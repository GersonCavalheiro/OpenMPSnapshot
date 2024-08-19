#pragma once
#include "shared_struct.hpp"

void serial_spmv(
const double alpha,
const CSR &mat_A,
const std::vector<double> &vec_x,
const double beta,
std::vector<double> &vec_b)
{
for (uint32_t idx_ptr = 0; idx_ptr < mat_A.mat_data.num_row; ++idx_ptr)
{
double ith_Ax = 0.0;
for (uint32_t idx_nz = mat_A.row_ptr[idx_ptr]; idx_nz < mat_A.row_ptr[idx_ptr + 1]; ++idx_nz)
{
ith_Ax += (mat_A.col_and_val[idx_nz].val * vec_x[mat_A.col_and_val[idx_nz].idx]);
}
vec_b[idx_ptr] = alpha * ith_Ax + beta * vec_b[idx_ptr];
}
}