#pragma once
#include "shared_struct.hpp"

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
vec_b[mat_A.row_and_val[idx_nz].idx] +=
(alpha * mat_A.row_and_val[idx_nz].val * vec_x[idx_ptr]);
}
}
}
