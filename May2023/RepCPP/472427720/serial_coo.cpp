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
vec_b[mat_A.mat_elements[idx_nz].row_idx] += 
(alpha * mat_A.mat_elements[idx_nz].val * vec_x[mat_A.mat_elements[idx_nz].col_idx]);
}
}

