#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class MATRIX_ALLOCATION
{
public:
MATRIX_ALLOCATION()
{

}

virtual ~MATRIX_ALLOCATION()
{

}

void alloc_vec(const uint32_t vector_len, std::string name, D_VECTOR &output)
{
output.vec_data.vec_name = name;
output.vec_data.len_vec = vector_len;
output.vec_data.num_nz = 0;
output.vec_element = std::vector<double>(vector_len, 0);
}


void alloc_vec(D_VECTOR &d_vec, std::string name, S_VECTOR &s_vec)
{
s_vec.vec_data.vec_name = name;
s_vec.vec_data.len_vec = d_vec.vec_data.len_vec;
s_vec.vec_data.num_nz = 0;
s_vec.vec_element.reserve(d_vec.vec_data.len_vec);
}


void alloc_vec(const uint32_t vec_len, uint32_t num_nz, std::string name, S_VECTOR &s_vec)
{
s_vec.vec_data.vec_name = name;
s_vec.vec_data.len_vec = vec_len;
s_vec.vec_data.num_nz = num_nz;
s_vec.vec_element.reserve(num_nz);
}

void alloc_vec(const VECTOR_INFO &vec_info, S_VECTOR &output)
{
output.vec_data = vec_info;
output.vec_element.reserve(vec_info.num_nz);
}

void alloc_vec(const VECTOR_INFO &vec_info, D_VECTOR &output)
{
output.vec_data = vec_info;
std::vector<double> temp_vec(vec_info.len_vec, 0);
output.vec_element = temp_vec;
}

void alloc_mat(uint32_t num_row, uint32_t num_col, std::string name, D_MATRIX &output)
{
output.mat_data.matrix_name = name;
output.mat_data.num_row = num_row;
output.mat_data.num_col = num_col;
output.mat_data.num_nz = 0;

for (uint32_t row = 0; row < num_row; row++)
{
std::vector<double> temp_vec(num_col, 0);
output.matrix.push_back(temp_vec);
}
}

void alloc_mat(const MATRIX_INFO &mat_info, D_MATRIX &output)
{
output.mat_data = mat_info;

for (uint32_t row = 0; row < mat_info.num_row; row++)
{
std::vector<double> temp_vec(mat_info.num_col, 0);
output.matrix.push_back(temp_vec);
}
}

void alloc_mat(const MATRIX_INFO &mat_info, COO &output)
{
output.mat_data = mat_info;
output.mat_elements.reserve(mat_info.num_nz);
}

void alloc_mat(const MATRIX_INFO &mat_info, CSR &output)
{
output.mat_data = mat_info;
output.row_ptr.reserve(mat_info.num_row + 1);
output.col_and_val.reserve(mat_info.num_nz);
}

void alloc_mat(const MATRIX_INFO &mat_info, CSC &output)
{
output.mat_data = mat_info;
output.col_ptr.reserve(mat_info.num_col + 1);
output.row_and_val.reserve(mat_info.num_nz);
}

void delete_vec(D_VECTOR &d_vec)
{
d_vec.vec_data.vec_name = " ";
d_vec.vec_data.len_vec = 0;
d_vec.vec_data.num_nz = 0;

d_vec.vec_element.clear();
d_vec.vec_element.shrink_to_fit();
}

void delete_vec(S_VECTOR &s_vec)
{
s_vec.vec_element.clear();
s_vec.vec_element.shrink_to_fit();
}

void delete_mat(D_MATRIX &d_mat)
{
d_mat.mat_data.matrix_name = " ";
d_mat.mat_data.num_row = 0;
d_mat.mat_data.num_col = 0;
d_mat.mat_data.num_nz = 0;

for (auto &iter_row : d_mat.matrix)
{
iter_row.clear();
iter_row.shrink_to_fit();
}
d_mat.matrix.clear();
d_mat.matrix.shrink_to_fit();
}

void delete_mat(COO &s_mat)
{
s_mat.mat_data.matrix_name = " ";
s_mat.mat_data.num_row = 0;
s_mat.mat_data.num_col = 0;
s_mat.mat_data.num_nz = 0;

s_mat.mat_elements.clear();
s_mat.mat_elements.shrink_to_fit();
}

void delete_mat(CSR &s_mat)
{
s_mat.mat_data.matrix_name = " ";
s_mat.mat_data.num_row = 0;
s_mat.mat_data.num_col = 0;
s_mat.mat_data.num_nz = 0;

s_mat.row_ptr.clear();
s_mat.row_ptr.shrink_to_fit();
s_mat.col_and_val.clear();
s_mat.col_and_val.shrink_to_fit();
}

void delete_mat(CSC &s_mat)
{
s_mat.mat_data.matrix_name = " ";
s_mat.mat_data.num_row = 0;
s_mat.mat_data.num_col = 0;
s_mat.mat_data.num_nz = 0;

s_mat.col_ptr.clear();
s_mat.col_ptr.shrink_to_fit();
s_mat.row_and_val.clear();
s_mat.row_and_val.shrink_to_fit();
}
};