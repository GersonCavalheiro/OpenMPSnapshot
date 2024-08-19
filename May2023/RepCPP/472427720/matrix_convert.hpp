#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"
#include "matrix_allocation.hpp"

class MATRIX_CONVERT
{
public:
MATRIX_CONVERT()
{
{

}
}
virtual ~MATRIX_CONVERT()
{

}

bool convert(
const std::vector<uint32_t> &row_idx,
const std::vector<uint32_t> &col_idx,
const std::vector<double> &val,
const uint32_t num_row,
const uint32_t num_col,
COO &coo)
{
if ((row_idx.size() != col_idx.size()) || (row_idx.size() != val.size()))
{
std::string e_message = "Logic Error\n"
"Length mitmatch between the given vectors.\n"
"row idx - " + std::to_string(row_idx.size()) + "\n"
"col idx - " + std::to_string(col_idx.size()) + "\n"
"value - " + std::to_string(val.size()) + "\n";
throw std::logic_error(e_message);
}

uint32_t track_row = 0;
uint32_t track_col = 0;
coo.mat_elements = std::vector<MAT_ELE_DATA>(coo.mat_data.num_nz, MAT_ELE_DATA{0, 0, 0});
for (uint32_t ele_A = 0; ele_A < coo.mat_data.num_nz; ele_A++)
{
track_row = track_row < row_idx[ele_A] ? row_idx[ele_A] : track_row;
coo.mat_elements[ele_A].row_idx = row_idx[ele_A];

track_col = track_row < col_idx[ele_A] ? col_idx[ele_A] : track_row;
coo.mat_elements[ele_A].col_idx = col_idx[ele_A];

coo.mat_elements[ele_A].val = val[ele_A];
}

if (track_row > num_row - 1 || track_col > num_col - 1)
{
std::string e_message = "Logic Error\n"
"One of idx exceed matrix dimension.\n"
"Row Limit - " +
std::to_string(num_row - 1) + "\n"
"Row idx - " +
std::to_string(track_row) + "\n"
"Col Limit - " +
std::to_string(num_col - 1) + "\n"
"Col idx - " +
std::to_string(track_col) + "\n";
throw std::logic_error(e_message);
}

coo.mat_data.matrix_name = "NONE";
coo.mat_data.num_row = num_row;
coo.mat_data.num_col = num_col;
coo.mat_data.num_nz = (uint32_t)coo.mat_elements.size();

return true;
}

bool convert(
const std::vector<uint32_t> &row_idx,
const std::vector<uint32_t> &col_idx,
const std::vector<double> &val,
const uint32_t num_row,
const uint32_t num_col,
CSC &csc)
{
COO *temp = new COO();
this->convert(row_idx, col_idx, val, num_row, num_col, *temp);
this->convert(*temp, csc);
return true;
}

bool convert(
const std::vector<uint32_t> &row_idx,
const std::vector<uint32_t> &col_idx,
const std::vector<double> &val,
const uint32_t num_row,
const uint32_t num_col,
CSR &csr)
{
COO *temp = new COO();
this->convert(row_idx, col_idx, val, num_row, num_col, *temp);
this->convert(*temp, csr);
return true;
}

bool convert(COO &coo, CSR &csr)
{
sort_COO(coo, sort_target::CSR);

m_alloc.alloc_mat(coo.mat_data, csr);

IDX_VAL temp;
uint32_t ptr_tracker = 0;
csr.row_ptr.push_back(0);
for (uint32_t iter_nz = 0; iter_nz < coo.mat_data.num_nz; iter_nz++)
{
for (; ptr_tracker < coo.mat_elements[iter_nz].row_idx; ptr_tracker++)
{
csr.row_ptr.push_back(iter_nz);
}
temp.idx = coo.mat_elements[iter_nz].col_idx;
temp.val = coo.mat_elements[iter_nz].val;
csr.col_and_val.push_back(temp);
}

for (; ptr_tracker < coo.mat_data.num_row - 1; ptr_tracker++)
{
csr.row_ptr.push_back(coo.mat_data.num_nz);
}

csr.row_ptr.push_back(coo.mat_data.num_nz);

return true;
}

bool convert(COO &coo, CSC &csc)
{
sort_COO(coo, sort_target::CSC);
m_alloc.alloc_mat(coo.mat_data, csc);

IDX_VAL temp;
uint32_t ptr_tracker = 0;
csc.col_ptr.push_back(0);

for (uint32_t iter_nz = 0; iter_nz < coo.mat_data.num_nz; iter_nz++)
{

for (; ptr_tracker < coo.mat_elements[iter_nz].col_idx; ptr_tracker++)
{
csc.col_ptr.push_back(iter_nz);
}
temp.idx = coo.mat_elements[iter_nz].row_idx;
temp.val = coo.mat_elements[iter_nz].val;
csc.row_and_val.push_back(temp);
}

for (; ptr_tracker < coo.mat_data.num_col - 1; ptr_tracker++)
{
csc.col_ptr.push_back(coo.mat_data.num_nz);
}

csc.col_ptr.push_back(coo.mat_data.num_nz);

return true;
}

bool convert(CSR &csr, COO &coo)
{
m_alloc.alloc_mat(csr.mat_data, coo);

for (uint32_t row_ptr = 0; row_ptr < csr.mat_data.num_row; row_ptr++)
{
uint32_t row_start = csr.row_ptr[row_ptr];
uint32_t row_end = csr.row_ptr[row_ptr + 1];

MAT_ELE_DATA temp;
for (uint32_t row = row_start; row < row_end; row++)
{
temp.row_idx = row_ptr;
temp.col_idx = csr.col_and_val[row].idx;
temp.val = csr.col_and_val[row].val;
coo.mat_elements.push_back(temp);
}
}

sort_COO(coo, sort_target::CSR);

return true;
}

bool convert(CSR &csr, CSC &csc)
{
COO *temp = new COO();
this->convert(csr, *temp);
this->convert(*temp, csc);
delete temp;
return true;
}

bool convert(CSC &csc, COO &coo)
{
m_alloc.alloc_mat(csc.mat_data, coo);

for (uint32_t col_ptr = 0; col_ptr < csc.mat_data.num_col; col_ptr++)
{
uint32_t row_start = csc.col_ptr[col_ptr];
uint32_t row_end = csc.col_ptr[col_ptr + 1];

MAT_ELE_DATA temp;
for (uint32_t col = row_start; col < row_end; col++)
{
temp.col_idx = col_ptr;
temp.row_idx = csc.row_and_val[col].idx;
temp.val = csc.row_and_val[col].val;
coo.mat_elements.push_back(temp);
}
}

sort_COO(coo, sort_target::CSC);

return true;
}

bool convert(CSC &csc, CSR &csr)
{
COO *temp = new COO();
this->convert(csc, *temp);
this->convert(*temp, csr);
delete temp;
return true;
}

bool convert(D_MATRIX &d_mat, COO &coo)
{
m_alloc.alloc_mat(d_mat.mat_data, coo);

MAT_ELE_DATA temp;
for (uint32_t row = 0; row < d_mat.mat_data.num_row; row++)
{
for (uint32_t col = 0; col < d_mat.mat_data.num_col; col++)
{
if (d_mat.matrix[row][col] != 0)
{
temp.row_idx = row;
temp.col_idx = col;
temp.val = d_mat.matrix[row][col];
coo.mat_elements.push_back(temp);
}
}
}

sort_COO(coo, sort_target::CSC);

return true;
}

bool convert(COO &coo, D_MATRIX &d_mat)
{
m_alloc.alloc_mat(coo.mat_data.num_row, coo.mat_data.num_col, coo.mat_data.matrix_name, d_mat);

for (const auto &mat_ele : coo.mat_elements)
{
d_mat.matrix[mat_ele.row_idx][mat_ele.col_idx] = mat_ele.val;
}

return true;
}

bool transpose(COO &input, COO &output)
{
output.mat_data.matrix_name = input.mat_data.matrix_name + "_transposed";
output.mat_data.num_row = input.mat_data.num_col;
output.mat_data.num_col = input.mat_data.num_row;
output.mat_data.num_nz = input.mat_data.num_nz;

output.mat_elements = std::vector<MAT_ELE_DATA>(input.mat_data.num_nz, MAT_ELE_DATA{0, 0, 0});
for (uint32_t ele_A = 0; ele_A < input.mat_data.num_nz; ele_A++)
{
output.mat_elements[ele_A].row_idx = input.mat_elements[ele_A].col_idx;
output.mat_elements[ele_A].col_idx = input.mat_elements[ele_A].row_idx;
output.mat_elements[ele_A].val = input.mat_elements[ele_A].val;
}
return true;
}

bool transpose(CSR &input, CSR &output)
{
MATRIX_INFO temp_info;
temp_info.matrix_name = input.mat_data.matrix_name + "_Transposed";
temp_info.num_row = input.mat_data.num_col;
temp_info.num_col = input.mat_data.num_row;
temp_info.num_nz = input.mat_data.num_nz;

COO temp_coo;
m_alloc.alloc_mat(temp_info, temp_coo);
this->convert(input, temp_coo);

sort_COO(temp_coo, sort_target::CSC);
m_alloc.alloc_mat(temp_info, output);

IDX_VAL temp;
uint32_t ptr_tracker = 0;
output.row_ptr.push_back(0);
for (uint32_t iter_nz = 0; iter_nz < temp_coo.mat_data.num_nz; iter_nz++)
{
for (; ptr_tracker < temp_coo.mat_elements[iter_nz].col_idx; ptr_tracker++)
{
output.row_ptr.push_back(iter_nz);
}
temp.idx = temp_coo.mat_elements[iter_nz].row_idx;
temp.val = temp_coo.mat_elements[iter_nz].val;
output.col_and_val.push_back(temp);
}

for (; ptr_tracker < temp_coo.mat_data.num_col - 1; ptr_tracker++)
{
output.row_ptr.push_back(temp_coo.mat_data.num_nz);
}

output.row_ptr.push_back(temp_coo.mat_data.num_nz);

return true;
}

bool transpose(CSC &input, CSC &output)
{
MATRIX_INFO temp_info;
temp_info.matrix_name = input.mat_data.matrix_name + "_Transposed";
temp_info.num_row = input.mat_data.num_col;
temp_info.num_col = input.mat_data.num_row;
temp_info.num_nz = input.mat_data.num_nz;

COO temp_coo;
m_alloc.alloc_mat(temp_info, temp_coo);
this->convert(input, temp_coo);

sort_COO(temp_coo, sort_target::CSR);
m_alloc.alloc_mat(temp_info, output);

IDX_VAL temp;
uint32_t ptr_tracker = 0;
output.col_ptr.push_back(0);
for (uint32_t iter_nz = 0; iter_nz < temp_coo.mat_data.num_nz; iter_nz++)
{
for (; ptr_tracker < temp_coo.mat_elements[iter_nz].row_idx; ptr_tracker++)
{
output.col_ptr.push_back(iter_nz);
}
temp.idx = temp_coo.mat_elements[iter_nz].col_idx;
temp.val = temp_coo.mat_elements[iter_nz].val;
output.row_and_val.push_back(temp);
}

for (; ptr_tracker < temp_coo.mat_data.num_col - 1; ptr_tracker++)
{
output.col_ptr.push_back(temp_coo.mat_data.num_nz);
}

output.col_ptr.push_back(temp_coo.mat_data.num_nz);

return true;
}

private:
MATRIX_ALLOCATION m_alloc;

enum class sort_target
{
CSR,
CSC
};

void sort_COO(COO &s_mat, sort_target target)
{
switch (target)
{
case sort_target::CSR:
std::sort(s_mat.mat_elements.begin(), s_mat.mat_elements.end(), sort_for_CSR);
break;
case sort_target::CSC:
std::sort(s_mat.mat_elements.begin(), s_mat.mat_elements.end(), sort_for_CSC);
break;
}
}

struct
{
bool operator()(MAT_ELE_DATA one, MAT_ELE_DATA two) const
{
if (one.row_idx > two.row_idx)
{
return false;
}
else if (one.row_idx == two.row_idx && one.col_idx > two.col_idx)
{
return false;
}
else
{
return true;
}
}
} sort_for_CSR;

struct
{
bool operator()(MAT_ELE_DATA one, MAT_ELE_DATA two) const
{
if (one.col_idx > two.col_idx)
{
return false;
}
else if (one.col_idx == two.col_idx && one.row_idx > two.row_idx)
{
return false;
}
else
{
return true;
}
}
} sort_for_CSC;
};