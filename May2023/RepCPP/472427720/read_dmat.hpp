#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class READ_DMAT
{
public:
READ_DMAT() {}

~READ_DMAT() {}

bool load_dmat_ANSI_BASE(std::string file_name, D_MATRIX &output_mat)
{
return false;
}

bool load_dmat_UNSAFE(std::string file_name, D_MATRIX &output_mat)
{
std::ifstream load_d_mat;
load_d_mat.open(file_name, std::ios::in);
if (load_d_mat.is_open() == false)
{
std::cout << "Failed to open: " << file_name << "\n"
<< "Please check " << file_name << " is exist.\n"
<< std::endl;
return false;
}

std::string read_line;
while (load_d_mat.peek() == '%')
{
std::getline(load_d_mat, read_line);
}

uint32_t num_row = 0;
uint32_t num_col = 0;
uint32_t num_nz = 0;
load_d_mat >> num_row >> num_col >> num_nz;

for (uint32_t row = 0; row < num_row; row++)
{
std::vector<double> temp_vec(num_col, 0);
output_mat.matrix.push_back(temp_vec);
}

double temp;
for (uint32_t row = 0; row < num_row; row++)
{
for (uint32_t col = 0; col < num_col; col++)
{
load_d_mat >> temp;
output_mat.matrix[row][col] = temp;
}
}

output_mat.mat_data.matrix_name = file_name;
output_mat.mat_data.num_row = num_row;
output_mat.mat_data.num_col = num_col;
output_mat.mat_data.num_nz = num_nz;

return true;
}

bool load_dmat_SAFE(std::string file_name, D_MATRIX &output_mat)
{
return false;
}

private:
};