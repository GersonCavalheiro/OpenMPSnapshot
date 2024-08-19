#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class READ_MTX
{
public:
READ_MTX() {}

virtual ~READ_MTX() {}

bool load_matrix_mtx_ANSI_based(std::string file_name, COO &output_mat)
{
FILE *load_mtx;

load_mtx = fopen(file_name.c_str(), "r");
if (load_mtx == NULL)
{
printf("Failed to open file\n");
return false;
}

#define read_max 256
char read_line[read_max] = {'0'};

uint32_t length_col = 0;
uint32_t length_row = 0;
uint32_t num_none_zero = 0;
while (1)
{
int retrived = fgetc(load_mtx);
if (retrived == '%')
{
fgets(read_line, read_max, load_mtx);
}
else
{
ungetc(retrived, load_mtx);
fscanf(load_mtx, "%u %u %u\n", &length_row, &length_col, &num_none_zero);
break;
}
}

output_mat.mat_elements.reserve(num_none_zero);
MAT_ELE_DATA temp;
for (uint32_t i = 0; i < num_none_zero; i++)
{
fscanf(load_mtx, "%u %u %lg\n", &temp.row_idx, &temp.col_idx, &temp.val);
temp.row_idx--;
temp.col_idx--;
output_mat.mat_elements.push_back(temp);
}

fclose(load_mtx);

output_mat.mat_data.matrix_name = file_name;
output_mat.mat_data.num_row = length_row;
output_mat.mat_data.num_col = length_col;
output_mat.mat_data.num_nz = num_none_zero;

return true;
}

bool load_matrix_mtx_UNSAFE(std::string file_name, COO &output_mat)
{
std::ifstream load_mtx_file;
load_mtx_file.open(file_name, std::ios::in);
if (load_mtx_file.is_open() == false)
{
std::cout << "Failed to open: " << file_name << "\n"
<< "Please check " << file_name << " is exist.\n"
<< std::endl;
return false;
}

std::string read_line;
while (load_mtx_file.peek() == '%')
{
std::getline(load_mtx_file, read_line);
}

uint32_t num_row = 0;
uint32_t num_col = 0;
uint32_t num_nz = 0;
load_mtx_file >> num_row >> num_col >> num_nz;

MAT_ELE_DATA temp_data;
std::vector<MAT_ELE_DATA> vec_init(num_nz, temp_data);
output_mat.mat_elements = vec_init;
for (uint32_t track_nnz = 0; track_nnz < num_nz; track_nnz++)
{
load_mtx_file >> temp_data.row_idx >> temp_data.col_idx >> temp_data.val;
if (idx_check(num_row, num_col, temp_data.row_idx, temp_data.col_idx) == false)
{
return false;
}
else
{
temp_data.row_idx--;
temp_data.col_idx--;
output_mat.mat_elements[track_nnz] = temp_data;
}
}
load_mtx_file.close();

output_mat.mat_data.matrix_name = file_name;
output_mat.mat_data.num_row = num_row;
output_mat.mat_data.num_col = num_col;
output_mat.mat_data.num_nz = num_nz;

return true;
}

bool load_matrix_mtx_SAFE(std::string file_name, COO &output_mat)
{
std::ifstream load_mtx_file;
load_mtx_file.open(file_name, std::ios::in);
if (load_mtx_file.is_open() == false)
{
std::cout << "Failed to open: " << file_name << "\n"
<< "Please check " << file_name << " is exist.\n"
<< std::endl;
return false;
}

std::string read_line;
while (load_mtx_file.peek() == '%')
{
std::getline(load_mtx_file, read_line);
}

uint32_t num_row = 0;
uint32_t num_col = 0;
uint32_t num_nz = 0;
std::regex info_regex("\\d+ \\d+ [-+]?(\\d+)$");
std::getline(load_mtx_file, read_line, '\n');
if (std::regex_match(read_line, info_regex))
{
std::istringstream str_stream(read_line);
str_stream >> num_row >> num_col >> num_nz;
}
else
{
std::cout << "Detected Error in matrix information line.\n"
<< "Matrix information must contains three data - row column number_of_non-zero\n"
<< "The variables cannot be string nor negative number\n"
<< "Retrived Information line: " << read_line << "\n"
<< std::endl;
return false;
}

uint32_t nnz_counter = 0;
MAT_ELE_DATA temp;
output_mat.mat_elements.reserve(num_nz);
std::regex element_regex("\\d+ \\d+ [-+]?(\\d+)?([.]?(\\d+))$");
while (std::getline(load_mtx_file, read_line, '\n'))
{
nnz_counter++;
if (std::regex_match(read_line, element_regex) == true)
{
std::istringstream str_stream(read_line);
str_stream >> temp.row_idx >> temp.col_idx >> temp.val;

if (idx_check(num_row, num_col, temp.row_idx, temp.col_idx) == false)
{
return false;
}
else
{
temp.row_idx--;
temp.col_idx--;
output_mat.mat_elements.push_back(temp);
}
}
else
{
std::cout << "Detected Error in matrix element format.\n"
<< "The program supports .mtx file with regular matrix only.\n"
<< "The program does not supports symmetric matrix nor matrix with complex value.\n"
<< "Retrived matrix elements: " << read_line << "\n"
<< std::endl;
return false;
}
}

if (nnz_counter != num_nz && load_mtx_file.eof())
{
std::cout << "Detected Error in number of non-zero\n"
<< "Matrix loading process terminated by reaching end of file.\n"
<< "However, number of loaded non-zero does match with matrix information\n"
<< "Expected NZ | Counted NZ =>" << num_nz << " | " << nnz_counter << "\n"
<< std::endl;
return false;
}

load_mtx_file.close();

output_mat.mat_data.matrix_name = file_name;
output_mat.mat_data.num_row = num_row;
output_mat.mat_data.num_col = num_col;
output_mat.mat_data.num_nz = num_nz;

return true;
}

private:
bool idx_check(uint32_t row_limit, uint32_t col_limit, uint32_t row_ind, uint32_t col_ind) const
{
if (row_ind > row_limit || col_ind > col_limit)
{
std::cout << "CORRUPTED DATA\n"
<< "One of idx value exceed row/column of matrix\n"
<< "MAX ROW/COL - " << row_limit << "/" << col_limit << "\n"
<< "RECEIVED    - " << row_ind << "/" << col_ind << "\n"
<< std::endl;
return false;
}
return true;
}
};