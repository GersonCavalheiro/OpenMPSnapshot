#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class WRITE_MTX
{
public:
WRITE_MTX() {}

virtual ~WRITE_MTX() {}

bool write_mtx_ANSI_based(std::string file_name, COO &output_mat)
{
FILE *write_mtx;
write_mtx = fopen(file_name.c_str(), "w");

std::string mat_name = "%" + output_mat.mat_data.matrix_name + "\n";
fprintf(write_mtx, "%s", mat_name.c_str());

fprintf(write_mtx, "%u %u %u\n", output_mat.mat_data.num_row, output_mat.mat_data.num_col, output_mat.mat_data.num_nz);

for (uint32_t ele_A = 0; ele_A < output_mat.mat_data.num_nz; ele_A++)
{
fprintf(write_mtx, "%u %u %lg\n",
output_mat.mat_elements[ele_A].row_idx + 1,
output_mat.mat_elements[ele_A].col_idx + 1,
output_mat.mat_elements[ele_A].val);
}
fclose(write_mtx);

return false;
}

bool write_mtx_UNSAFE(std::string file_name, COO &output_mat)
{
std::ofstream write_mtx_file;
write_mtx_file.open(file_name, std::ios::out);
if (write_mtx_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_mat.mat_data.matrix_name + "\n";
write_mtx_file << output_line;

output_line =
std::to_string(output_mat.mat_data.num_row) + " " +
std::to_string(output_mat.mat_data.num_col) + " " +
std::to_string(output_mat.mat_data.num_nz) + "\n";

write_mtx_file << output_line;

for (uint32_t ele_A = 0; ele_A < output_mat.mat_data.num_nz; ele_A++)
{
output_line =
std::to_string(output_mat.mat_elements[ele_A].row_idx + 1) + " " +
std::to_string(output_mat.mat_elements[ele_A].col_idx + 1) + " " +
std::to_string(output_mat.mat_elements[ele_A].val) + "\n";
write_mtx_file << output_line;
}
write_mtx_file.close();
return true;
}

bool write_mtx_SAFE(std::string file_name, COO &output_mat)
{
std::ofstream write_mtx_file;
write_mtx_file.open(file_name, std::ios::out);
if (write_mtx_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_mat.mat_data.matrix_name + "\n";
write_mtx_file << output_line;

output_line =
std::to_string(output_mat.mat_data.num_row) + " " +
std::to_string(output_mat.mat_data.num_col) + " " +
std::to_string(output_mat.mat_data.num_nz) + "\n";

std::regex info_regex("\\d+ \\d+ \\d+$");
if (std::regex_match(output_line, info_regex) == false)
{
return false;
}
write_mtx_file << output_line;

std::regex element_regex("\\d+ \\d+ [-+]?(\\d+)?([.]?(\\d+))$");
for (uint32_t ele_A = 0; ele_A < output_mat.mat_data.num_nz; ele_A++)
{
output_line =
std::to_string(output_mat.mat_elements[ele_A].row_idx + 1) + " " +
std::to_string(output_mat.mat_elements[ele_A].col_idx + 1) + " " +
std::to_string(output_mat.mat_elements[ele_A].val) + "\n";

if (std::regex_match(output_line, element_regex) == false)
{
return false;
}
write_mtx_file << output_line;
}
write_mtx_file.close();
return true;
}

private:
};