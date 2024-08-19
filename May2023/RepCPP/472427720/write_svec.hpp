#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class WRITE_SVEC
{
public:
WRITE_SVEC() {}

virtual ~WRITE_SVEC() {}

bool write_svec_ANSI_based(std::string file_name, S_VECTOR &output_vec)
{
FILE *write_svec;
write_svec = fopen(file_name.c_str(), "w");

std::string vec_name = "%" + output_vec.vec_data.vec_name + "\n";
fprintf(write_svec, "%s", vec_name.c_str());

fprintf(write_svec, "%u %u\n", output_vec.vec_data.len_vec, output_vec.vec_data.num_nz);

for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.num_nz; ele_A++)
{
fprintf(write_svec, "%u %lg\n",
output_vec.vec_element[ele_A].idx + 1,
output_vec.vec_element[ele_A].val);
}
fclose(write_svec);

return false;
}

bool write_svec_UNSAFE(std::string file_name, S_VECTOR &output_vec)
{
std::ofstream write_svec_file;
write_svec_file.open(file_name, std::ios::out);
if (write_svec_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_vec.vec_data.vec_name + "\n";
write_svec_file << output_line;

output_line =
std::to_string(output_vec.vec_data.len_vec) + " " +
std::to_string(output_vec.vec_data.num_nz) + "\n";
write_svec_file << output_line;

for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.num_nz; ele_A++)
{
output_line =
std::to_string(output_vec.vec_element[ele_A].idx + 1) + " " +
std::to_string(output_vec.vec_element[ele_A].val) + "\n";
write_svec_file << output_line;
}
write_svec_file.close();
return true;
}

bool write_svec_SAFE(std::string file_name, S_VECTOR &output_vec)
{
std::ofstream write_svec_file;
write_svec_file.open(file_name, std::ios::out);
if (write_svec_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_vec.vec_data.vec_name + "\n";
write_svec_file << output_line;

output_line =
std::to_string(output_vec.vec_data.len_vec) + " " +
std::to_string(output_vec.vec_data.num_nz) + "\n";

std::regex info_regex("\\d+ \\d+$");
if (std::regex_match(output_line, info_regex) == false)
{
return false;
}
write_svec_file << output_line;

std::regex element_regex("\\d+ [-+]?(\\d+)?([.]?(\\d+))$");
for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.num_nz; ele_A++)
{
output_line =
std::to_string(output_vec.vec_element[ele_A].idx + 1) + " " +
std::to_string(output_vec.vec_element[ele_A].val) + "\n";

if (std::regex_match(output_line, element_regex) == false)
{
return false;
}
write_svec_file << output_line;
}
write_svec_file.close();
return true;
}

private:
};