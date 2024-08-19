#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class WRITE_DVEC
{
public:
WRITE_DVEC() {}

virtual ~WRITE_DVEC() {}

bool write_dvec_ANSI_based(std::string file_name, D_VECTOR &output_vec)
{
FILE *write_dvec;
write_dvec = fopen(file_name.c_str(), "w");

std::string vec_name = "%" + output_vec.vec_data.vec_name + "\n";
fprintf(write_dvec, "%s", vec_name.c_str());

fprintf(write_dvec, "%u %u\n", output_vec.vec_data.len_vec, output_vec.vec_data.num_nz);

for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.len_vec; ele_A++)
{
fprintf(write_dvec, "%lg\n", output_vec.vec_element[ele_A]);
}
fclose(write_dvec);

return true;
}

bool write_dvec_UNSAFE(std::string file_name, D_VECTOR &output_vec)
{
std::ofstream write_dvec_file;
write_dvec_file.open(file_name, std::ios::out);
if (write_dvec_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_vec.vec_data.vec_name + "\n";
write_dvec_file << output_line;

output_line =
std::to_string(output_vec.vec_data.len_vec) + " " +
std::to_string(output_vec.vec_data.num_nz) + "\n";
write_dvec_file << output_line;

for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.len_vec; ele_A++)
{
write_dvec_file << std::to_string(output_vec.vec_element[ele_A]) << "\n";
}
write_dvec_file.close();
return true;
}

bool write_dvec_SAFE(std::string file_name, D_VECTOR &output_vec)
{
std::ofstream write_dvec_file;
write_dvec_file.open(file_name, std::ios::out);
if (write_dvec_file.is_open() == false)
{
std::cout << "Failed to generate: " << file_name << "\n";
return false;
}

std::string output_line;
output_line = "%" + output_vec.vec_data.vec_name + "\n";
write_dvec_file << output_line;

output_line =
std::to_string(output_vec.vec_data.len_vec) + " " +
std::to_string(output_vec.vec_data.num_nz) + "\n";

std::regex info_regex("\\d+ \\d+$");
if (std::regex_match(output_line, info_regex) == false)
{
return false;
}
write_dvec_file << output_line;

std::regex element_regex("[-+]?(\\d+)?([.]?(\\d+))$");
for (uint32_t ele_A = 0; ele_A < output_vec.vec_data.num_nz; ele_A++)
{
output_line = std::to_string(output_vec.vec_element[ele_A]);
if (std::regex_match(output_line, element_regex) == false)
{
return false;
}
write_dvec_file << output_line << "\n";
}
write_dvec_file.close();
return true;
}

private:
};