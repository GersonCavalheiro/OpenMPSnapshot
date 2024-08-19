#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class READ_DVEC
{
public:
READ_DVEC() {}

~READ_DVEC() {}

bool load_dvec_ANSI_BASE(std::string file_name, D_VECTOR &output_vec)
{
FILE *load_dvec;

load_dvec = fopen(file_name.c_str(), "r");
if (load_dvec == NULL)
{
printf("Failed to open file\n");
return false;
}

#define read_max 256
char read_line[read_max] = {'0'};

uint32_t vec_len = 0;
uint32_t num_nz = 0;
while (1)
{
int retrived = fgetc(load_dvec);
if (retrived == '%')
{
fgets(read_line, read_max, load_dvec);
}
else
{
ungetc(retrived, load_dvec);
fscanf(load_dvec, "%u %u\n", &vec_len, &num_nz);
break;
}
}

output_vec.vec_element = std::vector<double>(vec_len, 0);
for (uint32_t i = 0; i < vec_len; i++)
{
fscanf(load_dvec, "%lg\n", &output_vec.vec_element[i]);
}
fclose(load_dvec);

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}

bool load_dvec_UNSAFE(std::string file_name, D_VECTOR &output_vec)
{
std::ifstream load_d_vec;
load_d_vec.open(file_name, std::ios::in);
if (load_d_vec.is_open() == false)
{
std::cout << "Failed to open: " << file_name << "\n"
<< "Please check " << file_name << " is exist.\n"
<< std::endl;
return false;
}

std::string read_line;
while (load_d_vec.peek() == '%')
{
std::getline(load_d_vec, read_line);
}

uint32_t vec_len = 0;
uint32_t num_nz = 0;
load_d_vec >> vec_len >> num_nz;

double temp_val;
output_vec.vec_element.reserve(num_nz);
for (uint32_t track_nnz = 0; track_nnz < vec_len; track_nnz++)
{
load_d_vec >> temp_val;
output_vec.vec_element.push_back(temp_val);
}
load_d_vec.close();

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}

bool load_dvec_SAFE(std::string file_name, D_VECTOR &output_vec)
{
std::ifstream load_d_vec;
load_d_vec.open(file_name, std::ios::in);
if (load_d_vec.is_open() == false)
{
std::cout << "Failed to open: " << file_name << "\n"
<< "Please check " << file_name << " is exist.\n"
<< std::endl;
return false;
}

std::string read_line;
while (load_d_vec.peek() == '%')
{
std::getline(load_d_vec, read_line);
}

uint32_t vec_len = 0;
uint32_t num_nz = 0;
load_d_vec >> vec_len >> num_nz;

std::string readline;
output_vec.vec_element.reserve(num_nz);
for (uint32_t track_nnz = 0; track_nnz < vec_len; track_nnz++)
{
std::getline(load_d_vec, read_line, '\n');
output_vec.vec_element.push_back(std::stod(readline));
}
load_d_vec.close();

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}
};