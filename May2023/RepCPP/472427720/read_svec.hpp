#pragma once
#include "PCH.hpp"
#include "matrix_struct.hpp"

class READ_SVEC
{
public:
READ_SVEC() {}

~READ_SVEC() {}

bool load_svec_ANSI_BASE(std::string file_name, S_VECTOR &output_vec)
{
FILE *load_svec;

load_svec = fopen(file_name.c_str(), "r");
if (load_svec == NULL)
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
int retrived = fgetc(load_svec);
if (retrived == '%')
{
fgets(read_line, read_max, load_svec);
}
else
{
ungetc(retrived, load_svec);
fscanf(load_svec, "%u %u\n", &vec_len, &num_nz);
break;
}
}

output_vec.vec_element = std::vector<IDX_VAL>(num_nz, {0, 0});
for (uint32_t i = 0; i < num_nz; i++)
{
fscanf(load_svec, "%u %lg\n", &output_vec.vec_element[i].idx, &output_vec.vec_element[i].val);
output_vec.vec_element[i].idx--;
}
fclose(load_svec);

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}

bool load_svec_UNSAFE(std::string file_name, S_VECTOR &output_vec)
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

IDX_VAL temp;
output_vec.vec_element.reserve(num_nz);
for (uint32_t track_nnz = 0; track_nnz < num_nz; track_nnz++)
{
load_d_vec >> temp.idx >> temp.val;
temp.idx--;
output_vec.vec_element.push_back(temp);
}
load_d_vec.close();

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}

bool load_svec_SAFE(std::string file_name, S_VECTOR &output_vec)
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
std::regex info_regex("\\d+ \\d+");
std::getline(load_d_vec, read_line, '\n');
if (std::regex_match(read_line, info_regex))
{
std::istringstream str_stream(read_line);
str_stream >> vec_len >> num_nz;
}
else
{
std::cout << "Detected Error in vector information line.\n"
<< "vector information must contains two data - length number_of_non-zero\n"
<< "The variables cannot be string nor negative number\n"
<< "Retrived Information line: " << read_line << "\n"
<< std::endl;
return false;
}

uint32_t nnz_counter = 0;
IDX_VAL temp;
output_vec.vec_element.reserve(num_nz);
std::regex element_regex("\\d+ [-+]?(\\d+)?([.]?(\\d+))$");
while (std::getline(load_d_vec, read_line, '\n'))
{
nnz_counter++;
if (std::regex_match(read_line, element_regex) == true)
{
std::istringstream str_stream(read_line);
str_stream >> temp.idx >> temp.val;

if (idx_check(vec_len, temp.idx) == false)
{
return false;
}
else
{
temp.idx--;
output_vec.vec_element.push_back(temp);
}
}
else
{
std::cout << "Detected Error in vector element format.\n"
<< "Retrived matrix elements: " << read_line << "\n"
<< std::endl;
return false;
}
}

output_vec.vec_data.vec_name = file_name;
output_vec.vec_data.len_vec = vec_len;
output_vec.vec_data.num_nz = num_nz;

return true;
}

private:
bool idx_check(uint32_t idx_limit, uint32_t idx) const
{
if (idx > idx_limit)
{
std::cout << "CORRUPTED DATA\n"
<< "One of idx value exceed length of vector\n"
<< "Vector Length - " << idx_limit << "\n"
<< "RECEIVED    - " << idx << "\n"
<< std::endl;
return false;
}
return true;
}
};