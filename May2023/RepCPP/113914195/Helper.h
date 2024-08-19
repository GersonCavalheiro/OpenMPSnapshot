#pragma once





#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;





long int file_size(const char file_path[]);

void read_datafile(const char file_path[], unsigned char *plaintexts);

unsigned char* read_key(const string &file_path);

unsigned char* random_byte_array(const unsigned int &length);

bool check_byte_arrays(unsigned char *arr1, unsigned char *arr2, const unsigned int &size);

void print_byte(const unsigned char &byte);

unsigned char* XOR(const unsigned char *arr1, const unsigned char *arr2);