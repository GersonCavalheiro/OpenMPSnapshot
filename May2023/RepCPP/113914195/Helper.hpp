#pragma once





#include <vector>
#include <iostream>
#include <bitset>

#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;





const vector<ByteArray> read_datafile(const string &file_path);

const ByteArray read_key(const string &file_path);

const ByteArray random_byte_array(const unsigned int &length);

void print_byte_array(ByteArray &arr);

bool check_byte_arrays(const ByteArray &arr1, const ByteArray &arr2);

bool check_vector_of_byte_arrays(const vector<ByteArray> &arr1, const vector<ByteArray> &arr2);

void print_byte(const unsigned char &byte);

ByteArray XOR(const ByteArray &arr1, const ByteArray &arr2);

void XOR(ByteArray &arr1, const ByteArray &arr2, const unsigned int &length);