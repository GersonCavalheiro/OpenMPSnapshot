#pragma once





#include <vector>
#include <string>
#include <iostream>

#include "Helper.hpp"
#include "AES.hpp"

using std::cout;
using std::endl;
using std::vector;
using std::string;





const vector<ByteArray> read_datafile(const string &file_path);

const ByteArray read_key(const string &file_path);

const ByteArray random_byte_array(const unsigned int &length);

ByteArray increment_counter(const ByteArray &start_counter,
const unsigned int &round);

void generate_counters(vector<ByteArray> &ctrs, const ByteArray &IV);

const vector<ByteArray> counter_mode(const vector<ByteArray> &messages,
const ByteArray &key,
const ByteArray &IV);

const vector<ByteArray> counter_mode_inverse(const vector<ByteArray> &encrypted_messages,
const ByteArray &key,
const ByteArray &IV);