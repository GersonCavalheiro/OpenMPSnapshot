#include <iostream>
#include "crypto/sha1.hpp"
#include "Utility.h"
#include <omp.h>
#include <chrono>
#include <cstring>

int main(int argc, char* argv[]) {

const std::string authdata("abcdefghijklmnoprstxwyz1234567890!@#$%^&*()_+");
size_t length = 10;
char destptr[50];
std::string suffix;
suffix.reserve(length);
SHA1 checksum;
const size_t difficulty = 9;
omp_set_num_threads(omp_get_max_threads());


int sstop =0;
bool found;

std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel private(found, suffix, destptr)
{
while (!sstop) {
utility::randomStr(destptr, length);
std::string(buffer);
buffer.append(authdata).append(destptr);
checksum.update(buffer);
const std::string hash = checksum.final();
found = hash.substr(0, difficulty).find_first_not_of(definition::zero) == std::string::npos;

if (found) {
sstop = 1; 
printf("suffix: %s ->\t hash: %s\n", destptr, hash.c_str());
#pragma omp flush(sstop)
}
}
}

std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::minutes>(t2 - t1).count();

printf("Process time (in minutes) -> %d\n", duration);
return 0;
}
