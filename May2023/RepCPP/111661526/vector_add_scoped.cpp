#include <iostream>
#include <cstdint>
#include <vector>

#include "../include/hpc_helpers.hpp"

int main() {
TIMERSTART(alloc)
const uint64_t num_entries = 1UL << 30;
std::vector<no_init_t<uint64_t>> x(num_entries);
std::vector<no_init_t<uint64_t>> y(num_entries);
std::vector<no_init_t<uint64_t>> z(num_entries);
TIMERSTOP(alloc)

TIMERSTART(alltogether)
#pragma omp parallel 	    
{
#pragma omp for
for (uint64_t i = 0; i < num_entries; i++) {
x[i] = i;
y[i] = num_entries - i;
}

#pragma omp for
for (uint64_t i = 0; i < num_entries; i++)
z[i] = x[i] + y[i];

#pragma omp for
for (uint64_t i = 0; i < num_entries; i++)
if (z[i] - num_entries)
std::cout << "error at position "
<< i << std::endl;
}
TIMERSTOP(alltogether)
}
