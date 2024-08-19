#include <iostream>   
#include <vector>     

#include "../include/hpc_helpers.hpp"
#include "../include/binary_IO.hpp"

#define MODE dynamic

template <typename value_t,
typename index_t>
void inner_product(value_t * data,
value_t * delta,
index_t num_entries,
index_t num_features,
bool    parallel) {

#pragma omp parallel for schedule(MODE) if(parallel)
for (index_t i = 0; i < num_entries; i++)
for (index_t j = i; j < num_entries; j++) {
value_t accum = value_t(0);
for (index_t k = 0; k < num_features; k++)
accum += data[i*num_features+k] *
data[j*num_features+k];
delta[i*num_entries+j] =
delta[j*num_entries+i] = accum;
}
}

int main(int argc, char* argv[]) {

const bool parallel = argc > 1;

std::cout << "running "
<< (parallel ? "in parallel" : "sequentially")
<< std::endl;

const uint64_t num_features = 28*28;
const uint64_t num_entries = 65000;

TIMERSTART(alloc)
std::vector<float> input(num_entries*num_features);
std::vector<float> delta(num_entries*num_entries);
TIMERSTOP(alloc)

TIMERSTART(read_data)
load_binary(input.data(), input.size(), "./data/X.bin");
TIMERSTOP(read_data)

TIMERSTART(inner_product)
inner_product(input.data(), delta.data(),
num_entries, num_features, parallel);
TIMERSTOP(inner_product)
}
