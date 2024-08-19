#include "hpc++11.h"

#include <cassert>
#include <thread>
#include <omp.h>

void sum_vectors(std::vector<float> const &a,
std::vector<float> const &b,
std::vector<float> &c) {

assert(a.size() == b.size());
assert(a.size() == c.size());

unsigned nothreads = std::thread::hardware_concurrency();

const auto a_data = a.data();
const auto b_data = b.data();
auto c_data = c.data();
size_t size = c.size();

omp_set_dynamic(0);

#pragma omp parallel for num_threads(nothreads) shared(size)
for (size_t i = 0; i < size; ++i) {
c_data[i] = a_data[i] + b_data[i];
}
}
