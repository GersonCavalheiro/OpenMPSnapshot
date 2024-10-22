#include <iostream>
#include <cstdint>
#include <vector>

#include "../include/hpc_helpers.hpp"

template <typename value_t,
typename index_t>
void init(std::vector<value_t>& A,
std::vector<value_t>& x,
index_t m,
index_t n) {

for (index_t row = 0; row < m; row++)
for (index_t col = 0; col < n; col++)
A[row*n+col] = row >= col ? 1 : 0;

for (index_t col = 0; col < m; col++)
x[col] = col;
}

template <typename value_t,
typename index_t>
void mult(std::vector<value_t>& A,
std::vector<value_t>& x,
std::vector<value_t>& b,
index_t m,
index_t n,
bool parallel) {

#pragma omp parallel for if(parallel)
for (index_t row = 0; row < m; row++) {
value_t accum = value_t(0);
for (index_t col = 0; col < n; col++)
accum += A[row*n+col]*x[col];
b[row] = accum;
}
}

int main() {
const uint64_t n = 1UL << 15;
const uint64_t m = 1UL << 15;

TIMERSTART(overall)
TIMERSTART(alloc)
std::vector<no_init_t<uint64_t>> A(m*n);
std::vector<no_init_t<uint64_t>> x(n);
std::vector<no_init_t<uint64_t>> b(m);
TIMERSTOP(alloc)

TIMERSTART(init)
init(A, x, m, n);
TIMERSTOP(init)

for (uint64_t k = 0; k < 3; k++) {
TIMERSTART(mult_seq)
mult(A, x, b, m, n, false);
TIMERSTOP(mult_seq)
}
for (uint64_t k = 0; k < 3; k++) {
TIMERSTART(mult_par)
mult(A, x, b, m, n, true);
TIMERSTOP(mult_par)
}
TIMERSTOP(overall)

for (uint64_t index = 0; index < m; index++)
if (b[index] != index*(index+1)/2)
std::cout << "error at position " << index 
<< " " << b[index] << std::endl;
}

