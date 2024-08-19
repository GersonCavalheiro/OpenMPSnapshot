#pragma once

#include "blas.hpp"
#include "csmat.hpp"

#include <utility>
#include <vector>

namespace algorithms {
template <typename T = double>
auto hits(CsMat<T> const& A, double error) -> std::pair<std::vector<T>, std::vector<T>> {
size_t const n = A.get_nrows();

std::vector<T> a(n, 1.0);
std::vector<T> h(n, 1.0);

CsMat<T> AT = A.transposed();

T norm_a_prev;
T norm_h_prev;

do {
std::vector<T> a_prev = a;
std::vector<T> h_prev = h;

blas::spmv(1.0, A, h, 0.0, a);
blas::normalize(a);

blas::spmv(1.0, AT, a, 0.0, h);
blas::normalize(h);

norm_a_prev = blas::norm2(a_prev);
norm_h_prev = blas::norm2(h_prev);
} while (blas::norm2(a) - norm_a_prev > error && blas::norm2(h) - norm_h_prev > error);

return std::make_pair(a, h);
}
} 
