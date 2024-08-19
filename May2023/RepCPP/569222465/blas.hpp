#pragma once

#include "csmat.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <omp.h>

namespace blas {

template <typename T>
auto norm2(std::vector<T> const& x) -> T {
return sqrt(std::reduce(x.begin(), x.end(), 0.0));
}


template <typename T>
auto normalize(std::vector<T>& x) -> void {
T norm = norm2(x);
#if defined(PARALLEL)
#pragma omp parallel
#endif
std::for_each(x.begin(), x.end(), [norm](T& v) {
v /= norm;
});
}


template <typename T>
auto normalize_rows(CsMat<T> const& A) -> CsMat<T> {
CsMat<T> A_normalized_rows = A;

#if defined(PARALLEL)
#pragma omp parallel for
#endif
for (size_t i = 0; i < A.get_nrows(); ++i) {
std::vector<T> row_data;
#pragma omp simd
for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; ++j) {
row_data.push_back(A.get_data()[j]);
}

normalize(row_data);

#pragma omp simd
for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; ++j) {
A_normalized_rows.get_mut_data()[j] = row_data[j - A.get_indptr()[i]];
}
}

return A_normalized_rows;
}


template <typename T>
auto spmv(
T alpha,
CsMat<T> const& A,
std::vector<T> const& x,
T beta,
std::vector<T>& y
) -> void {
assert(A.get_ncols() == x.size());
assert(A.get_ncols() == y.size());

std::vector<T> const& values = A.get_data();
std::vector<size_t> const& indices = A.get_indices();
std::vector<size_t> const& indptr = A.get_indptr();

#if defined(PARALLEL)
#pragma omp parallel for
#endif
for (size_t i = 0; i < A.get_nrows(); ++i) {
T tmp = y[i] * beta;
#pragma omp simd
for (size_t j = indptr[i]; j < indptr[i + 1]; ++j) {
tmp += alpha * values[j] * x[indices[j]];
}
y[i] += tmp;
}
}

template <typename T>
auto spmm(CsMat<T> const& A, CsMat<T> const& B) -> CsMat<T> {
assert(A.get_ncols() == B.get_nrows());

CsMat<T> C;
C.get_mut_nrows() = A.get_nrows();
C.get_mut_ncols() = B.get_ncols();

C.get_mut_indptr().push_back(0);
#if defined(PARALLEL)
#pragma omp parallel for
#endif
for (size_t i = 0; i < A.get_nrows(); i++) {
std::vector<T> crow(B.get_ncols(), 0.0);
for (size_t j = A.get_indptr()[i]; j < A.get_indptr()[i + 1]; j++) {
T val = A.get_data()[j];
#pragma omp simd
for (
size_t k = B.get_indptr()[A.get_indices()[j]];
k < B.get_indptr()[A.get_indices()[j] + 1];
k++
) {
crow[B.get_indices()[k]] += val * B.get_data()[k];
}
}

#pragma omp simd
for (size_t j = 0; j < B.get_ncols(); j++) {
if (crow[j] != 0.0) {
C.get_mut_data().push_back(crow[j]);
C.get_mut_indices().push_back(j);
}
}
C.get_mut_indptr().push_back(C.get_indices().size());
}

return C;
}
} 
