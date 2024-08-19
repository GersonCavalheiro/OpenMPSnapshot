#pragma once

#include "blas.hpp"
#include "csmat.hpp"

#include <utility>
#include <vector>

namespace algorithms {
template <typename T>
auto salsa(CsMat<T> const& A) -> std::pair<std::vector<T>, std::vector<T>> {
CsMat<T> A_r = blas::normalize_rows(A);

CsMat<T> AT = A.transpose();

CsMat<T> A_cT = blas::normalize_rows(AT);

CsMat<T> A_tilde = blas::spmm(A_cT, A_r);

CsMat<T> H_tilde = blas::spmm(A_r, A_cT);


}
} 