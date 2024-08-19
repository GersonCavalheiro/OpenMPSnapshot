#pragma once

#include "bnmf_algs.hpp"


namespace benchmark {

template <typename T>
bnmf_algs::matrix_t<T> make_matrix(long x, long y, T beg, T scale) {
using namespace bnmf_algs;

matrix_t<T> res =
matrix_t<T>::Random(x, y) + matrix_t<T>::Constant(x, y, beg + 1);
res = res.array() * (scale / 2);

return res;
}


template <typename Scalar>
bnmf_algs::alloc_model::Params<Scalar> make_params(long x, long y, long z) {
bnmf_algs::alloc_model::Params<Scalar> params(
100, 1, std::vector<Scalar>(x, 0.05), std::vector<Scalar>(z, 10));
params.beta.back() = 60;

return params;
}
} 
