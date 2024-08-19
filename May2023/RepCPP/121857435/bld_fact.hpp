#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"

namespace bnmf_algs {


namespace bld {

template <typename T, typename Scalar>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bld_fact(const tensor_t<T, 3>& S,
const alloc_model::Params<Scalar>& model_params, double eps = 1e-50) {
auto x = static_cast<size_t>(S.dimension(0));
auto y = static_cast<size_t>(S.dimension(1));
auto z = static_cast<size_t>(S.dimension(2));

BNMF_ASSERT(model_params.alpha.size() == x,
"Number of alpha parameters must be equal to S.dimension(0)");
BNMF_ASSERT(model_params.beta.size() == z,
"Number of beta parameters must be equal to z");

tensor_t<T, 2> S_ipk = S.sum(shape<1>({1}));
tensor_t<T, 2> S_pjk = S.sum(shape<1>({0}));
tensor_t<T, 1> S_pjp = S.sum(shape<2>({0, 2}));

matrix_t<T> W(x, z);
matrix_t<T> H(z, y);
vector_t<T> L(y);

for (size_t i = 0; i < x; ++i) {
for (size_t k = 0; k < z; ++k) {
W(i, k) = model_params.alpha[i] + S_ipk(i, k) - 1;
}
}
for (size_t k = 0; k < z; ++k) {
for (size_t j = 0; j < y; ++j) {
H(k, j) = model_params.beta[k] + S_pjk(j, k) - 1;
}
}
for (size_t j = 0; j < y; ++j) {
L(j) = (model_params.a + S_pjp(j) - 1) / (model_params.b + 1 + eps);
}

vector_t<T> W_colsum =
W.colwise().sum() + vector_t<T>::Constant(W.cols(), eps);
vector_t<T> H_colsum =
H.colwise().sum() + vector_t<T>::Constant(H.cols(), eps);
W = W.array().rowwise() / W_colsum.array();
H = H.array().rowwise() / H_colsum.array();

return std::make_tuple(W, H, L);
}
} 
} 
